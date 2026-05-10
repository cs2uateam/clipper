"""Pull viral-content ideas out of a downloaded video via Gemini Flash.

Pipeline: ffmpeg extracts 6 evenly-spaced keyframes → faster-whisper
transcribes audio (skipped if a `.vtt` sidecar already exists) → Gemini
2.0 Flash gets text + 6 inline images and returns a strict JSON with
hype score, TikTok captions, Shorts titles, and tags.

The user's API key is in-flight only: it arrives in the start-job
request, lives on the job object until the Gemini call completes, then
is wiped. Never written to disk, never logged."""
import base64, json, subprocess, threading, time, urllib.request, urllib.error
from pathlib import Path
from typing import Dict, Optional

import core
import subs_burn

V1 = Path(__file__).resolve().parent

_LOCK = threading.Lock()
JOBS: Dict[str, "IdeaJob"] = {}

# Free tier moves around — Google rotates which model is free.
# `gemini-2.5-flash` is the current free-tier-eligible vision model
# (Gemini 2.0 Flash was deprecated for free use). If Google moves it
# again, update here. List of currently-free models:
# https://ai.google.dev/gemini-api/docs/rate-limits#free-tier
GEMINI_URL = ("https://generativelanguage.googleapis.com/v1beta/"
              "models/gemini-2.5-flash:generateContent")

_PROMPT = """You are a viral-content strategist analyzing a short video for TikTok and YouTube Shorts.

You receive: the spoken transcript (may be empty if silent) and 6 keyframes sampled evenly from the clip.

Return a strict JSON object with these keys:
- hype_score: number 1-10 — how viral/hook-strong this clip is (be honest, mediocre clips get 4-5)
- hook: string — one sentence in English describing the single most attention-grabbing moment, referencing what's visible
- tiktok: array of EXACTLY 3 English captions for TikTok (each ≤100 chars, hook-driven, 1-2 emoji each, written like a real creator not a corporate intern)
- shorts: array of EXACTLY 3 English titles for YouTube Shorts (each ≤80 chars, click-driven, no clickbait lies, no emoji)
- long_caption: string — a longer English caption suitable for an Instagram/YouTube description field. 120-220 words, 2-4 short paragraphs separated by `\\n\\n`. Open with a hook, give context, end with a CTA inviting comment/follow/save. 1-3 emoji total, sprinkled (not stacked). NO hashtags inside this field — those go to `tags`.
- tags: array of EXACTLY 15 English hashtag words (no # prefix, no spaces, ranked most-relevant first)

Output ONLY the JSON, no preamble, no code fence."""

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "hype_score":   {"type": "number"},
        "hook":         {"type": "string"},
        "tiktok":       {"type": "array", "items": {"type": "string"}},
        "shorts":       {"type": "array", "items": {"type": "string"}},
        "long_caption": {"type": "string"},
        "tags":         {"type": "array", "items": {"type": "string"}},
    },
    "required": ["hype_score", "hook", "tiktok", "shorts",
                 "long_caption", "tags"],
}


def _cache_path(file_path: Path) -> Path:
    """Idea results are cached as `<stem>.idea.json` next to the media
    so a second click on the same video returns instantly without
    re-paying the Gemini cost. The user can force-regenerate via the
    `force` flag when they want a different take."""
    return file_path.with_suffix(".idea.json")


class IdeaJob:
    def __init__(self, file_path: Path, api_key: str,
                 force: bool = False, hint: str = ""):
        if not file_path.exists():
            raise ValueError(f"file missing: {file_path}")
        # If a hint was supplied, the user is refining — that always means
        # a fresh Gemini call (cache hit doesn't make sense). Treat as force.
        if hint and hint.strip():
            force = True
        # Allow empty api_key when a cache hit is possible — we won't
        # call Gemini in that case.
        if not force and _cache_path(file_path).exists():
            api_key = (api_key or "").strip()
        elif not api_key or not api_key.strip():
            raise ValueError("api_key empty")
        self.file = file_path
        self._api_key = (api_key or "").strip()
        self._force = force
        self._hint = (hint or "").strip()
        self.id = f"idea_{int(time.time() * 1000)}_{id(self) & 0xffff:04x}"

        self.created_at = int(time.time())
        self.started_at: Optional[int] = None
        self.finished_at: Optional[int] = None
        self.status = "queued"   # queued | extracting | transcribing | generating | done | failed
        self.phase_msg = ""
        self.error: Optional[str] = None
        self.result: Optional[dict] = None   # the parsed Gemini JSON
        self.from_cache: bool = False        # set when result loaded from sidecar

        self._thread: Optional[threading.Thread] = None
        self._proc: Optional[subprocess.Popen] = None

    def start(self):
        with _LOCK:
            JOBS[self.id] = self
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def status_dict(self) -> dict:
        return {
            "id":          self.id,
            "file_name":   self.file.name,
            "status":      self.status,
            "phase_msg":   self.phase_msg,
            "created_at":  self.created_at,
            "started_at":  self.started_at,
            "finished_at": self.finished_at,
            "result":      self.result,
            "from_cache":  self.from_cache,
            "error":       self.error,
        }

    # ---------- internal ----------
    def _run(self):
        try:
            self.started_at = int(time.time())
            cache = _cache_path(self.file)
            # 0. Cache hit — return instantly. `_force` skips this so
            # the user can ask for a fresh take.
            if cache.exists() and not self._force:
                try:
                    self.result = json.loads(cache.read_text(encoding="utf-8"))
                    self.from_cache = True
                    self.status = "done"
                    self.phase_msg = "из кэша"
                    return
                except Exception:
                    # Bad/corrupt cache — fall through and regenerate.
                    pass
            # 1. Pull 6 keyframes evenly across the clip duration.
            self.status = "extracting"
            self.phase_msg = "извлекаю 6 кадров"
            frames = self._extract_frames(n=6)
            if not frames:
                raise RuntimeError("no frames extracted")
            # 2. Transcript: prefer existing .vtt sidecar (Whisper or
            # platform-side captions), fall back to running Whisper.
            self.status = "transcribing"
            transcript = self._read_or_make_transcript()
            # 3. Gemini call. The key never survives this call.
            self.status = "generating"
            self.phase_msg = "Gemini думает"
            result = self._call_gemini(transcript, frames)
            self.result = result
            # Persist for next time — sidecar lives next to the media,
            # so deleting the media via the UI also drops the cache.
            try:
                cache.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2),
                    encoding="utf-8")
                # Hide the cache from File Explorer — it's a working
                # file, the user shouldn't have to look at it.
                core.hide_file(cache)
            except Exception:
                # Cache failure isn't fatal — user still gets the
                # result for this session.
                pass
            self.status = "done"
            self.phase_msg = ""
        except Exception as e:
            self.status = "failed"
            self.error = f"{type(e).__name__}: {e}"
        finally:
            # Wipe the key from RAM the moment the job ends — covers both
            # success and failure paths.
            self._api_key = ""
            self.finished_at = int(time.time())

    def _video_duration(self) -> float:
        out = subprocess.run(
            [core.FFPROBE, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(self.file)],
            capture_output=True, text=True)
        try:
            return float((out.stdout or "0").strip())
        except ValueError:
            return 0.0

    def _extract_frames(self, n: int = 6) -> list:
        """n frames evenly spaced across the duration; returned as
        list of bytes (PNG each). Stays in RAM — no temp files."""
        dur = self._video_duration()
        if dur <= 0: return []
        # Sample at fractions 1/(n+1), 2/(n+1), …, n/(n+1) so we never
        # land on the very first or last frame (often black).
        timestamps = [dur * (i + 1) / (n + 1) for i in range(n)]
        frames = []
        for t in timestamps:
            cmd = [core.FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
                   "-ss", f"{t:.2f}", "-i", str(self.file),
                   "-frames:v", "1",
                   # Cap each frame at 1280px wide to keep payload small —
                   # Gemini gets plenty of detail at this size, and we
                   # stay well under the 20MB request limit.
                   "-vf", "scale='min(1280,iw)':-2",
                   "-f", "image2", "-c:v", "png", "pipe:1"]
            p = subprocess.run(cmd, capture_output=True)
            if p.returncode == 0 and p.stdout:
                frames.append(p.stdout)
        return frames

    def _read_or_make_transcript(self) -> str:
        # Look for an existing .vtt sidecar next to the media file.
        existing = subs_burn.find_subtitle_files(self.file)
        if existing:
            text = self._vtt_to_plain(existing[0])
            if text.strip():
                self.phase_msg = f"использую субтитры {existing[0].name}"
                return text
        # No captions on disk — generate via Whisper. Slowest step on
        # videos longer than ~30s; for 15s reels it's near-instant.
        self.phase_msg = "Whisper транскрибирует"
        try:
            subs_burn.generate_subs_via_whisper(
                self.file,
                on_phase=lambda m: setattr(self, "phase_msg", m))
        except Exception as e:
            # Whisper failure isn't fatal — Gemini can still describe
            # what's on screen from the frames alone.
            self.phase_msg = f"Whisper упал: {e} — работаю только по кадрам"
            return ""
        again = subs_burn.find_subtitle_files(self.file)
        if again:
            return self._vtt_to_plain(again[0])
        return ""

    @staticmethod
    def _vtt_to_plain(vtt: Path) -> str:
        """Strip VTT timing/formatting → join cue text into one string."""
        lines = []
        try:
            for raw in vtt.read_text("utf-8", errors="ignore").splitlines():
                s = raw.strip()
                if not s: continue
                if s.upper().startswith("WEBVTT"): continue
                if "-->" in s: continue
                if s.isdigit(): continue
                # Drop inline timing tags <00:00:01.000>
                import re as _re
                s = _re.sub(r"<[^>]+>", "", s)
                lines.append(s)
        except Exception:
            return ""
        return " ".join(lines).strip()

    def _call_gemini(self, transcript: str, frames: list) -> dict:
        # Build the multipart request: one text block, then N inline
        # images. Gemini's `inline_data` wants base64 strings.
        prompt_text = _PROMPT
        # If the user is refining a previous result, splice the prior
        # output + their hint into the prompt so Gemini knows what to
        # change rather than starting from scratch.
        if self._hint:
            prev = ""
            cache = _cache_path(self.file)
            if cache.exists():
                try: prev = cache.read_text(encoding="utf-8")
                except Exception: pass
            refine_block = (
                "\n\n=== USER REFINEMENT ===\n"
                "The user already saw a previous output and wants you to "
                "redo it with this guidance:\n"
                f"  {self._hint}\n"
                "Respect the new direction. Keep the same JSON schema and "
                "all required keys.\n")
            if prev:
                refine_block += "\nPrevious output (for context — do not just copy):\n" + prev
            prompt_text += refine_block
        parts = [{"text": prompt_text
                  + "\n\n=== TRANSCRIPT ===\n"
                  + (transcript or "(silent — no speech detected)")}]
        for img in frames:
            parts.append({
                "inline_data": {
                    "mime_type": "image/png",
                    "data": base64.b64encode(img).decode("ascii"),
                }
            })
        body = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": _RESPONSE_SCHEMA,
                "temperature": 0.9,
            },
        }
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            GEMINI_URL,
            data=data,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self._api_key,
            },
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_body = ""
            try: err_body = e.read().decode("utf-8", errors="ignore")[:500]
            except Exception: pass
            raise RuntimeError(f"Gemini HTTP {e.code}: {err_body or e.reason}")
        try:
            text = payload["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError):
            raise RuntimeError(
                f"Gemini returned unexpected payload: "
                f"{json.dumps(payload)[:300]}")
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Gemini returned non-JSON: {e}; raw: {text[:300]}")


# ---------- module-level helpers ----------
def start_idea(file_path: Path, api_key: str,
               force: bool = False, hint: str = "") -> IdeaJob:
    job = IdeaJob(file_path, api_key, force=force, hint=hint)
    job.start()
    return job

def has_cached(file_path: Path) -> bool:
    return _cache_path(file_path).exists()

def get_job(job_id: str) -> Optional[IdeaJob]:
    with _LOCK:
        return JOBS.get(job_id)

def drop_job(job_id: str) -> bool:
    with _LOCK:
        return JOBS.pop(job_id, None) is not None
