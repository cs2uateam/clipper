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
- long_caption: string — the COMPLETE Instagram post body, ready to paste into the description field. Must follow this exact structure (use real `\\n` line breaks between sections):

    Line 1: short hooky header in English with one emoji and a `👇` arrow at the end (max ~60 chars).
    Blank line.
    Body: EXACTLY 5 paragraphs in English, each 70-110 words, separated by ONE blank line. Each paragraph develops a distinct angle (context, what changed, why it matters, deeper layer, takeaway). No clickbait, no fake hype. Maximum 3 emoji TOTAL across the body, sprinkled (not stacked). NO hashtags inside the body. NO bullet lists inside the body.
    Blank line.
    A line containing literally `🔑 Keywords`.
    Then 12-14 plain keyword lines, one per line, NO `#` prefix, NO commas. Mix specific (CS2 ranked match, CS2 smoke play) and broad (PC FPS gaming, esports gameplay) — ranked most-relevant first.
    Blank line.
    Final single line: EXACTLY 5 hashtags separated by single spaces, each starts with `#`, no spaces inside hashtags.

  CRITICAL — these 5 hashtags drive niche discoverability. Generic safe tags are FORBIDDEN. You MUST reason from the actual content.

  TIER STRUCTURE (use this exact order):
    1. HYPER-NICHE — combine concrete content nouns into a compound tag.
       Format examples: `#donk1v4`, `#m0NESYflick`, `#mirageDeagleAce`, `#nukeRetake`,
       `#zywooAWPshot`, `#s1mpleClutch`, `#majorGrandFinal`.
       This tag MUST cite a SPECIFIC element visible/heard in this exact clip.
    2. MICRO-NICHE — single specific noun from the clip: a player name, map name,
       weapon name, or situation type. Examples: `#mirage`, `#deagleAce`, `#1v4Clutch`,
       `#noscope`, `#wallbang`, `#ninjaDefuse`.
    3. GAME-SPECIFIC — `#CS2` or `#CounterStrike2` (only one of these).
    4. GAME-BROAD — `#CounterStrike` or `#CSGO` (only if older clip, otherwise pick
       another micro-niche or skip to a content-angle tag like `#proPlay`, `#tier1cs`,
       `#majorMoment`).
    5. ANGLE TAG — describes the CONTENT TYPE not the platform: `#fragMovie`,
       `#clutchHighlight`, `#aceMontage`, `#proAnalysis`, `#esportsHighlight`.

  CS2-TAXONOMY (use as a lookup when scanning transcript/keyframes — do not invent
  names that aren't actually there):
    Players: donk, m0NESY, ZywOo, s1mple, NiKo, sh1ro, FalleN, broky, ropz, Twistzz,
             NAF, device, electronic, dupreeh, Magisk, frozen, b1t, jL, w0nderful,
             KSCERATO, yuurih, malbsMd, hampus, jks, EliGE, nitr0.
    Teams:   FaZe, NAVI, G2, Vitality, Spirit, MOUZ, Astralis, Heroic, Liquid, Cloud9,
             FURIA, ENCE, BIG, Eternal Fire, paiN, MIBR, Imperial.
    Maps:    Mirage, Inferno, Nuke, Dust2, Ancient, Anubis, Vertigo, Train, Overpass.
    Weapons: AWP, Deagle, USP, M4A4, M4A1S, AK47, Tec9, FiveSeven, Scout, Krieg, Negev.
    Situations: ace, clutch, 1v2, 1v3, 1v4, 1v5, noscope, wallbang, ninjaDefuse,
                triple, quadra, prefire, flick, retake, eco, antiEco, pistolRound.
    Tournaments: Major, IEM, BLAST, ESL Pro, PGL, Cologne, Katowice, Paris, Copenhagen.

  ABSOLUTELY FORBIDDEN tags (do not use these — they add zero discoverability signal):
    #FPSGaming, #PCGaming, #VideoGames, #Gamer, #Gaming, #GamingContent, #GamingReels,
    #Esports (alone, without context), #Game, #PC, #viral, #fyp, #foryou, #trending,
    #shorts, #reels, #explore, #insta, #instagram.

  EXAMPLES of good vs bad output for a clip showing donk's 1v4 on Mirage with a Deagle:
    BAD (rejected):   `#CS2 #CounterStrike2 #FPSGaming #PCGaming #Esports`
    GOOD (accepted):  `#donk1v4 #mirageDeagle #CS2 #majorMoment #clutchHighlight`

  Total length: ~500-650 words. Output as ONE STRING with embedded newlines, NOT an object.

- tags: array of EXACTLY 5 English hashtag words (no # prefix, no spaces) — MUST be the same 5 tags as the final hashtag line of long_caption, in the same order. Same tier structure, same forbidden list.

Output ONLY the JSON, no preamble, no code fence."""

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "hype_score":   {"type": "number"},
        "hook":         {"type": "string"},
        # Explicit length bounds — Gemini в structured-output режиме
        # принимает schema как hard-constraint. Без minItems/maxItems
        # модель свободно расширяет массивы ("EXACTLY N" в текстовом
        # промпте часто игнорируется, особенно для tags).
        "tiktok": {"type": "array", "items": {"type": "string"},
                   "minItems": 3, "maxItems": 3},
        "shorts": {"type": "array", "items": {"type": "string"},
                   "minItems": 3, "maxItems": 3},
        "long_caption": {"type": "string"},
        "tags":   {"type": "array", "items": {"type": "string"},
                   "minItems": 5, "maxItems": 5},
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


def prepare_inputs(file_path: Path, n_frames: int = 0) -> dict:
    """Run the EXTRACT + TRANSCRIBE half of the idea pipeline and stop —
    no Gemini call. Used by external agents (curator) that want to write
    their OWN caption using Claude/whatever, not Gemini.

    `n_frames=0` → auto: clamp(12, dur_sec / 2, 30). Floor 12 keeps short
    reels dense (every beat visible); ~1 frame per 2 sec in the middle;
    cap 30 prevents long matches from blowing up the caller's vision
    budget. Explicit positive `n_frames` overrides the formula.

    Frames are written into a sibling `.frames/<stem>/fN.jpg` cache
    folder so they don't pollute the user's "Open folder" view — the
    media file stays alone in its parent directory."""
    if not file_path.exists():
        raise ValueError(f"file missing: {file_path}")
    # 1. Frames — JPEG cache in a per-media subfolder.
    probe = subprocess.run(
        [core.FFPROBE, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(file_path)],
        capture_output=True, text=True)
    try:
        dur = float((probe.stdout or "0").strip())
    except ValueError:
        dur = 0.0
    if dur <= 0:
        raise RuntimeError("ffprobe failed to read duration")
    if n_frames <= 0:
        n_frames = max(12, min(30, int(dur / 2)))
    frames_dir = file_path.parent / ".frames" / file_path.stem
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = []
    for i in range(1, n_frames + 1):
        t = dur * i / (n_frames + 1)
        out_p = frames_dir / f"f{i}.jpg"
        r = subprocess.run(
            [core.FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
             "-ss", f"{t:.2f}", "-i", str(file_path),
             "-vframes", "1", "-vf", "scale='min(1280,iw)':-2",
             "-q:v", "4", str(out_p)],
            capture_output=True, timeout=30)
        if r.returncode == 0 and out_p.exists():
            frame_paths.append(str(out_p.resolve()))
    if not frame_paths:
        raise RuntimeError("frame extraction yielded nothing")
    # 2. Transcript — prefer existing .vtt sidecar, fall back to Whisper.
    transcript = ""
    existing = subs_burn.find_subtitle_files(file_path)
    if existing:
        transcript = _vtt_to_plain(existing[0])
    if not transcript.strip():
        try:
            subs_burn.generate_subs_via_whisper(file_path)
            again = subs_burn.find_subtitle_files(file_path)
            if again:
                transcript = _vtt_to_plain(again[0])
        except Exception:
            # Silent clips are fine — agent can describe from frames alone.
            transcript = ""
    return {
        "transcript": transcript,
        "frames":     frame_paths,
        "duration_s": round(dur, 2),
    }


def _vtt_to_plain(vtt: Path) -> str:
    """Standalone copy of IdeaJob._vtt_to_plain so prepare_inputs doesn't
    need a job instance."""
    import re as _re
    lines = []
    try:
        for raw in vtt.read_text("utf-8", errors="ignore").splitlines():
            s = raw.strip()
            if not s: continue
            if s.upper().startswith("WEBVTT"): continue
            if "-->" in s: continue
            if s.isdigit(): continue
            s = _re.sub(r"<[^>]+>", "", s)
            lines.append(s)
    except Exception:
        return ""
    return " ".join(lines).strip()
