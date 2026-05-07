"""Public Instagram reel download via yt-dlp.

Mirrors `download.py` but stripped to the bare minimum: there's nothing to
configure on a public reel — single MP4 stream, fixed resolution from the
server, no time-sectioning (reels are ≤90s, you take the whole thing).
Optional post-process: Whisper transcribe → burn subs (shared with the
upload tab via `subs_burn`).

Files land in `youtube-analyzer/v1/insta/<reel_id>/`. No cookies, no
authenticated content — anything that 403s is just out of scope."""
import re, subprocess, threading, time, urllib.parse
from pathlib import Path
from typing import Dict, Optional

import core
import subs_burn

V1 = Path(__file__).resolve().parent
INSTA_ROOT = V1 / "downloads" / "insta"
INSTA_ROOT.mkdir(parents=True, exist_ok=True)

_LOCK = threading.Lock()
JOBS: Dict[str, "InstaJob"] = {}

# Reel / post / IGTV — all carry a short shortcode after the URL kind.
# Instagram occasionally redirects /p/ to /reel/ for videos; both work
# as input here because yt-dlp resolves either to the same media.
_REEL_RE = re.compile(
    r"instagram\.com/(?:reel|reels|p|tv)/([A-Za-z0-9_-]+)")

_PROG_RE = re.compile(
    r"\[download\]\s+([\d.]+)%\s+of\s+~?\s*([\d.]+\s*\w+)"
    r"(?:\s+at\s+([\d.]+\s*\w+/s))?(?:\s+ETA\s+([\d:]+))?")
_DEST_RE = re.compile(r"^\[download\]\s+Destination:\s+(.+)$")
_MERGE_DEST_RE = re.compile(r"^\[Merger\]\s+Merging formats into\s+\"(.+)\"$")


def reel_id_from_url(url: str) -> str:
    m = _REEL_RE.search(url or "")
    if not m:
        raise ValueError("not an Instagram reel/post URL")
    return m.group(1)


class InstaJob:
    def __init__(self, url: str, cfg: dict):
        self.url = url
        self.cfg = {
            "subtitles":     False,      # off by default — reels are short
            "burn_subs":     False,
            "subs_size":     "medium",
            "subs_color":    "white",
            "subs_bg":       "none",
            "subs_position": "bottom",
            **(cfg or {}),
        }
        self.reel_id = reel_id_from_url(url)
        self.id = f"ig_{self.reel_id}_{int(time.time() * 1000)}"

        self.created_at = int(time.time())
        self.started_at: Optional[int] = None
        self.finished_at: Optional[int] = None
        self.status = "queued"   # queued | downloading | transcribing | burning | done | failed | cancelled
        self.progress_pct = 0.0
        self.size_total = ""
        self.speed = ""
        self.eta = ""
        self.title = ""
        self.file: Optional[Path] = None
        self.file_size: Optional[int] = None
        self.error: Optional[str] = None
        self.phase_msg = ""

        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None

    # ---------- public ----------
    def start(self):
        with _LOCK:
            JOBS[self.id] = self
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        if self._proc and self._proc.poll() is None:
            try: self._proc.terminate()
            except Exception: pass
        if self.status in ("queued", "downloading", "transcoding",
                           "transcribing", "burning"):
            self.status = "cancelled"

    def status_dict(self) -> dict:
        file_url = None
        if self.file:
            file_url = f"/insta/{urllib.parse.quote(self.file.name)}"
        return {
            "id":           self.id,
            "reel_id":      self.reel_id,
            "url":          self.url,
            "title":        self.title,
            "status":       self.status,
            "progress_pct": round(self.progress_pct, 1),
            "size_total":   self.size_total,
            "speed":        self.speed,
            "eta":          self.eta,
            "phase_msg":    self.phase_msg,
            "created_at":   self.created_at,
            "started_at":   self.started_at,
            "finished_at":  self.finished_at,
            "file_url":     file_url,
            "file_name":    self.file.name if self.file else None,
            "file_size_b":  self.file_size,
            "error":        self.error,
            "cfg":          self.cfg,
        }

    # ---------- internal ----------
    def _run(self):
        try:
            self.started_at = int(time.time())
            self.status = "downloading"
            cmd = self._build_cmd()
            # Per-job log filename — concurrent downloads to the same flat
            # folder must not stomp each other's stderr output.
            log_path = INSTA_ROOT / f"{self.reel_id}.ytdlp.log"
            try: log_path.unlink()
            except Exception: pass
            log_handle = open(log_path, "ab", buffering=0)
            self._proc = subprocess.Popen(
                cmd, stdout=log_handle, stderr=subprocess.STDOUT)
            last_off = 0
            while True:
                try: cur = log_path.stat().st_size
                except FileNotFoundError: cur = 0
                if cur > last_off:
                    with open(log_path, "rb") as f:
                        f.seek(last_off)
                        chunk = f.read(cur - last_off)
                    last_off = cur
                    for raw in chunk.replace(b"\r", b"\n").split(b"\n"):
                        if not raw: continue
                        self._parse_line(
                            raw.decode("utf-8", errors="ignore").rstrip())
                if self._proc.poll() is not None and cur == last_off:
                    break
                time.sleep(0.4)
            try: log_handle.close()
            except Exception: pass
            self._proc.wait()
            if self._proc.returncode != 0:
                if self.status != "cancelled":
                    self.status = "failed"
                    if not self.error:
                        self.error = (f"yt-dlp exit {self._proc.returncode} "
                                      f"— see {log_path.name}")
                self.finished_at = int(time.time())
                return
            # Locate the downloaded media file. yt-dlp may have emitted a
            # Destination line we already captured; if not, scan the
            # flat root for a media file matching this reel's id.
            if not (self.file and self.file.exists()):
                cands = sorted(
                    [p for p in INSTA_ROOT.iterdir()
                     if p.is_file() and p.suffix.lower() in
                     (".mp4", ".mov", ".mkv", ".webm")
                     and p.stem == self.reel_id],
                    key=lambda p: -p.stat().st_mtime)
                if cands:
                    self.file = cands[0]
            if not (self.file and self.file.exists()):
                self.status = "failed"
                self.error = "yt-dlp finished but no media file found"
                self.finished_at = int(time.time())
                return
            self.file_size = self.file.stat().st_size
            try: log_path.unlink()
            except Exception: pass
            # Instagram sometimes serves reels as VP9-in-MP4. The
            # container says mp4 so yt-dlp accepts it without remuxing,
            # but VP9 inside mp4 isn't a format After Effects / Premiere
            # / most NLEs accept natively — they need H.264. Transcode
            # if needed; cheap no-op (single ffprobe) for h264 sources.
            try:
                prev_status = self.status
                self.status = "transcoding"
                self._ensure_h264()
                self.status = prev_status
                self.file_size = self.file.stat().st_size
            except Exception as e:
                self.error = f"h264 transcode failed: {e}"
            # Optional Whisper transcription. Instagram never serves
            # platform-side captions for reels → Whisper is the only
            # path. Off by default because reels are short and the user
            # often just wants the raw clip.
            if self.cfg.get("subtitles"):
                try:
                    self.status = "transcribing"
                    self.phase_msg = "генерирую субтитры через Whisper"
                    subs_burn.generate_subs_via_whisper(
                        self.file,
                        on_phase=lambda m: setattr(self, "phase_msg", m))
                except Exception as e:
                    self.error = f"Whisper failed: {e}"
            # Optional burn. Same module as download/upload tabs — fix
            # there propagates everywhere.
            if (self.cfg.get("burn_subs") and self.cfg.get("subtitles")
                    and self.file):
                try:
                    self.status = "burning"
                    self.phase_msg = "вшиваю субтитры в видео — re-encoding"
                    subs_burn.burn_subtitles(
                        self.file, self.cfg,
                        on_phase=lambda m: setattr(self, "phase_msg", m),
                        on_proc=lambda p: setattr(self, "_proc", p))
                    self.file_size = self.file.stat().st_size
                except Exception as e:
                    self.error = f"burn-subs failed: {e}"
            self.status = "done"
            self.phase_msg = ""
            self.finished_at = int(time.time())
        except Exception as e:
            if self.status != "cancelled":
                self.status = "failed"
                self.error = f"{type(e).__name__}: {e}"
            self.finished_at = int(time.time())

    def _ensure_h264(self):
        """If the downloaded video isn't H.264, re-encode it. Instagram
        serves some reels as VP9 inside mp4 — that breaks AE/Premiere
        even though the file extension says .mp4."""
        if not (self.file and self.file.exists()): return
        probe = subprocess.run(
            [core.FFPROBE, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name", "-of",
             "default=noprint_wrappers=1:nokey=1", str(self.file)],
            capture_output=True, text=True)
        codec = (probe.stdout or "").strip().lower()
        if codec in ("h264", "avc1"):
            return
        self.phase_msg = f"перекодирую {codec}→h264 (для AE/Premiere)"
        tmp = self.file.with_name(self.file.stem + ".h264.mp4")
        cmd = [core.FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
               "-i", str(self.file),
               "-c:v", "libx264", "-preset", "fast", "-crf", "18",
               "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-b:a", "192k",
               "-movflags", "+faststart",
               str(tmp)]
        p = subprocess.Popen(cmd)
        self._proc = p
        p.wait()
        self._proc = None
        if p.returncode != 0 or not tmp.exists():
            try: tmp.unlink()
            except Exception: pass
            raise RuntimeError(f"ffmpeg exit {p.returncode}")
        # Replace original with the h264 copy.
        try: self.file.unlink()
        except Exception: pass
        tmp.rename(self.file)
        self.phase_msg = ""

    def _build_cmd(self):
        # Flat layout: file lands directly in INSTA_ROOT.
        out_tpl = str(INSTA_ROOT / "%(id)s.%(ext)s")
        cmd = [core.YTDLP, "--newline", "--no-warnings",
               "--ffmpeg-location", core.FFMPEG,
               "-o", out_tpl,
               "--restrict-filenames",
               "--merge-output-format", "mp4",
               self.url]
        return cmd

    def _parse_line(self, line: str):
        m = _PROG_RE.search(line)
        if m:
            try: self.progress_pct = float(m.group(1))
            except: pass
            self.size_total = (m.group(2) or "").strip()
            self.speed = (m.group(3) or "").strip()
            self.eta = (m.group(4) or "").strip()
            return
        m = _DEST_RE.match(line)
        if m:
            p = Path(m.group(1))
            if p.suffix.lower() in (".mp4", ".mov", ".mkv", ".webm"):
                self.file = p
                if not self.title: self.title = p.stem
            return
        m = _MERGE_DEST_RE.match(line)
        if m:
            self.file = Path(m.group(1))
            if not self.title: self.title = self.file.stem
            return
        low = line.lower()
        if any(k in low for k in ("error:", "unsupported url", "private",
                                    "login required", "rate-limit",
                                    "video unavailable", "404")):
            if not self.error:
                self.error = line.strip()[:200]


# ---------- module-level helpers ----------
def start_insta(url: str, cfg: dict) -> InstaJob:
    job = InstaJob(url, cfg)
    job.start()
    return job

def get_job(job_id: str) -> Optional[InstaJob]:
    with _LOCK:
        return JOBS.get(job_id)

def stop_job(job_id: str) -> Optional[InstaJob]:
    job = get_job(job_id)
    if job: job.stop()
    return job

# Sidecar files (logs, captions, metadata) sit next to the media in the
# flat root and must be hidden from the user-facing list.
_SIDECAR_EXTS = (".vtt", ".srt", ".srv1", ".srv2", ".srv3", ".ttml",
                 ".json3", ".info.json", ".description", ".log")
_MEDIA_EXTS = (".mp4", ".mov", ".mkv", ".webm")

def _is_sidecar(p: Path) -> bool:
    name = p.name.lower()
    # Per-job yt-dlp logs land as `<reel_id>.ytdlp.log` in the flat root.
    if name.endswith(".ytdlp.log"): return True
    return any(name.endswith(ext) for ext in _SIDECAR_EXTS)

def _delete_with_sidecars(media: Path):
    """Remove media file + every sidecar sharing its stem (`<stem>.*`).
    Used by both completion-cleanup and user-driven delete."""
    if not media.exists(): return
    parent = media.parent
    prefix = media.stem + "."   # matches <stem>.en.vtt, <stem>.ytdlp.log…
    try: media.unlink()
    except Exception: pass
    for sib in parent.iterdir():
        if not sib.is_file(): continue
        if sib.name.startswith(prefix):
            try: sib.unlink()
            except Exception: pass

def list_jobs() -> list:
    with _LOCK:
        live = [j.status_dict() for j in JOBS.values()]
    seen = {j.get("file_name") for j in live if j.get("file_name")}
    out = list(live)
    if INSTA_ROOT.exists():
        for f in INSTA_ROOT.iterdir():
            if not f.is_file(): continue
            if _is_sidecar(f): continue
            if f.suffix.lower() not in _MEDIA_EXTS: continue
            if f.name in seen: continue
            try: stat = f.stat()
            except Exception: continue
            out.append({
                "id":          f"ig_disk:{f.name}",
                "reel_id":     f.stem,
                "url":         f"https://www.instagram.com/reel/{f.stem}/",
                "title":       f.stem,
                "status":      "done",
                "progress_pct":100.0,
                "file_url":    f"/insta/{urllib.parse.quote(f.name)}",
                "file_name":   f.name,
                "file_size_b": stat.st_size,
                "finished_at": int(stat.st_mtime),
            })
    out.sort(key=lambda j: -(j.get("finished_at") or j.get("created_at") or 0))
    return out

def delete_job(job_id: str) -> bool:
    """Remove the media file plus any sidecars sharing its stem."""
    target_file: Optional[Path] = None
    job = get_job(job_id)
    if job:
        job.stop()
        target_file = job.file
        with _LOCK:
            JOBS.pop(job_id, None)
    elif job_id.startswith("ig_disk:"):
        rest = job_id[len("ig_disk:"):]
        target_file = INSTA_ROOT / rest
    if target_file is None or not target_file.exists():
        return False
    try:
        _delete_with_sidecars(target_file)
        return True
    except Exception:
        return False
