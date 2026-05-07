"""Public TikTok video download via yt-dlp.

Mirrors `insta.py` — same job shape, same post-process (h264 transcode if
needed, optional Whisper + burn). Different URL parser and output folder.

Files land in `youtube-analyzer/v1/tiktok/<video_id>/`. Short links
(`vm.tiktok.com/...`, `vt.tiktok.com/...`) are accepted; their shortcode
becomes the folder name even though yt-dlp resolves to the canonical
numeric ID for the filename — the difference doesn't matter, the user
operates on file_url not folder names."""
import re, subprocess, threading, time, urllib.parse
from pathlib import Path
from typing import Dict, Optional

import core
import subs_burn

V1 = Path(__file__).resolve().parent
TIKTOK_ROOT = V1 / "downloads" / "tiktok"
TIKTOK_ROOT.mkdir(parents=True, exist_ok=True)

_LOCK = threading.Lock()
JOBS: Dict[str, "TikTokJob"] = {}

# Canonical: tiktok.com/@user/video/<id> or /photo/<id>
# Embed:     tiktok.com/embed/v2/<id>
# Short:     vm.tiktok.com/<code>, vt.tiktok.com/<code>, m.tiktok.com/v/<id>
# T:         tiktok.com/t/<code>
_TIKTOK_RE = re.compile(
    r"tiktok\.com/(?:@[\w.-]+/(?:video|photo)/(?P<id>\d+)"
    r"|t/(?P<t>\w+)"
    r"|v/(?P<v>\d+)"
    r"|embed/v\d+/(?P<e>\d+))"
    r"|(?:vm|vt|m)\.tiktok\.com/(?P<short>\w+)")

_PROG_RE = re.compile(
    r"\[download\]\s+([\d.]+)%\s+of\s+~?\s*([\d.]+\s*\w+)"
    r"(?:\s+at\s+([\d.]+\s*\w+/s))?(?:\s+ETA\s+([\d:]+))?")
_DEST_RE = re.compile(r"^\[download\]\s+Destination:\s+(.+)$")
_MERGE_DEST_RE = re.compile(r"^\[Merger\]\s+Merging formats into\s+\"(.+)\"$")


def video_id_from_url(url: str) -> str:
    m = _TIKTOK_RE.search(url or "")
    if not m:
        raise ValueError("not a TikTok video URL")
    for key in ("id", "t", "v", "e", "short"):
        val = m.group(key)
        if val: return val
    raise ValueError("could not extract TikTok video ID")


class TikTokJob:
    def __init__(self, url: str, cfg: dict):
        self.url = url
        self.cfg = {
            "subtitles":     False,
            "burn_subs":     False,
            "subs_size":     "medium",
            "subs_color":    "white",
            "subs_bg":       "none",
            "subs_position": "bottom",
            **(cfg or {}),
        }
        self.video_id = video_id_from_url(url)
        self.id = f"tt_{self.video_id}_{int(time.time() * 1000)}"

        self.created_at = int(time.time())
        self.started_at: Optional[int] = None
        self.finished_at: Optional[int] = None
        self.status = "queued"   # queued | downloading | transcoding | transcribing | burning | done | failed | cancelled
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
            file_url = f"/tiktok/{urllib.parse.quote(self.file.name)}"
        return {
            "id":           self.id,
            "video_id":     self.video_id,
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
            log_path = TIKTOK_ROOT / f"{self.video_id}.ytdlp.log"
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
            if not (self.file and self.file.exists()):
                cands = sorted(
                    [p for p in TIKTOK_ROOT.iterdir()
                     if p.is_file() and p.suffix.lower() in
                     (".mp4", ".mov", ".mkv", ".webm")
                     and p.stem == self.video_id],
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
            # TikTok generally serves H.264, but the same VP9-in-mp4
            # situation that hits Instagram can show up here too — keep
            # the safety net so AE/Premiere always get a clean h264 file.
            try:
                prev_status = self.status
                self.status = "transcoding"
                self._ensure_h264()
                self.status = prev_status
                self.file_size = self.file.stat().st_size
            except Exception as e:
                self.error = f"h264 transcode failed: {e}"
            if self.cfg.get("subtitles"):
                try:
                    self.status = "transcribing"
                    self.phase_msg = "генерирую субтитры через Whisper"
                    subs_burn.generate_subs_via_whisper(
                        self.file,
                        on_phase=lambda m: setattr(self, "phase_msg", m))
                except Exception as e:
                    self.error = f"Whisper failed: {e}"
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
        try: self.file.unlink()
        except Exception: pass
        tmp.rename(self.file)
        self.phase_msg = ""

    def _build_cmd(self):
        out_tpl = str(TIKTOK_ROOT / "%(id)s.%(ext)s")
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
def start_tiktok(url: str, cfg: dict) -> TikTokJob:
    job = TikTokJob(url, cfg)
    job.start()
    return job

def get_job(job_id: str) -> Optional[TikTokJob]:
    with _LOCK:
        return JOBS.get(job_id)

def stop_job(job_id: str) -> Optional[TikTokJob]:
    job = get_job(job_id)
    if job: job.stop()
    return job

_SIDECAR_EXTS = (".vtt", ".srt", ".srv1", ".srv2", ".srv3", ".ttml",
                 ".json3", ".info.json", ".description", ".log")
_MEDIA_EXTS = (".mp4", ".mov", ".mkv", ".webm")

def _is_sidecar(p: Path) -> bool:
    name = p.name.lower()
    if name.endswith(".ytdlp.log"): return True
    return any(name.endswith(ext) for ext in _SIDECAR_EXTS)

def _delete_with_sidecars(media: Path):
    if not media.exists(): return
    parent = media.parent
    prefix = media.stem + "."
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
    if TIKTOK_ROOT.exists():
        for f in TIKTOK_ROOT.iterdir():
            if not f.is_file(): continue
            if _is_sidecar(f): continue
            if f.suffix.lower() not in _MEDIA_EXTS: continue
            if f.name in seen: continue
            try: stat = f.stat()
            except Exception: continue
            out.append({
                "id":          f"tt_disk:{f.name}",
                "video_id":    f.stem,
                "url":         f"https://www.tiktok.com/embed/v2/{f.stem}",
                "title":       f.stem,
                "status":      "done",
                "progress_pct":100.0,
                "file_url":    f"/tiktok/{urllib.parse.quote(f.name)}",
                "file_name":   f.name,
                "file_size_b": stat.st_size,
                "finished_at": int(stat.st_mtime),
            })
    out.sort(key=lambda j: -(j.get("finished_at") or j.get("created_at") or 0))
    return out

def delete_job(job_id: str) -> bool:
    target_file: Optional[Path] = None
    job = get_job(job_id)
    if job:
        job.stop()
        target_file = job.file
        with _LOCK:
            JOBS.pop(job_id, None)
    elif job_id.startswith("tt_disk:"):
        rest = job_id[len("tt_disk:"):]
        target_file = TIKTOK_ROOT / rest
    if target_file is None or not target_file.exists():
        return False
    try:
        _delete_with_sidecars(target_file)
        return True
    except Exception:
        return False
