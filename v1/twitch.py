"""Twitch VOD download via yt-dlp.

Mirrors `download.py` for YouTube. Supports the archived-broadcast case
(`twitch.tv/videos/12345`) — the most common Twitch link a creator saves
to clip highlights from later. No live-stream / DVR seek (see clipper's
YouTube «Стрим» tab pattern if that ever needs to be added). No captions
(Twitch VODs don't ship burn-ready CC tracks).

Files land in `clipper/v1/downloads/twitch/<vid>[_startS-endS].<ext>` +
sidecars (`.ytdlp.log` on failure, `.info.json` when yt-dlp emits it).
"""
import re, shutil, subprocess, threading, time, urllib.parse
from pathlib import Path
from typing import Dict, Optional

import core
import subs_burn

V1 = Path(__file__).resolve().parent
TWITCH_ROOT = V1 / "downloads" / "twitch"
TWITCH_ROOT.mkdir(parents=True, exist_ok=True)

_LOCK = threading.Lock()
JOBS: Dict[str, "TwitchJob"] = {}

# Only VOD URLs supported here: /videos/<numeric_id>. Clips and live
# channels are OUT OF SCOPE for this module (user picked "VOD only").
# Note: `www.` optional; `m.` mobile subdomain also matches.
_VOD_RE = re.compile(
    r"twitch\.tv/videos/(\d+)")

_PROG_RE = re.compile(
    r"\[download\]\s+([\d.]+)%\s+of\s+~?\s*([\d.]+\s*\w+)"
    r"(?:\s+at\s+([\d.]+\s*\w+/s))?(?:\s+ETA\s+([\d:]+))?"
)
_DEST_RE = re.compile(r"^\[download\]\s+Destination:\s+(.+)$")
_MERGE_DEST_RE = re.compile(r"^\[Merger\]\s+Merging formats into\s+\"(.+)\"$")
_FFMPEG_TIME_RE = re.compile(
    r"time=(\d{2}):(\d{2}):(\d{2})(?:\.(\d{2,3}))?")
_SECTION_RE = re.compile(
    r"Downloading 1 time ranges?:\s*([\d.]+)-([\d.]+)")


def _hms(sec: int) -> str:
    """Seconds → 'HH:MM:SS' for yt-dlp --download-sections."""
    h, r = divmod(int(sec), 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def vod_id_from_url(url: str) -> str:
    """Extract the numeric VOD id from a Twitch VOD URL. Raises
    ValueError if the URL isn't a twitch.tv/videos/<n> link."""
    m = _VOD_RE.search(url or "")
    if not m:
        raise ValueError("not a Twitch VOD URL (expected twitch.tv/videos/<id>)")
    return m.group(1)


class TwitchJob:
    def __init__(self, url: str, cfg: dict):
        self.url = url
        self.cfg = {
            "quality":      "1080",       # 720 / 1080 / 1440 / 2160 / best
            "audio_only":   False,
            "audio_format": "mp3",        # mp3 / m4a (only if audio_only)
            "start_t":      None,         # seconds since VOD start (optional)
            "end_t":        None,         # seconds since VOD start (optional)
            # Watermark — empty `watermark_name` (or position=none) disables.
            "watermark_name":     "",
            "watermark_path":     "",
            "watermark_size":     20,
            "watermark_opacity":  70,
            "watermark_position": "center",
            **(cfg or {}),
        }
        # Strict URL check — reject before spawning a doomed subprocess.
        # yt-dlp would fail on non-Twitch URLs anyway, but that failure
        # happens minutes later and the user sees a queued/downloading job
        # for a while. Raise here → app.py converts to HTTP 400 upfront.
        self.vid = vod_id_from_url(url)
        self.id = f"tw_{self.vid}_{int(time.time() * 1000)}"

        self.created_at = int(time.time())
        self.started_at: Optional[int] = None
        self.finished_at: Optional[int] = None
        self.status = "queued"           # queued | downloading | burning | done | failed | cancelled
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
        self._section_start = 0.0
        self._section_end = 0.0
        self._section_duration = 0.0
        # Output stem: `<vid>` for full downloads, `<vid>_<start>-<end>`
        # for sectioned. Sectioned-distinct names avoid yt-dlp thinking a
        # different range is "already downloaded" and returning stale bytes.
        s_t = self.cfg.get("start_t"); e_t = self.cfg.get("end_t")
        if s_t is not None or e_t is not None:
            s_tag = int(s_t or 0)
            e_tag = int(e_t) if e_t is not None else 0
            self._stem = f"{self.vid}_{s_tag}-{e_tag}"
        else:
            self._stem = self.vid

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
        if self.status in ("queued", "downloading"):
            self.status = "cancelled"

    def status_dict(self) -> dict:
        file_url = None
        if self.file:
            file_url = f"/downloads/twitch/{urllib.parse.quote(self.file.name)}"
        return {
            "id":           self.id,
            "vid":          self.vid,
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
            print(f"[twitch] cfg: quality={self.cfg.get('quality')!r} "
                  f"audio_only={self.cfg.get('audio_only')} "
                  f"start={self.cfg.get('start_t')} end={self.cfg.get('end_t')}",
                  flush=True)
            cmd = self._build_cmd()
            log_path = TWITCH_ROOT / f"{self._stem}.ytdlp.log"
            try: log_path.unlink()
            except Exception: pass
            log_handle = open(log_path, "ab", buffering=0)
            self._proc = subprocess.Popen(
                cmd, stdout=log_handle, stderr=subprocess.STDOUT)
            # Tail the log file for progress lines.
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
            if self._proc.returncode == 0:
                if self.file and self.file.exists():
                    self.file_size = self.file.stat().st_size
                else:
                    candidates = sorted(
                        [p for p in TWITCH_ROOT.iterdir()
                         if p.is_file() and p.stem == self._stem
                         and p.suffix.lower() in (".mp4",".m4a",".mp3",".webm",".mkv",".mov")],
                        key=lambda p: -p.stat().st_mtime)
                    if candidates:
                        self.file = candidates[0]
                        self.file_size = self.file.stat().st_size
                # Drop the log on success — kept on failure for diagnostics.
                try: log_path.unlink()
                except Exception: pass
                # Optional watermark composite (single ffmpeg pass).
                if (not self.cfg.get("audio_only") and self.file
                        and self.cfg.get("watermark_name")):
                    try:
                        self.status = "burning"
                        subs_burn.apply_watermark(
                            self.file, self.cfg,
                            on_phase=lambda m: setattr(self, "phase_msg", m),
                            on_proc=lambda p: setattr(self, "_proc", p))
                        self.file_size = self.file.stat().st_size
                    except Exception as e:
                        self.error = f"watermark failed: {e}"
                self.status = "done"
            else:
                if self.status != "cancelled":
                    self.status = "failed"
                    if not self.error:
                        self.error = f"yt-dlp exit {self._proc.returncode}"
            self.finished_at = int(time.time())
        except Exception as e:
            self.status = "failed"
            self.error = f"download crashed: {e}"
            self.finished_at = int(time.time())

    def _build_cmd(self):
        out_tpl = str(TWITCH_ROOT / f"{self._stem}.%(ext)s")
        cmd = [core.YTDLP, "--newline", "--no-warnings",
               "--ffmpeg-location", core.FFMPEG,
               "-o", out_tpl,
               "--no-playlist",
               "--restrict-filenames"]
        if self.cfg.get("audio_only"):
            af = (self.cfg.get("audio_format") or "mp3").lower()
            cmd += ["-x", "--audio-format", af, "--audio-quality", "0"]
        else:
            q = (self.cfg.get("quality") or "1080").lower()
            # Twitch VOD HLS variants come out as "best" / "1080p60" /
            # "720p60" / "480p" / "360p" / "160p". yt-dlp's height-cap
            # selectors work fine on those. Prefer combined `best[h<=N]`
            # (HLS single-stream, seekable) — same reason as YouTube DL:
            # section-cutting via ffmpeg-seek-into-single-stream is
            # robust; DASH-style split streams deadlock on some CDNs.
            if q in ("best", "max"):
                fmt = "best/bestvideo+bestaudio"
            elif q in ("720", "720p"):
                fmt = "best[height<=720]/bestvideo[height<=720]+bestaudio/best"
            elif q in ("1080", "1080p", "fhd"):
                fmt = "best[height<=1080]/bestvideo[height<=1080]+bestaudio/best"
            elif q in ("1440", "1440p", "qhd"):
                fmt = "best[height<=1440]/bestvideo[height<=1440]+bestaudio/best"
            elif q in ("2160", "2160p", "4k"):
                fmt = "best[height<=2160]/bestvideo[height<=2160]+bestaudio/best"
            else:
                fmt = "best[height<=1080]/bestvideo[height<=1080]+bestaudio/best"
            cmd += ["-f", fmt, "--merge-output-format", "mp4"]
        s_t = self.cfg.get("start_t"); e_t = self.cfg.get("end_t")
        if s_t is not None or e_t is not None:
            s = _hms(int(s_t or 0))
            e = _hms(int(e_t)) if e_t is not None else "inf"
            cmd += ["--download-sections", f"*{s}-{e}",
                    "--force-keyframes-at-cuts"]
        cmd.append(self.url)
        return cmd

    def _parse_line(self, line: str):
        if not self.title:
            t = self._extract_title(line)
            if t: self.title = t
        m = _SECTION_RE.search(line)
        if m:
            try:
                self._section_start = float(m.group(1))
                self._section_end = float(m.group(2))
                self._section_duration = max(
                    0.001, self._section_end - self._section_start)
            except Exception: pass
            return
        m = _PROG_RE.search(line)
        if m:
            try: self.progress_pct = float(m.group(1))
            except: pass
            self.size_total = (m.group(2) or "").strip()
            self.speed = (m.group(3) or "").strip()
            self.eta = (m.group(4) or "").strip()
            return
        m = _FFMPEG_TIME_RE.search(line)
        if m and getattr(self, "_section_duration", 0) > 0:
            h, mm, ss = int(m.group(1)), int(m.group(2)), int(m.group(3))
            ms = int(m.group(4) or 0) * (10 if m.group(4) and len(m.group(4)) == 2 else 1)
            elapsed = h * 3600 + mm * 60 + ss + ms / 1000.0
            pct = min(99.9, 100.0 * elapsed / self._section_duration)
            self.progress_pct = pct
            if not self.size_total:
                self.size_total = f"~{int(self._section_duration)}s"
            return
        m = _DEST_RE.match(line)
        if m:
            p = Path(m.group(1))
            if p.suffix.lower() not in (".vtt", ".srt", ".ttml", ".srv1",
                                          ".srv2", ".srv3", ".json3",
                                          ".description") \
                    and not p.name.endswith(".info.json") \
                    and not p.name.endswith(".live_chat.json"):
                self.file = p
            return
        m = _MERGE_DEST_RE.match(line)
        if m:
            self.file = Path(m.group(1))
            return
        low = line.lower()
        if any(k in low for k in ("error:", "error ", "unsupported url",
                                    "subscriber only", "video unavailable",
                                    "vod is subscriber-only")):
            if not self.error:
                self.error = line.strip()[:200]

    @staticmethod
    def _extract_title(line: str) -> Optional[str]:
        m = _DEST_RE.match(line) or _MERGE_DEST_RE.match(line)
        if m:
            p = Path(m.group(1))
            stem = p.stem
            stem = re.sub(r"\s*\[[A-Za-z0-9_-]{8,15}\]$", "", stem)
            return stem
        return None


# ---------- module-level helpers ----------
def start_download(url: str, cfg: dict) -> TwitchJob:
    job = TwitchJob(url, cfg)
    job.start()
    return job


def get_job(job_id: str) -> Optional[TwitchJob]:
    with _LOCK:
        return JOBS.get(job_id)


def stop_job(job_id: str) -> Optional[TwitchJob]:
    j = get_job(job_id)
    if j: j.stop()
    return j


# Match download.py's sidecar filter for list/delete
_SIDECAR_EXTS = (".vtt", ".srt", ".srv1", ".srv2", ".srv3", ".ttml",
                 ".json3", ".description", ".ytdlp.log")
_MEDIA_EXTS = (".mp4", ".m4a", ".mp3", ".webm", ".mkv", ".mov")


def _is_sidecar(p: Path) -> bool:
    if p.suffix.lower() in _SIDECAR_EXTS: return True
    if p.name.endswith(".info.json"): return True
    if p.name.endswith(".ytdlp.log"): return True
    return False


def list_jobs() -> list:
    with _LOCK:
        out = [j.status_dict() for j in JOBS.values()]
    # Also surface files on disk that don't correspond to a live job (e.g.
    # after clipper restart) so the UI can still reveal / delete them.
    known = set()
    for j in out:
        fn = j.get("file_name")
        if fn: known.add(fn)
        if fn and j.get("status") == "done" and not (TWITCH_ROOT / fn).exists():
            j["status"] = "missing"
    if TWITCH_ROOT.exists():
        for f in TWITCH_ROOT.iterdir():
            if not f.is_file(): continue
            if f.suffix.lower() not in _MEDIA_EXTS: continue
            if _is_sidecar(f): continue
            if f.name in known: continue
            # `<vid>.temp.mp4` is a yt-dlp intermediate — download was
            # killed before the final rename to `<vid>.mp4`. Surface it
            # as an orphan so the user can see + delete via the UI, not
            # as "DONE" which reads like a finished VOD.
            is_orphan = ".temp." in f.name.lower()
            out.append({
                "id":          f"tw_disk_{f.stem}",
                "vid":         f.stem,
                "url":         "",
                "title":       f.stem,
                "status":      "orphan" if is_orphan else "done",
                "progress_pct": 100.0,
                "size_total":  "",
                "speed":       "",
                "eta":         "",
                "phase_msg":   "orphan (killed mid-download)" if is_orphan else "on disk",
                "created_at":  int(f.stat().st_mtime),
                "started_at":  None,
                "finished_at": int(f.stat().st_mtime),
                "file_url":    f"/downloads/twitch/{urllib.parse.quote(f.name)}",
                "file_name":   f.name,
                "file_size_b": f.stat().st_size,
                "error":       None,
                "cfg":         {},
            })
    return sorted(out, key=lambda j: -(j.get("created_at") or 0))


def delete_job(job_id: str) -> bool:
    with _LOCK:
        j = JOBS.pop(job_id, None)
    fn = j.file.name if (j and j.file) else None
    # Also allow deleting a disk-only entry via its synthetic id. The
    # synthetic id is `tw_disk_<Path.stem>` — for orphan `<vid>.temp.mp4`
    # that stem is `<vid>.temp`, not the plain `<vid>`.
    if not j and job_id.startswith("tw_disk_"):
        fn = job_id[len("tw_disk_"):]
    if not fn: return False
    # Match by VID prefix rather than Path.stem — orphan `<vid>.temp.mp4`
    # has stem `<vid>.temp` which doesn't match sidecar `<vid>.ytdlp.log`
    # (stem `<vid>.ytdlp`) if we used stem-equality. Prefix-match nukes
    # the whole vid family in one pass: main file + `.temp.mp4` orphan +
    # `.ytdlp.log` + `.info.json` + any sectioned variants.
    vid = fn.split(".", 1)[0].split("_", 1)[0]  # strip .anything AND _range suffix
    if not vid: return False
    removed = False
    if TWITCH_ROOT.exists():
        for p in list(TWITCH_ROOT.iterdir()):
            if not p.is_file(): continue
            n = p.name
            if n == vid or n.startswith(f"{vid}.") or n.startswith(f"{vid}_"):
                try: p.unlink(); removed = True
                except Exception: pass
    return removed
