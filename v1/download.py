"""Whole-video download via yt-dlp. Separate from the analyze pipeline —
no scoring, no captions, no clipping. Just grab the file.

Each download is a DownloadJob owning a subprocess + a parsing thread.
Progress is parsed from yt-dlp's stdout (`[download] 12.3% of 4.5GiB at
1.2MiB/s ETA 00:42`) and exposed via status() for the UI to poll.

Files land FLAT in `clipper/v1/downloads/youtube/<vid>.<ext>` (matches
insta/tiktok layout). Sidecars (`.vtt`, `.ytdlp.log`, `.info.json`)
sit next to the mp4 with the same `<vid>.` stem prefix and are hidden
from the UI list."""
import re, subprocess, threading, time, urllib.parse
from pathlib import Path
from typing import Dict, Optional

import core
import subs_burn

V1 = Path(__file__).resolve().parent
# All source-ingest tabs land under one common `downloads/` parent so the
# user has a single place to look. YouTube keeps its per-video subfolder
# layout (sectioned/sidecar-friendly); insta/tiktok stay flat. See
# `insta.py` / `tiktok.py` for their roots.
DOWNLOADS_ROOT = V1 / "downloads" / "youtube"
DOWNLOADS_ROOT.mkdir(parents=True, exist_ok=True)

_LOCK = threading.Lock()
JOBS: Dict[str, "DownloadJob"] = {}


_PROG_RE = re.compile(
    r"\[download\]\s+([\d.]+)%\s+of\s+~?\s*([\d.]+\s*\w+)"
    r"(?:\s+at\s+([\d.]+\s*\w+/s))?(?:\s+ETA\s+([\d:]+))?"
)
_DEST_RE = re.compile(r"^\[download\]\s+Destination:\s+(.+)$")
_MERGE_DEST_RE = re.compile(r"^\[Merger\]\s+Merging formats into\s+\"(.+)\"$")
# yt-dlp's --download-sections delegates the actual fetch to an internal
# ffmpeg, whose progress prints `frame=… time=00:00:33.00 …` lines using
# carriage-returns to overwrite a single line. We watch for the `time=`
# token to compute % against the section duration.
_FFMPEG_TIME_RE = re.compile(
    r"time=(\d{2}):(\d{2}):(\d{2})(?:\.(\d{2,3}))?")
_SECTION_RE = re.compile(
    r"Downloading 1 time ranges?:\s*([\d.]+)-([\d.]+)")

def _hms(sec: int) -> str:
    """Seconds → 'HH:MM:SS' for yt-dlp --download-sections. Same format
    core.py uses for clip rendering (which is the proven-working path)."""
    h, r = divmod(int(sec), 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def _iter_lines(stream):
    """Yield logical lines from a subprocess BINARY stream, treating both
    '\\n' AND '\\r' as line breaks. yt-dlp prints with '\\n' (we pass
    --newline), but the ffmpeg child it spawns for --download-sections
    uses '\\r' to update a single line of progress in place.

    Binary mode + byte-level reads avoid Python's text-mode buffering,
    which on Windows can wait for '\\n' before flushing the read buffer
    even when '\\r'-terminated data is already in the OS pipe. Without
    this, the pipe fills up, ffmpeg blocks on write, and the entire
    download deadlocks at 0%."""
    buf = bytearray()
    while True:
        chunk = stream.read(1)
        if not chunk:
            if buf:
                yield buf.decode("utf-8", errors="ignore")
            return
        if chunk in (b"\r", b"\n"):
            if buf:
                yield buf.decode("utf-8", errors="ignore")
                buf = bytearray()
        else:
            buf.extend(chunk)

class DownloadJob:
    def __init__(self, url: str, cfg: dict):
        self.url = url
        self.cfg = {
            "quality":      "1080",      # 720 / 1080 / 1440 / 2160 / best
            "audio_only":   False,
            "audio_format": "mp3",       # mp3 / m4a (only if audio_only)
            "subtitles":    False,
            "burn_subs":    False,       # burn into video pixels (re-encode)
            "subs_size":    "medium",    # small / medium / large / huge
            "subs_color":   "white",     # white / yellow / red / green / cyan / black
            "subs_bg":      "none",      # none / shadow / dark_box / solid_box
            "subs_position":"bottom",    # bottom / top / center
            "start_t":      None,
            "end_t":        None,
            "aspect":       "source",    # source / 9:16 / 16:9 / 1:1 / 4:5 / W:H
            **(cfg or {}),
        }
        try:
            self.vid = core.yt_id_from_url(url)
        except ValueError:
            self.vid = "unknown"
        self.id = f"dl_{self.vid}_{int(time.time() * 1000)}"

        self.created_at = int(time.time())
        self.started_at: Optional[int] = None
        self.finished_at: Optional[int] = None
        self.status = "queued"           # queued | downloading | burning | done | failed | cancelled
        self.progress_pct = 0.0
        self.size_total = ""             # "4.5GiB"
        self.speed = ""
        self.eta = ""
        self.title = ""
        self.file: Optional[Path] = None
        self.file_size: Optional[int] = None
        self.error: Optional[str] = None
        self.phase_msg = ""              # extra status detail for the UI

        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        # Populated when yt-dlp emits "Downloading 1 time ranges: A-B" —
        # used to compute % from ffmpeg's `time=…` lines for sections.
        self._section_start = 0.0
        self._section_end = 0.0
        self._section_duration = 0.0

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
            file_url = f"/downloads/{urllib.parse.quote(self.file.name)}"
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
            # Surface what cfg the job actually received — when the user
            # reports "не сжалось / не наложилось" we want to see at a
            # glance whether the request even arrived with the right
            # aspect/watermark/etc. or the browser sent stale defaults.
            print(f"[dl] cfg: aspect={self.cfg.get('aspect')!r} "
                  f"quality={self.cfg.get('quality')!r} "
                  f"audio_only={self.cfg.get('audio_only')} "
                  f"start={self.cfg.get('start_t')} end={self.cfg.get('end_t')}",
                  flush=True)
            cmd = self._build_cmd()
            # Write yt-dlp output to a log file rather than a pipe. On
            # Windows, when yt-dlp invokes a child ffmpeg for
            # --download-sections, ffmpeg writes '\r'-only progress
            # straight to the inherited handle. With Popen(stdout=PIPE)
            # the kernel pipe buffer fills up (Python's read can't
            # consume bytes faster than ffmpeg produces them in some
            # configurations), ffmpeg blocks on write, and the entire
            # download deadlocks at 0%. A real file has no such limit
            # — we tail it for progress lines instead.
            # Per-job log lives at `<vid>.ytdlp.log` so concurrent
            # downloads don't stomp each other in the flat root.
            log_path = DOWNLOADS_ROOT / f"{self.vid}.ytdlp.log"
            try: log_path.unlink()
            except Exception: pass
            log_handle = open(log_path, "ab", buffering=0)
            self._proc = subprocess.Popen(
                cmd, stdout=log_handle, stderr=subprocess.STDOUT)
            # Tail the log file while the process runs.
            last_off = 0
            while True:
                if self._proc.poll() is not None:
                    # Drain any tail bytes one last time
                    pass
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
                # Find the resulting file
                if self.file and self.file.exists():
                    self.file_size = self.file.stat().st_size
                else:
                    # Find a media file matching this vid in the flat root.
                    candidates = sorted(
                        [p for p in DOWNLOADS_ROOT.iterdir()
                         if p.is_file() and p.stem == self.vid
                         and p.suffix.lower() in (".mp4",".m4a",".mp3",".webm",".mkv",".mov")],
                        key=lambda p: -p.stat().st_mtime)
                    if candidates:
                        self.file = candidates[0]
                        self.file_size = self.file.stat().st_size
                # Drop the per-job yt-dlp log on success — it was just for
                # progress tailing, not user-facing. Kept on failure for
                # post-mortem diagnostics.
                try: log_path.unlink()
                except Exception: pass
                # Aspect-ratio post-process: re-encode with crop+scale to
                # the user-chosen format. Skip when "source" (default) so
                # we don't pointlessly re-encode 16:9 → 16:9.
                aspect = (self.cfg.get("aspect") or "source").strip()
                if aspect and aspect != "source" and self.file and not self.cfg.get("audio_only"):
                    try:
                        self.status = "reformatting"
                        self.phase_msg = f"конвертирую в {aspect}"
                        self._apply_aspect(aspect)
                        self.file_size = self.file.stat().st_size
                    except Exception as e:
                        self.error = f"aspect convert failed: {e}"
                # If user wanted subtitles but YouTube didn't provide any
                # (no manual subs, no auto-captions — common on Shorts and
                # older / unverified channels), fall back to Whisper to
                # generate them locally. Without this the user just gets a
                # silent un-subbed mp4 and wonders why.
                if (self.cfg.get("subtitles") and not self.cfg.get("audio_only")
                        and self.file):
                    yt_vtts = subs_burn.find_subtitle_files(
                        self.file, exclude_whisper=True)
                    if not yt_vtts:
                        try:
                            self.status = "transcribing"
                            self.phase_msg = "субтитров на YouTube нет — гоню Whisper"
                            subs_burn.generate_subs_via_whisper(
                                self.file,
                                on_phase=lambda m: setattr(self, "phase_msg", m))
                        except Exception as e:
                            self.error = (f"YouTube has no captions and Whisper "
                                          f"fallback failed: {e}")
                # Optional post-process: burn subtitles into video pixels
                if (self.cfg.get("burn_subs") and self.cfg.get("subtitles")
                        and not self.cfg.get("audio_only") and self.file):
                    try:
                        self.status = "burning"
                        subs_burn.burn_subtitles(
                            self.file, self.cfg,
                            on_phase=lambda m: setattr(self, "phase_msg", m),
                            on_proc=lambda p: setattr(self, "_proc", p))
                        self.file_size = self.file.stat().st_size
                    except Exception as e:
                        self.error = f"burn-subs failed: {e}"
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

    def _apply_aspect(self, aspect: str) -> None:
        """Re-encode the downloaded file in-place to a target aspect.
        Crop centered + scale to canvas size. Replaces the original.
        Audio passes through (-c:a copy) — only video is re-encoded."""
        if not (self.file and self.file.exists()): return
        vf = core.aspect_vf(aspect)
        tmp = self.file.with_name(self.file.stem + ".aspect.mp4")
        cmd = [core.FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
               "-i", str(self.file),
               "-vf", vf,
               "-c:v", "libx264", "-preset", "fast", "-crf", "20",
               "-pix_fmt", "yuv420p",
               "-c:a", "copy",
               "-movflags", "+faststart",
               str(tmp)]
        p = subprocess.Popen(cmd)
        self._proc = p
        p.wait()
        self._proc = None
        if p.returncode != 0 or not tmp.exists():
            try: tmp.unlink()
            except Exception: pass
            raise RuntimeError(f"ffmpeg aspect-convert exit {p.returncode}")
        try: self.file.unlink()
        except Exception: pass
        tmp.rename(self.file)

    def _build_cmd(self):
        # Output template uses `%(id)s.%(ext)s` so each video is
        # `<vid>.mp4` directly in the flat root. Drops the title from
        # the filename — title still surfaces in `self.title` and the UI.
        out_tpl = str(DOWNLOADS_ROOT / "%(id)s.%(ext)s")
        cmd = [core.YTDLP, "--newline", "--no-warnings",
               "--ffmpeg-location", core.FFMPEG,
               "-o", out_tpl,
               "--no-playlist",
               # ASCII-only filenames: strip '#?&' and other URL-special chars
               # from titles. Otherwise YouTube Shorts with hashtag-loaded
               # titles produce filenames that browsers cut at the first '#'
               # when used as href, returning the FastAPI JSON 404 instead
               # of the file.
               "--restrict-filenames"]
        if self.cfg.get("audio_only"):
            af = (self.cfg.get("audio_format") or "mp3").lower()
            cmd += ["-x", "--audio-format", af, "--audio-quality", "0"]
        else:
            q = (self.cfg.get("quality") or "1080").lower()
            # Prefer single-stream HLS (`best[h<=N]`) over DASH
            # (`bestvideo+bestaudio`). HLS 1080p60 (format 301) is
            # progressive — ffmpeg can seek into it cleanly for
            # `--download-sections`. DASH (formats 399/137 + 251)
            # delegates section-cutting to ffmpeg with `-ss before -i`,
            # which hangs against googlevideo CDN URLs because input-seek
            # has to read from byte 0. Same selector for full + section
            # downloads — matches the proven path used by core.py.
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
        if self.cfg.get("subtitles"):
            cmd += ["--write-subs", "--write-auto-subs",
                    "--sub-langs", "en,en-orig", "--convert-subs", "vtt"]
        s_t = self.cfg.get("start_t"); e_t = self.cfg.get("end_t")
        if s_t is not None or e_t is not None:
            # Use HH:MM:SS format and --force-keyframes-at-cuts — same
            # combo `core.py` uses for clip rendering, which works
            # reliably on DASH because keyframe alignment lets ffmpeg
            # cut without re-encoding the encrypted DASH stream.
            s = _hms(int(s_t or 0))
            e = _hms(int(e_t)) if e_t is not None else "inf"
            cmd += ["--download-sections", f"*{s}-{e}",
                    "--force-keyframes-at-cuts"]
        cmd.append(self.url)
        return cmd

    def _parse_line(self, line: str):
        # Title from first emit (`[youtube] ID: Downloading webpage` doesn't
        # carry it, but `[info]` lines and merger destinations do)
        if not self.title:
            t = self._extract_title(line)
            if t: self.title = t
        # Detect "Downloading 1 time ranges: 842.0-875.0" so we can
        # compute % against the section length when ffmpeg starts.
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
        # ffmpeg's section-download progress: `time=00:00:33.00`
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
            # yt-dlp emits "[download] Destination:" for EVERY output it
            # creates — the video, the .vtt subs, the .info.json, etc.
            # We only want the primary media file. Without this skip the
            # LAST destination (usually .vtt) would override self.file
            # and the burn step would try to ffmpeg a vtt as input.
            if p.suffix.lower() not in (".vtt", ".srt", ".ttml", ".srv1",
                                          ".srv2", ".srv3", ".json3",
                                          ".description") \
                    and not p.name.endswith(".info.json") \
                    and not p.name.endswith(".live_chat.json"):
                self.file = p
            return
        m = _MERGE_DEST_RE.match(line)
        if m:
            # After merger the final file is here; supersedes any earlier dest
            self.file = Path(m.group(1))
            return
        # Stash useful errors
        low = line.lower()
        if any(k in low for k in ("error:", "error ", "unsupported url",
                                    "private video", "members-only",
                                    "video unavailable")):
            if not self.error:
                self.error = line.strip()[:200]

    @staticmethod
    def _extract_title(line: str) -> Optional[str]:
        # yt-dlp prints `[info] VIDEO_ID: Downloading 1 format(s): ...`
        # which doesn't contain title. Title shows in destination path though.
        m = _DEST_RE.match(line) or _MERGE_DEST_RE.match(line)
        if m:
            p = Path(m.group(1))
            # Strip the "[VIDEO_ID].EXT" suffix from our template
            stem = p.stem
            stem = re.sub(r"\s*\[[A-Za-z0-9_-]{8,15}\]$", "", stem)
            return stem
        return None


# ---------- module-level helpers ----------
def start_download(url: str, cfg: dict) -> DownloadJob:
    job = DownloadJob(url, cfg)
    job.start()
    return job

def get_job(job_id: str) -> Optional[DownloadJob]:
    with _LOCK:
        return JOBS.get(job_id)

def stop_job(job_id: str) -> Optional[DownloadJob]:
    with _LOCK:
        job = JOBS.get(job_id)
    if not job: return None
    job.stop()
    return job

# Sidecar extensions — captions / metadata / our internal yt-dlp log.
# Hidden from the UI list and auto-deleted alongside the parent media
# file when the user removes the parent.
_SIDECAR_EXTS = (".vtt", ".srt", ".srv1", ".srv2", ".srv3", ".ttml", ".json3",
                 ".info.json", ".description", ".live_chat.json", ".log")
_MEDIA_EXTS = (".mp4", ".m4a", ".mp3", ".webm", ".mkv", ".mov")

def _is_sidecar(p: Path) -> bool:
    name = p.name.lower()
    if name.endswith(".ytdlp.log"): return True
    return any(name.endswith(ext) for ext in _SIDECAR_EXTS)

def list_jobs() -> list:
    with _LOCK:
        live = [j.status_dict() for j in JOBS.values()]
    seen = {j.get("file_name") for j in live if j.get("file_name")}
    out = list(live)
    if DOWNLOADS_ROOT.exists():
        for f in DOWNLOADS_ROOT.iterdir():
            if not f.is_file(): continue
            if _is_sidecar(f): continue
            if f.suffix.lower() not in _MEDIA_EXTS: continue
            if f.name in seen: continue
            try: stat = f.stat()
            except Exception: continue
            out.append({
                "id":          f"disk:{f.name}",
                "vid":         f.stem,
                "url":         f"https://www.youtube.com/watch?v={f.stem}",
                "title":       f.stem,
                "status":      "done",
                "progress_pct":100.0,
                "file_url":    f"/downloads/{urllib.parse.quote(f.name)}",
                "file_name":   f.name,
                "file_size_b": stat.st_size,
                "finished_at": int(stat.st_mtime),
                "has_idea":    f.with_suffix(".idea.json").exists(),
            })
    out.sort(key=lambda j: -(j.get("finished_at") or j.get("created_at") or 0))
    return out

def delete_job(job_id: str) -> bool:
    """Stop in-memory job (if active) AND remove the file from disk
    plus any sidecars sharing the same `<stem>.` prefix."""
    target_file: Optional[Path] = None
    with _LOCK:
        job = JOBS.get(job_id)
    if job:
        job.stop()
        target_file = job.file
        with _LOCK:
            JOBS.pop(job_id, None)
    elif job_id.startswith("disk:"):
        rest = job_id[len("disk:"):]
        # Old format was "disk:<vid>/<filename>" before the flat refactor.
        # Strip the leading "<vid>/" so it still resolves under the flat
        # root if the user has stale UI state from an older session.
        if "/" in rest:
            rest = rest.split("/", 1)[1]
        target_file = DOWNLOADS_ROOT / rest
    if target_file is None or not target_file.exists():
        return False
    try:
        _delete_file_with_sidecars(target_file)
        return True
    except Exception:
        return False

def _delete_file_with_sidecars(media_path: Path):
    """Remove the media file plus any sidecars sharing its stem
    (`<vid>.en.vtt`, `<vid>.ytdlp.log`, `<vid>.info.json`, etc.)."""
    if not media_path.exists(): return
    parent = media_path.parent
    prefix = media_path.stem + "."
    try: media_path.unlink()
    except Exception: pass
    if not parent.exists(): return
    for sib in parent.iterdir():
        if not sib.is_file(): continue
        if sib.name.startswith(prefix):
            try: sib.unlink()
            except Exception: pass
