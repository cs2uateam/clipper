"""Multi-segment trim/cut for any source mp4 in the clipper.

Mirrors the structure of `upload.py` / `download.py`: a CutJob class,
a small _LOCK + JOBS dict, plus start/get/stop/list/delete helpers.
The cutter takes a source file (already on disk, addressed by
(source, name) like the Idea modal) plus a list of {start_s, end_s,
label} segments and produces one output mp4 per segment.

Output location: `downloads/cuts/<src-stem>_seg{NN}_{start}-{end}.mp4`.
This sits beside the YouTube/Insta/TikTok roots inside `downloads/` so
the existing reveal/thumb endpoints can serve it.

Default ffmpeg mode is stream-copy (`-c copy`) — instant, no
re-encode, accurate to the nearest video keyframe. That's "TikTok
accurate" per the user. For frame-exact cuts caller can pass
`precise=True` which re-encodes with libx264 (slower)."""
import subprocess, threading, time, urllib.parse
from pathlib import Path
from typing import Dict, List, Optional

import core
import subs_burn

V1 = Path(__file__).resolve().parent
CUTS_ROOT = V1 / "downloads" / "cuts"
CUTS_ROOT.mkdir(parents=True, exist_ok=True)

_LOCK = threading.Lock()
JOBS: Dict[str, "CutJob"] = {}


def _safe_segment_name(stem: str, idx: int, start_s: float, end_s: float) -> str:
    """Filename for one cut segment. Pattern: `<stem>_seg<NN>_<start>-<end>.mp4`.
    Time values render as seconds with one decimal so a re-cut of the
    same window overwrites cleanly (same filename → ffmpeg -y)."""
    # Strip filesystem-unfriendly chars from the source stem so we don't
    # end up with `[VIDEO_ID]` or similar in the output filename — that
    # broke other parts of the system in the past (see analyzer memory).
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in stem)
    return f"{safe}_seg{idx:02d}_{start_s:.1f}-{end_s:.1f}.mp4"


class CutJob:
    """Cut N segments out of a single source mp4. Each segment writes to
    its own output file under CUTS_ROOT. Stream-copy by default."""

    def __init__(self, source_path: Path, segments: List[dict],
                  precise: bool = False, label: str = "",
                  wm_cfg: Optional[dict] = None):
        self.source = source_path
        # Normalize segment list. Drop anything where end<=start.
        self.segments = []
        for i, s in enumerate(segments):
            try:
                a = float(s.get("start_s") or 0)
                b = float(s.get("end_s") or 0)
            except (TypeError, ValueError):
                continue
            if b <= a: continue
            self.segments.append({
                "start_s": a, "end_s": b,
                "label":   str(s.get("label") or "").strip(),
            })
        self.precise = bool(precise)
        # Watermark cfg lives separately from `precise` because a wm
        # always forces re-encode regardless of the user's precise
        # checkbox. Empty / position=none = no overlay.
        self.wm_cfg = dict(wm_cfg or {})
        self.wm_path: Optional[Path] = subs_burn.resolve_watermark(self.wm_cfg)
        # Job id includes source stem + timestamp so list() can show
        # multiple cut runs on the same source without colliding.
        self.id = f"cut_{int(time.time()*1000)}_{source_path.stem[:24]}"
        self.label = label or source_path.stem
        self.created_at = int(time.time())
        self.started_at: Optional[int] = None
        self.finished_at: Optional[int] = None
        self.status = "queued"  # queued | cutting | done | failed | cancelled
        self.phase_msg = ""
        self.error: Optional[str] = None
        self.outputs: List[dict] = []  # populated as each segment finishes

        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._cancel = False

    # ---------- public ----------
    def start(self):
        if not self.segments:
            self.status = "failed"
            self.error = "no valid segments"
            self.finished_at = int(time.time())
            with _LOCK: JOBS[self.id] = self
            return self
        with _LOCK: JOBS[self.id] = self
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._cancel = True
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
                try: self._proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    try: self._proc.kill()
                    except Exception: pass
            except Exception: pass
        if self.status in ("queued", "cutting"):
            self.status = "cancelled"

    def status_dict(self) -> dict:
        return {
            "id":          self.id,
            "source":      str(self.source),
            "source_name": self.source.name,
            "label":       self.label,
            "status":      self.status,
            "phase_msg":   self.phase_msg,
            "created_at":  self.created_at,
            "started_at":  self.started_at,
            "finished_at": self.finished_at,
            "precise":     self.precise,
            "n_segments":  len(self.segments),
            "outputs":     list(self.outputs),
            "error":       self.error,
        }

    # ---------- internal ----------
    def _run(self):
        try:
            self.started_at = int(time.time())
            self.status = "cutting"
            for i, seg in enumerate(self.segments):
                if self._cancel:
                    self.status = "cancelled"
                    return
                self.phase_msg = (
                    f"сегмент {i+1}/{len(self.segments)} — "
                    f"{seg['start_s']:.1f}s → {seg['end_s']:.1f}s"
                )
                out_name = _safe_segment_name(self.source.stem, i,
                                              seg["start_s"], seg["end_s"])
                out = CUTS_ROOT / out_name
                duration = seg["end_s"] - seg["start_s"]
                # Three modes:
                # 1) Watermark set → cut + overlay in ONE re-encode pass
                #    (precise flag is implied — overlay needs re-encode
                #    anyway, so we use frame-accurate seek too).
                # 2) precise=True without wm → re-encode, frame-accurate.
                # 3) Default → stream-copy, keyframe-bound, instant.
                if self.wm_path is not None:
                    out_w, out_h = subs_burn.video_dimensions(self.source)
                    fc = subs_burn.build_watermark_filter_complex(
                        out_w, out_h, self.wm_path, self.wm_cfg,
                        in_label="0:v", out_label="out")
                    cmd = [core.FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
                           "-ss", f"{seg['start_s']:.3f}",
                           "-i", str(self.source),
                           "-i", str(self.wm_path),
                           "-t", f"{duration:.3f}",
                           "-filter_complex", fc,
                           "-map", "[out]", "-map", "0:a?",
                           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                           "-c:a", "aac", "-b:a", "128k",
                           "-movflags", "+faststart",
                           str(out)]
                elif self.precise:
                    cmd = [core.FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
                           "-ss", f"{seg['start_s']:.3f}",
                           "-i", str(self.source),
                           "-t", f"{duration:.3f}",
                           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                           "-c:a", "aac", "-b:a", "128k",
                           "-movflags", "+faststart",
                           str(out)]
                else:
                    # -ss BEFORE -i = fast seek (keyframe). Combined with
                    # -avoid_negative_ts to clean up PTS at the boundary.
                    cmd = [core.FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
                           "-ss", f"{seg['start_s']:.3f}",
                           "-i", str(self.source),
                           "-t", f"{duration:.3f}",
                           "-c", "copy",
                           "-avoid_negative_ts", "make_zero",
                           "-movflags", "+faststart",
                           str(out)]
                self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                              stderr=subprocess.PIPE)
                _, stderr = self._proc.communicate(timeout=60 * 60)
                rc = self._proc.returncode
                self._proc = None
                if rc != 0:
                    err = (stderr or b"").decode(errors="ignore")[:300]
                    raise RuntimeError(f"ffmpeg exit {rc}: {err}")
                size = 0
                try: size = out.stat().st_size
                except Exception: pass
                self.outputs.append({
                    "idx":      i,
                    "file":     out.name,
                    "url":      f"/downloads/cuts/{urllib.parse.quote(out.name)}",
                    "start_s":  seg["start_s"],
                    "end_s":    seg["end_s"],
                    "duration": duration,
                    "label":    seg["label"],
                    "size_b":   size,
                })
            self.status = "done"
            self.phase_msg = ""
        except subprocess.TimeoutExpired:
            self.status = "failed"
            self.error = "ffmpeg timed out (>60min)"
        except Exception as e:
            if self.status != "cancelled":
                self.status = "failed"
                self.error = f"{type(e).__name__}: {e}"
        finally:
            self.finished_at = int(time.time())
            self._proc = None


# ---------- API helpers ----------
def start_cut(source_path: Path, segments: List[dict],
              precise: bool = False, label: str = "",
              wm_cfg: Optional[dict] = None) -> CutJob:
    if not source_path.exists():
        raise ValueError(f"source missing: {source_path}")
    if not segments:
        raise ValueError("no segments to cut")
    return CutJob(source_path, segments,
                  precise=precise, label=label, wm_cfg=wm_cfg).start()


def get_job(job_id: str) -> Optional[CutJob]:
    with _LOCK: return JOBS.get(job_id)


def stop_job(job_id: str) -> Optional[CutJob]:
    job = get_job(job_id)
    if job: job.stop()
    return job


def list_jobs() -> list:
    """In-memory jobs + disk scan of CUTS_ROOT. Each on-disk file becomes
    a synthetic 'done' entry so reloads after server restart still show
    everything the user produced."""
    with _LOCK:
        live = [j.status_dict() for j in JOBS.values()]
    # Files that belong to any live job — skip them in the disk scan to
    # avoid duplicating.
    live_files = set()
    for j in live:
        for o in j.get("outputs", []):
            live_files.add(o.get("file"))
    out = list(live)
    _VIDEO_EXTS = {".mp4"}
    _AUDIO_EXTS = {".mp3", ".m4a", ".wav"}
    if CUTS_ROOT.exists():
        disk = []
        for f in CUTS_ROOT.iterdir():
            if not f.is_file(): continue
            ext = f.suffix.lower()
            if ext not in _VIDEO_EXTS and ext not in _AUDIO_EXTS: continue
            if f.name in live_files: continue
            try: stat = f.stat()
            except Exception: continue
            # Parse start/end from filename if it follows our pattern.
            start_s = end_s = None
            try:
                # `<stem>_seg<NN>_<start>-<end>.<ext>`
                core_name = f.stem
                if "_seg" in core_name:
                    tail = core_name.rsplit("_seg", 1)[1]
                    parts = tail.split("_", 1)
                    if len(parts) == 2 and "-" in parts[1]:
                        a, b = parts[1].rsplit("-", 1)
                        start_s, end_s = float(a), float(b)
            except Exception:
                pass
            disk.append({
                "id":           f"cut_disk:{f.name}",
                "file":         f.name,
                "file_name":    f.name,
                "kind":         "audio" if ext in _AUDIO_EXTS else "video",
                "ext":          ext.lstrip("."),
                "url":          f"/downloads/cuts/{urllib.parse.quote(f.name)}",
                "title":        f.stem,
                "status":       "done",
                "phase_msg":    "",
                "finished_at":  int(stat.st_mtime),
                "file_size_b":  stat.st_size,
                "start_s":      start_s,
                "end_s":        end_s,
                "duration":     (end_s - start_s) if start_s is not None and end_s is not None else None,
            })
        out.extend(disk)
    out.sort(key=lambda j: -(j.get("finished_at") or j.get("created_at") or 0))
    return out


def delete_job(job_id: str) -> bool:
    """Stop in-memory job (if any) and remove its output files plus any
    sidecars from disk. Also supports `cut_disk:<filename>` for entries
    from the disk archive that aren't tied to a live job."""
    # Disk-archive entry — delete that one file + its sidecars.
    if job_id.startswith("cut_disk:"):
        fname = job_id[len("cut_disk:"):]
        fp = (CUTS_ROOT / fname).resolve()
        try: fp.relative_to(CUTS_ROOT.resolve())
        except ValueError: return False
        if not fp.exists(): return False
        try: fp.unlink()
        except Exception: return False
        _wipe_sidecars(fp)
        return True
    # Live or finished in-memory job — drop all segment files + sidecars
    # + the in-memory record.
    job = get_job(job_id)
    if not job: return False
    job.stop()
    removed = False
    for o in job.outputs:
        fp = CUTS_ROOT / o.get("file", "")
        try:
            if fp.exists(): fp.unlink(); removed = True
        except Exception: pass
        _wipe_sidecars(fp)
    with _LOCK: JOBS.pop(job_id, None)
    return removed or True


_SIDECAR_SUFFIXES = (".thumb.jpg", ".idea.json", ".en.vtt",
                     ".whisper.en.vtt", ".ytdlp.log", ".info.json")

def _wipe_sidecars(fp: Path) -> int:
    """Delete every sidecar that shares `fp.stem` in `fp.parent`. Returns
    the count removed. Sidecars include thumbnails, Gemini cache, vtt
    subs, ytdlp log, etc. — anything we know we ever create next to a
    media file."""
    n = 0
    stem = fp.stem
    parent = fp.parent
    if not parent.exists(): return 0
    for sib in list(parent.iterdir()):
        if not sib.is_file(): continue
        # `<stem>.thumb.jpg` etc. — name starts with stem + `.`
        if sib.name.startswith(stem + "."):
            for suf in _SIDECAR_SUFFIXES:
                if sib.name.endswith(suf):
                    try: sib.unlink(); n += 1
                    except Exception: pass
                    break
    return n


def delete_segment(filename: str) -> bool:
    """Remove a single output file (segment mp4 OR extracted audio) plus
    any sidecars (thumb, etc.). Filename-only — no path components."""
    if "/" in filename or "\\" in filename: return False
    fp = (CUTS_ROOT / filename).resolve()
    try: fp.relative_to(CUTS_ROOT.resolve())
    except ValueError: return False
    if not fp.exists(): return False
    try: fp.unlink()
    except Exception: return False
    _wipe_sidecars(fp)
    with _LOCK:
        for j in JOBS.values():
            j.outputs = [o for o in j.outputs if o.get("file") != filename]
    return True


def merge_segments(filenames: List[str], output_name: str = "") -> Path:
    """Concat 2+ already-cut segment files into a single mp4. Uses
    ffmpeg's concat demuxer with `-c copy` since all our segment
    outputs come from the same source through stream-copy, so codec
    params line up by construction. Order preserves the input list.

    Output lands in CUTS_ROOT with name `output_name` (if provided
    + sanitized) or auto-named `merged_<YYYY-MM-DD_HH-MM-SS>.mp4`.
    Raises ValueError on <2 inputs, missing file, or path-escape;
    RuntimeError on ffmpeg failure."""
    if not filenames or len(filenames) < 2:
        raise ValueError("merge requires at least 2 segments")
    # Resolve + validate every input lives under CUTS_ROOT.
    paths: List[Path] = []
    root = CUTS_ROOT.resolve()
    for fn in filenames:
        if "/" in fn or "\\" in fn:
            raise ValueError(f"bad filename: {fn}")
        fp = (CUTS_ROOT / fn).resolve()
        try: fp.relative_to(root)
        except ValueError:
            raise ValueError(f"path escape: {fn}")
        if not fp.exists():
            raise ValueError(f"segment missing: {fn}")
        paths.append(fp)
    # Pick output name. Sanitize user input the same way segment names do.
    if output_name:
        stem = "".join(c if c.isalnum() or c in "-_." else "_" for c in output_name)
        if not stem.lower().endswith(".mp4"): stem += ".mp4"
        out = CUTS_ROOT / stem
    else:
        from datetime import datetime
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out = CUTS_ROOT / f"merged_{ts}.mp4"
    # ffmpeg concat demuxer wants a list file with `file '<abs path>'`.
    # The quoting style protects against spaces but each path's own
    # single-quotes get escaped to `'\''` per ffmpeg docs.
    list_file = CUTS_ROOT / f".__concat_{int(time.time()*1000)}.txt"
    try:
        with open(list_file, "w", encoding="utf-8") as f:
            for p in paths:
                safe = str(p).replace("'", "'\\''")
                f.write(f"file '{safe}'\n")
        cmd = [core.FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
               "-f", "concat", "-safe", "0",
               "-i", str(list_file),
               "-c", "copy",
               "-movflags", "+faststart",
               str(out)]
        proc = subprocess.run(cmd, capture_output=True, timeout=60 * 30)
        if proc.returncode != 0:
            err = (proc.stderr or b"").decode(errors="ignore")[:400]
            raise RuntimeError(f"ffmpeg concat exit {proc.returncode}: {err}")
    finally:
        try: list_file.unlink()
        except Exception: pass
    return out


def extract_audio(filename: str, fmt: str = "mp3") -> Path:
    """Pull the audio track out of a cut segment into a standalone file.

    fmt:
      - "mp3" → libmp3lame @ 192k (lossy, small, plays everywhere)
      - "m4a" → AAC stream-copy if source is AAC, else encode to AAC 192k
      - "wav" → PCM 16-bit (lossless, big)

    Lands beside the source in CUTS_ROOT as `<stem>.<fmt>`. Raises
    ValueError for unknown filenames / unsupported fmt / path-escape;
    RuntimeError on ffmpeg failure."""
    if "/" in filename or "\\" in filename:
        raise ValueError(f"bad filename: {filename}")
    fmt = (fmt or "mp3").lower()
    if fmt not in ("mp3", "m4a", "wav"):
        raise ValueError(f"unsupported audio fmt: {fmt}")
    src = (CUTS_ROOT / filename).resolve()
    try: src.relative_to(CUTS_ROOT.resolve())
    except ValueError:
        raise ValueError("path escape")
    if not src.exists():
        raise ValueError(f"segment missing: {filename}")
    out = src.with_suffix(f".{fmt}")
    if fmt == "mp3":
        codec_args = ["-c:a", "libmp3lame", "-b:a", "192k"]
    elif fmt == "m4a":
        # AAC stream-copy when possible. ffmpeg figures it out via
        # `-c:a copy` only if input is already AAC; cheaper than always
        # re-encoding, fallback below in case it fails.
        codec_args = ["-c:a", "aac", "-b:a", "192k"]
    else:  # wav
        codec_args = ["-c:a", "pcm_s16le"]
    cmd = [core.FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
           "-i", str(src),
           "-vn",  # no video
           *codec_args,
           str(out)]
    proc = subprocess.run(cmd, capture_output=True, timeout=60 * 30)
    if proc.returncode != 0:
        err = (proc.stderr or b"").decode(errors="ignore")[:400]
        raise RuntimeError(f"ffmpeg audio extract exit {proc.returncode}: {err}")
    return out


def resolve_source(source: str, name: str) -> Path:
    """Map a (source-kind, filename) pair to an absolute path under one of
    the four media roots — same convention used by the Idea modal. Raises
    ValueError on unknown source or path-traversal."""
    import download as dl_mod
    import upload as up_mod
    import insta as ig_mod
    import tiktok as tt_mod
    roots = {
        "youtube": dl_mod.DOWNLOADS_ROOT,
        "insta":   ig_mod.INSTA_ROOT,
        "tiktok":  tt_mod.TIKTOK_ROOT,
        "upload":  up_mod.UPLOADS_ROOT,
        "cuts":    CUTS_ROOT,
    }
    root = roots.get(source)
    if root is None:
        raise ValueError(f"unknown source: {source}")
    root = root.resolve()
    candidate = (root / name).resolve()
    try: candidate.relative_to(root)
    except ValueError:
        raise ValueError("path escapes root")
    if not candidate.exists():
        raise ValueError(f"file not found: {source}/{name}")
    return candidate
