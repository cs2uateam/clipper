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
import re, subprocess, threading, time, urllib.parse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
                  wm_cfg: Optional[dict] = None,
                  subs_cfg: Optional[dict] = None):
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
        # Split-stack mode: when set, the runner takes a separate branch
        # that crops two regions from the same source and vstacks them
        # into a 1080x1920 (9:16) frame instead of cutting N segments.
        # Shape: {"start_t":float,"end_t":float,"r1":{x,y,w,h},
        #         "r2":{x,y,w,h},"split_pct":int}
        self.split_stack: Optional[dict] = None
        # Watermark cfg lives separately from `precise` because a wm
        # always forces re-encode regardless of the user's precise
        # checkbox. Empty / position=none = no overlay.
        self.wm_cfg = dict(wm_cfg or {})
        self.wm_path: Optional[Path] = subs_burn.resolve_watermark(self.wm_cfg)
        # Subtitle cfg: {subs_enabled: bool, subs_lang: "uk"|"en"|...,
        #                subs_size/color/bg/position: str}.
        # Empty / subs_enabled=False → no subs pass; ffmpeg branches keep
        # their historical shape. Turning on forces re-encode.
        self.subs_cfg = dict(subs_cfg or {})
        # Full-source VTT cached across segments of the SAME cut render.
        # First segment triggers Whisper (single pass), rest reuse.
        self._source_vtt: Optional[Path] = None
        # Track temp per-segment sliced VTTs so we can drop them after
        # the render — sidecar folder shouldn't accumulate junk.
        self._temp_vtts: List[Path] = []
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
        if not self.segments and not self.split_stack:
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
        # Split-stack jobs take a wholly different path: single output
        # file built from two crops of the same window, no per-segment
        # loop. Same status fields though, so /api/cut/status is reused.
        if self.split_stack:
            try:
                self.started_at = int(time.time())
                self.status = "cutting"
                self._run_split_stack()
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
                for p in self._temp_vtts:
                    try: p.unlink()
                    except Exception: pass
                self._temp_vtts = []
            return
        try:
            self.started_at = int(time.time())
            self.status = "cutting"
            # If subs are on and we don't yet have a full-source VTT,
            # run Whisper once BEFORE the segment loop. Every segment
            # slices from the same VTT.
            subs_on = bool(self.subs_cfg.get("subs_enabled"))
            if subs_on and self._source_vtt is None:
                self.phase_msg = "гоню Whisper по всему видео"
                self._source_vtt = _prepare_source_subs(
                    self.source, self.subs_cfg,
                    on_phase=lambda m: setattr(self, "phase_msg", m))
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
                # Slice the full-source VTT down to this segment's window.
                seg_vtt: Optional[Path] = None
                if subs_on and self._source_vtt:
                    tmp = CUTS_ROOT / f".{out.stem}.subs.vtt"
                    seg_vtt = _slice_vtt_window(
                        self._source_vtt, seg["start_s"], seg["end_s"], tmp)
                    if seg_vtt: self._temp_vtts.append(seg_vtt)
                subs_filter = (
                    _ffmpeg_subtitles_filter(seg_vtt, self.subs_cfg)
                    if seg_vtt else None
                )
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
                    # Append subs on top of the watermark composite.
                    map_v = "[out]"
                    if subs_filter:
                        fc = f"{fc};[out]{subs_filter}[vs]"
                        map_v = "[vs]"
                    cmd = [core.FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
                           "-ss", f"{seg['start_s']:.3f}",
                           "-i", str(self.source),
                           "-i", str(self.wm_path),
                           "-t", f"{duration:.3f}",
                           "-filter_complex", fc,
                           "-map", map_v, "-map", "0:a?",
                           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                           "-c:a", "aac", "-b:a", "128k",
                           "-movflags", "+faststart",
                           str(out)]
                elif self.precise or subs_filter:
                    # Subs force re-encode — stream-copy branch can't burn.
                    cmd = [core.FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
                           "-ss", f"{seg['start_s']:.3f}",
                           "-i", str(self.source),
                           "-t", f"{duration:.3f}"]
                    if subs_filter:
                        cmd += ["-vf", subs_filter]
                    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
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
            # Drop per-segment temp VTTs — the full-source `.whisper.*.vtt`
            # sidecar stays put (next Cut of the same source reuses it).
            for p in self._temp_vtts:
                try: p.unlink()
                except Exception: pass
            self._temp_vtts = []

    def _run_split_stack(self):
        """Single-pass split-stack render: crop two source regions, scale
        each to its half of a 1080x1920 canvas (cover-fit = scale-up to
        max-of-ratios, then center-crop the excess), vstack into one mp4.
        Output landed in CUTS_ROOT under `<stem>_splitstack_<ms>_<ms>.mp4`.
        """
        ss = self.split_stack
        start_t = float(ss["start_t"])
        end_t   = float(ss["end_t"])
        duration = max(0.001, end_t - start_t)
        split_pct = int(ss.get("split_pct") or 50)
        split_pct = max(10, min(100, split_pct))
        def _evn(v):
            v = int(v)
            return v - (v % 2)
        r1 = ss["r1"]
        x1, y1, w1, h1 = _evn(r1["x"]), _evn(r1["y"]), _evn(r1["w"]), _evn(r1["h"])
        if w1 < 2 or h1 < 2:
            raise ValueError("split-stack region too small")
        # Fit-WIDTH: width always = 1080 (no side bars). Vertical pads
        # with black if region's scaled height < slot, OR crops center
        # if scaled height > slot. Chain per slot:
        #   1. crop the user's region from source
        #   2. scale width to 1080, height auto-scales preserving aspect
        #   3. pad height up to slot height if shorter (centered, black)
        #   4. crop height down to slot height if taller (centered)
        # Comma inside max() is escaped \\, so ffmpeg parses it as part
        # of the expression rather than as a filter-arg separator.
        if split_pct >= 100:
            # Single-region variant: r1 fills the whole 1080x1920 canvas.
            fc = (
                f"[0:v]crop={w1}:{h1}:{x1}:{y1},"
                f"scale=1080:-2,"
                f"pad=1080:max(1920\\,ih):0:(oh-ih)/2:color=black,"
                f"crop=1080:1920:0:(ih-1920)/2,setsar=1[v]"
            )
        else:
            # Two-region stacked variant. 1080x1920 canvas, top H1, bot H2.
            H1 = int(round(1920 * split_pct / 100 / 2) * 2)
            H1 = max(2, min(1918, H1))
            H2 = 1920 - H1
            r2 = ss.get("r2")
            if not r2:
                raise ValueError("region2 required unless split_pct=100")
            x2, y2, w2, h2 = _evn(r2["x"]), _evn(r2["y"]), _evn(r2["w"]), _evn(r2["h"])
            if w2 < 2 or h2 < 2:
                raise ValueError("split-stack regions too small")
            fc = (
                f"[0:v]crop={w1}:{h1}:{x1}:{y1},"
                f"scale=1080:-2,"
                f"pad=1080:max({H1}\\,ih):0:(oh-ih)/2:color=black,"
                f"crop=1080:{H1}:0:(ih-{H1})/2,setsar=1[top];"
                f"[0:v]crop={w2}:{h2}:{x2}:{y2},"
                f"scale=1080:-2,"
                f"pad=1080:max({H2}\\,ih):0:(oh-ih)/2:color=black,"
                f"crop=1080:{H2}:0:(ih-{H2})/2,setsar=1[bot];"
                f"[top][bot]vstack=inputs=2[v]"
            )
        sms = int(start_t * 1000)
        ems = int(end_t * 1000)
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in self.source.stem)
        fname = f"{safe}_splitstack_{sms}_{ems}.mp4"
        out_path = CUTS_ROOT / fname
        # Subs: prepare full-source VTT (Whisper if no sidecar), slice
        # window to [start_t, end_t], append `subtitles=` after [v].
        subs_on = bool(self.subs_cfg.get("subs_enabled"))
        map_v = "[v]"
        if subs_on:
            if self._source_vtt is None:
                self.phase_msg = "гоню Whisper по всему видео"
                self._source_vtt = _prepare_source_subs(
                    self.source, self.subs_cfg,
                    on_phase=lambda m: setattr(self, "phase_msg", m))
            if self._source_vtt:
                tmp = CUTS_ROOT / f".{out_path.stem}.subs.vtt"
                seg_vtt = _slice_vtt_window(
                    self._source_vtt, start_t, end_t, tmp)
                if seg_vtt:
                    self._temp_vtts.append(seg_vtt)
                    subs_filter = _ffmpeg_subtitles_filter(seg_vtt, self.subs_cfg)
                    fc = f"{fc};[v]{subs_filter}[vs]"
                    map_v = "[vs]"
        self.phase_msg = (f"split-stack {start_t:.1f}s → {end_t:.1f}s "
                          f"(top {split_pct}%)")
        cmd = [
            core.FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
            "-ss", f"{start_t:.3f}",
            "-i", str(self.source),
            "-t", f"{duration:.3f}",
            "-filter_complex", fc,
            "-map", map_v, "-map", "0:a?",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            str(out_path),
        ]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                      stderr=subprocess.PIPE)
        _, stderr = self._proc.communicate(timeout=60 * 60)
        rc = self._proc.returncode
        self._proc = None
        if rc != 0:
            err = (stderr or b"").decode(errors="ignore")[:400]
            raise RuntimeError(f"ffmpeg exit {rc}: {err}")
        size = 0
        try: size = out_path.stat().st_size
        except Exception: pass
        self.outputs.append({
            "idx":      0,
            "file":     out_path.name,
            "url":      f"/downloads/cuts/{urllib.parse.quote(out_path.name)}",
            "start_s":  start_t,
            "end_s":    end_t,
            "duration": duration,
            "label":    "split-stack",
            "size_b":   size,
        })


# ---------- API helpers ----------
# ---------- Subtitle helpers (Cut editor only) ----------
# Cut needs to burn Whisper-generated subs into the SEGMENT it produces.
# The Whisper VTT covers the full source (0..source_duration); each cut
# segment covers [start_s, end_s]. So we slice the full VTT down to just
# the cues in that window and shift their timecodes by -start_s so ffmpeg
# `subtitles=` overlays them at wall-clock 0 of the output.

_VTT_TS_RE = re.compile(
    r"^(\d\d):(\d\d):(\d\d(?:\.\d+)?)\s+-->\s+(\d\d):(\d\d):(\d\d(?:\.\d+)?)"
)

def _vtt_secs(h: str, m: str, s: str) -> float:
    return int(h) * 3600 + int(m) * 60 + float(s)

def _fmt_vtt_ts(t: float) -> str:
    if t < 0: t = 0.0
    hh = int(t // 3600)
    mm = int((t % 3600) // 60)
    ss = t - hh * 3600 - mm * 60
    return f"{hh:02d}:{mm:02d}:{ss:06.3f}"

def _prepare_source_subs(source: Path, subs_cfg: dict,
                          on_phase=None) -> Optional[Path]:
    """Return the path to a full-source VTT for `source`. If a whisper
    sidecar for the requested language AND current model already exists
    next to the source, reuse it (instant). Otherwise run Whisper once.
    Returns None when subs are disabled in cfg. Model slug in sidecar
    name auto-invalidates cache when WHISPER_MODEL env var changes."""
    if not subs_cfg or not subs_cfg.get("subs_enabled"):
        return None
    lang = (subs_cfg.get("subs_lang") or "uk").lower()
    model_slug = re.sub(r"[^a-z0-9]+", "-",
                        core.WHISPER_MODEL_SIZE.lower()).strip("-")
    sidecar = source.with_name(
        f"{source.stem}.whisper.{lang}.{model_slug}.vtt")
    if sidecar.exists() and sidecar.stat().st_size > 20:
        return sidecar
    return subs_burn.generate_subs_via_whisper(
        source, on_phase=on_phase, language=lang)

def _slice_vtt_window(full_vtt: Path, start_s: float, end_s: float,
                       dst: Path) -> Optional[Path]:
    """Read `full_vtt` (source-timeline), keep only cues that overlap the
    [start_s, end_s] window, shift them so t=0 aligns with start_s, write
    result to `dst`. Returns dst (or None if window has no cues so subs
    filter can be skipped without failing the render)."""
    try:
        raw = full_vtt.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    out_lines = ["WEBVTT", ""]
    lines = raw.splitlines()
    i = 0
    kept_any = False
    while i < len(lines):
        line = lines[i]
        m = _VTT_TS_RE.match(line.strip())
        if not m:
            i += 1
            continue
        cue_start = _vtt_secs(m.group(1), m.group(2), m.group(3))
        cue_end   = _vtt_secs(m.group(4), m.group(5), m.group(6))
        # Collect cue text lines (until blank).
        j = i + 1
        text_lines = []
        while j < len(lines) and lines[j].strip():
            text_lines.append(lines[j])
            j += 1
        i = j + 1
        # Overlap test.
        if cue_end <= start_s or cue_start >= end_s:
            continue
        new_start = max(0.0, cue_start - start_s)
        new_end   = min(end_s - start_s, cue_end - start_s)
        if new_end <= new_start:
            continue
        out_lines.append(f"{_fmt_vtt_ts(new_start)} --> {_fmt_vtt_ts(new_end)}")
        out_lines.extend(text_lines)
        out_lines.append("")
        kept_any = True
    if not kept_any:
        return None
    dst.write_text("\n".join(out_lines), encoding="utf-8")
    return dst

def _ffmpeg_subtitles_filter(vtt_path: Path, subs_cfg: dict) -> str:
    """Compose the `subtitles=path:force_style=…` filter arg. Path needs
    ffmpeg-special-char escaping (`:`, `\\`, `'`). We normalize to forward
    slashes and escape `:` and `'`. Windows drive letter `C:` gets `C\\:`."""
    p = str(vtt_path).replace("\\", "/")
    # Escape ffmpeg filter separator and quote chars.
    p_esc = p.replace(":", "\\:").replace("'", r"\'")
    fs = subs_burn.build_force_style(subs_cfg)
    return f"subtitles='{p_esc}':force_style='{fs}'"


def start_cut(source_path: Path, segments: List[dict],
              precise: bool = False, label: str = "",
              wm_cfg: Optional[dict] = None,
              subs_cfg: Optional[dict] = None) -> CutJob:
    if not source_path.exists():
        raise ValueError(f"source missing: {source_path}")
    if not segments:
        raise ValueError("no segments to cut")
    return CutJob(source_path, segments,
                  precise=precise, label=label,
                  wm_cfg=wm_cfg, subs_cfg=subs_cfg).start()


def start_split_stack(source_path: Path, start_t: float, end_t: float,
                      r1: dict, r2: Optional[dict], split_pct: int = 50,
                      label: str = "",
                      wm_cfg: Optional[dict] = None,
                      subs_cfg: Optional[dict] = None) -> CutJob:
    """Kick off a split-stack render. Reuses CutJob so the status/list/
    delete plumbing all work as-is. `segments=[]` is intentional — the
    runner takes the split-stack branch when `job.split_stack` is set.

    At `split_pct == 100`, r2 may be None: the top region fills the whole
    1080x1920 canvas (single-region 9:16, no vstack)."""
    if not source_path.exists():
        raise ValueError(f"source missing: {source_path}")
    if end_t <= start_t:
        raise ValueError("end_t must be > start_t")
    single = int(split_pct) >= 100
    if not single and r2 is None:
        raise ValueError("region2 required unless split_pct=100")
    rects = [("region1", r1)]
    if r2 is not None:
        rects.append(("region2", r2))
    for tag, rect in rects:
        if not isinstance(rect, dict):
            raise ValueError(f"{tag} must be a rect dict")
        for k in ("x", "y", "w", "h"):
            if k not in rect:
                raise ValueError(f"{tag} missing key: {k}")
    job = CutJob(source_path, segments=[], precise=True,
                 label=label or source_path.stem,
                 wm_cfg=wm_cfg, subs_cfg=subs_cfg)
    job.split_stack = {
        "start_t":   float(start_t),
        "end_t":     float(end_t),
        "r1":        {k: int(r1[k]) for k in ("x","y","w","h")},
        "r2":        ({k: int(r2[k]) for k in ("x","y","w","h")} if r2 else None),
        "split_pct": int(split_pct),
    }
    return job.start()


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
    import twitch as tw_mod
    roots = {
        "youtube": dl_mod.DOWNLOADS_ROOT,
        "insta":   ig_mod.INSTA_ROOT,
        "tiktok":  tt_mod.TIKTOK_ROOT,
        "twitch":  tw_mod.TWITCH_ROOT,
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
