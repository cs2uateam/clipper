"""Shared subtitle generation + burn-in pipeline.

Used by:
  - download.py (yt-dlp pulls video + YouTube .vtt → burn into mp4)
  - upload.py   (user-supplied video → Whisper → burn into mp4)

A single source of truth for the styling presets, the rolling-VTT
cleaner, the Whisper fallback, and the ffmpeg subtitles-filter command.
Future style/burn fixes go here once and propagate everywhere.

Callbacks (on_phase, on_proc) let each Job class plug in its own status
plumbing without subs_burn caring about it.
"""
import re, subprocess, time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import core

# ---------- Style presets ----------
# Fontsize values are in ASS PlayResY=288 units — libass scales them to
# the actual rendered video height. So Fontsize=10 means ~3.5% of frame
# height regardless of whether the video is 720p horizontal or 1920
# vertical. ASS colour format is `&HAABBGGRR` (NOT RGB) — leading 'AA'
# is alpha (00 = fully opaque, FF = fully transparent).
SUBS_SIZES = {
    "small":   7,    # ~2.4% of frame height
    "medium": 10,    # ~3.5% — TikTok/Reels default
    "large":  13,    # ~4.5%
    "huge":   17,    # ~5.9%
}
SUBS_COLORS = {
    "white":  "&H00FFFFFF",
    "yellow": "&H0000FFFF",
    "red":    "&H000000FF",
    "green":  "&H0000FF00",
    "cyan":   "&H00FFFF00",
    "black":  "&H00000000",
}
SUBS_BG = {
    # BorderStyle=1 = outline+shadow only (no fill); =3 = opaque rectangle
    "none":      {"BorderStyle": 1, "Outline": 3, "Shadow": 1,
                  "OutlineColour": "&H00000000"},
    "shadow":    {"BorderStyle": 1, "Outline": 1, "Shadow": 3,
                  "OutlineColour": "&H80000000"},
    "dark_box":  {"BorderStyle": 3, "Outline": 2, "Shadow": 0,
                  "OutlineColour": "&H80000000"},
    "solid_box": {"BorderStyle": 3, "Outline": 2, "Shadow": 0,
                  "OutlineColour": "&H00000000"},
}
SUBS_POSITION = {
    "bottom": 2,    # ASS Alignment: 1-3 bottom, 4-6 middle, 7-9 top, +0/+1/+2 = L/C/R
    "top":    8,
    "center": 5,
}

PhaseCb = Optional[Callable[[str], None]]
ProcCb  = Optional[Callable[[subprocess.Popen], None]]

# ---------- VTT helpers ----------
def vtt_ts(t: float) -> str:
    """Format seconds as a VTT timestamp 'HH:MM:SS.mmm'."""
    if t < 0: t = 0.0
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t - h * 3600 - m * 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"

def clean_yt_rolling_vtt(vtt_path: Path) -> None:
    """YouTube auto-caption VTT uses an accumulating ('rolling') format:
    each cue's text repeats everything that was already said and tacks on
    the new words. Burned via ffmpeg the result looks like 6 stacked lines
    climbing up the screen. Rewrites the VTT in place so each cue shows
    ONLY the words newly spoken in its time window — strips inline word-
    timing tags, drops cue settings, dedupes prefix-overlap."""
    if not vtt_path.exists(): return
    raw = vtt_path.read_text(encoding="utf-8", errors="ignore")
    # Strip inline word timestamps like `<00:00:01.120>` and `<c>…</c>` wrappers
    raw = re.sub(r"<\d{2}:\d{2}:\d{2}\.\d{3}>", "", raw)
    raw = re.sub(r"</?c(?:\.[^>]*)?>", "", raw)

    cues = []
    for chunk in raw.split("\n\n"):
        lines = [l.rstrip() for l in chunk.split("\n") if l.strip()]
        if not lines: continue
        ts_line = next((l for l in lines if "-->" in l), None)
        if not ts_line: continue
        ts_clean = ts_line.split(" align")[0].split(" position")[0].split(" line")[0]
        text_lines = [l for l in lines if "-->" not in l
                      and not l.startswith("WEBVTT")
                      and not l.startswith("Kind:")
                      and not l.startswith("Language:")]
        text = re.sub(r"\s+", " ", " ".join(text_lines)).strip()
        if not text: continue
        cues.append((ts_clean, text))

    out = ["WEBVTT", ""]
    prev_full = ""
    for ts, full in cues:
        if prev_full and full.startswith(prev_full):
            new_text = full[len(prev_full):].strip()
        else:
            new_text = full
        prev_full = full
        if not new_text: continue
        out.append(ts)
        out.append(new_text)
        out.append("")
    vtt_path.write_text("\n".join(out), encoding="utf-8")
    core.hide_file(vtt_path)

# ---------- File / dimension helpers ----------
def find_subtitle_files(video_path: Path, *,
                         exclude_whisper: bool = False) -> List[Path]:
    """All .vtt files in the same dir as `video_path` whose filename
    starts with the video's stem. Manual scan (not Path.glob) because
    YouTube IDs in filenames have square brackets which glob treats as
    a character class."""
    if video_path is None: return []
    parent = video_path.parent
    if not parent.exists(): return []
    stem = video_path.stem
    out = []
    for f in parent.iterdir():
        if not f.is_file(): continue
        if f.suffix.lower() != ".vtt": continue
        if not f.name.startswith(stem): continue
        if exclude_whisper and ".whisper" in f.name: continue
        out.append(f)
    return out

def video_dimensions(path: Path) -> Tuple[int, int]:
    """Returns (width, height) of the first video stream. Falls back to
    (1280, 720) if ffprobe fails."""
    try:
        r = subprocess.run(
            [core.FFPROBE, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=30)
        if r.returncode == 0 and r.stdout.strip():
            w, h = r.stdout.strip().split(",")
            return int(w), int(h)
    except Exception: pass
    return 1280, 720

# ---------- Style builder ----------
def build_force_style(cfg: dict) -> str:
    """Compose the ffmpeg `force_style=...` string from user-picked
    options. All numeric values are in ASS PlayResY=288 units so libass
    can scale to any rendered resolution."""
    font_size = SUBS_SIZES.get((cfg.get("subs_size") or "medium"), 10)
    color     = SUBS_COLORS.get((cfg.get("subs_color") or "white"), "&H00FFFFFF")
    bg        = SUBS_BG.get((cfg.get("subs_bg") or "none"), SUBS_BG["none"])
    align     = SUBS_POSITION.get((cfg.get("subs_position") or "bottom"), 2)
    margin_v  = 17    # ~6% of frame height in ASS units
    parts = [
        "FontName=Arial",
        f"Fontsize={font_size}",
        "Bold=1",
        f"PrimaryColour={color}",
        f"OutlineColour={bg['OutlineColour']}",
        f"BorderStyle={bg['BorderStyle']}",
        f"Outline={bg['Outline']}",
        f"Shadow={bg['Shadow']}",
        f"Alignment={align}",
        f"MarginV={margin_v}",
    ]
    return ",".join(parts)

# ---------- Whisper fallback ----------
def generate_subs_via_whisper(
    video_path: Path,
    on_phase: PhaseCb = None,
) -> Path:
    """Transcribe the video's audio via Whisper (English-pinned) and
    write `<stem>.whisper.en.vtt` next to it. Returns that path.
    Used when YouTube has no captions OR for user-uploaded files where
    no captions exist."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise RuntimeError(
            "faster-whisper не установлен — pip install faster-whisper")
    if on_phase: on_phase("гоню Whisper")
    model = WhisperModel(
        core.WHISPER_MODEL_SIZE,
        device=core.WHISPER_DEVICE,
        compute_type=core.WHISPER_COMPUTE)
    segments, _info = model.transcribe(
        str(video_path), vad_filter=True, beam_size=1, language="en")
    vtt_path = video_path.with_name(video_path.stem + ".whisper.en.vtt")
    with open(vtt_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for seg in segments:
            text = (seg.text or "").strip()
            if not text: continue
            f.write(f"{vtt_ts(seg.start)} --> {vtt_ts(seg.end)}\n{text}\n\n")
    # Hide from File Explorer — sidecar is a working file, not user content.
    core.hide_file(vtt_path)
    if on_phase: on_phase("")
    return vtt_path

# ---------- Watermark helpers ----------
WATERMARKS_DIR = Path(__file__).resolve().parent / "watermarks"
WATERMARKS_DIR.mkdir(exist_ok=True)

def _watermark_xy(position: str) -> str:
    """ffmpeg overlay `x:y` expr for a position name. W,H = main video
    dims; w,h = watermark dims. 20px margin from edges for corners."""
    pos = (position or "center").lower()
    if pos == "br": return "W-w-20:H-h-20"
    if pos == "tr": return "W-w-20:20"
    if pos == "bl": return "20:H-h-20"
    if pos == "tl": return "20:20"
    return "(W-w)/2:(H-h)/2"

def resolve_watermark(cfg: dict) -> Optional[Path]:
    """Public wrapper around _resolve_watermark for callers outside this
    module (the cutter uses this to know whether to take the re-encode
    branch). Identical behaviour."""
    return _resolve_watermark(cfg)

def build_watermark_filter_complex(
    out_w: int,
    out_h: int,
    wm_path: Path,
    cfg: dict,
    in_label: str = "0:v",
    out_label: str = "out",
) -> str:
    """Construct the `-filter_complex` expression that overlays `wm_path`
    onto the video stream `[<in_label>]` and labels the result
    `[<out_label>]`. Shared between subs_burn's own burn pass and the
    cutter's single-pass cut+watermark mode.

    Honours `watermark_size` (% of out_w), `watermark_opacity` (%), and
    `watermark_position` ('center'/'tl'/'tr'/'bl'/'br'/'fill'). The
    'fill' mode tiles the watermark across the whole canvas via split+
    N overlays — the same approach used in the analyzer."""
    try: wm_pct = max(10, min(100, int(cfg.get("watermark_size") or 20))) / 100.0
    except (TypeError, ValueError): wm_pct = 0.20
    wm_w = max(40, int(out_w * wm_pct))
    try: opacity = max(0.10, min(1.0,
                                  int(cfg.get("watermark_opacity") or 70) / 100.0))
    except (TypeError, ValueError): opacity = 0.70
    position = (cfg.get("watermark_position") or "center").lower()

    if position == "fill":
        pad = max(20, wm_w // 8)
        # Probe the watermark image so the grid math accounts for its
        # real aspect — square fallback when probe fails.
        wm_h_real = wm_w
        try:
            pr = subprocess.run(
                [core.FFPROBE, "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height",
                 "-of", "csv=p=0", str(wm_path)],
                capture_output=True, text=True, timeout=10)
            pw, ph = pr.stdout.strip().split(",")
            wm_h_real = max(1, int(int(ph) * wm_w / int(pw)))
        except Exception: pass
        cols = max(1, (out_w // (wm_w + pad)) + 1)
        rows = max(1, (out_h // (wm_h_real + pad)) + 1)
        n = cols * rows
        split_labels = "".join(f"[wm{i}]" for i in range(n))
        chain_parts = [
            f"[1:v]scale={wm_w}:-1,format=rgba,"
            f"colorchannelmixer=aa={opacity},split={n}{split_labels}"
        ]
        step_w = wm_w + pad
        step_h = wm_h_real + pad
        grid_w = cols * wm_w + (cols - 1) * pad
        grid_h = rows * wm_h_real + (rows - 1) * pad
        x0 = max(0, (out_w - grid_w) // 2)
        y0 = max(0, (out_h - grid_h) // 2)
        cur = f"[{in_label}]"
        for i in range(n):
            r = i // cols; c = i % cols
            x = x0 + c * step_w
            y = y0 + r * step_h
            stage_out = f"[{out_label}]" if i == n - 1 else f"[v{i+1}]"
            chain_parts.append(f"{cur}[wm{i}]overlay={x}:{y}{stage_out}")
            cur = f"[v{i+1}]"
        return ";".join(chain_parts)
    xy = _watermark_xy(position)
    return (f"[1:v]scale={wm_w}:-1,format=rgba,"
            f"colorchannelmixer=aa={opacity}[wm];"
            f"[{in_label}][wm]overlay={xy}[{out_label}]")

def _resolve_watermark(cfg: dict) -> Optional[Path]:
    """Resolve cfg → absolute Path or None. Accepts either an already-
    resolved `watermark_path` (preferred — caller validated existence)
    or a bare `watermark_name` we look up under WATERMARKS_DIR. Returns
    None if no watermark requested, the file is missing, or the position
    is the sentinel 'none'."""
    if (cfg.get("watermark_position") or "").lower() == "none":
        return None
    wp = cfg.get("watermark_path")
    if wp:
        p = Path(wp)
        return p if p.exists() else None
    name = (cfg.get("watermark_name") or "").strip()
    if not name: return None
    p = WATERMARKS_DIR / name
    return p if p.exists() else None

def _build_overlay_pass(
    video_path: Path,
    cfg: dict,
    want_subs: bool,
) -> Tuple[Optional[list], Optional[Path], str]:
    """Build a SINGLE ffmpeg command that re-encodes `video_path` with
    subtitles ± watermark composited in one pass. Returns
    (cmd, tmp_out_path, summary_msg). cmd is None when there's nothing
    to do (e.g. want_subs=False and no watermark).

    Watermark is composited UNDER subtitles so captions stay readable
    over a 'fill'-tile background."""
    wm_path = _resolve_watermark(cfg)

    subs_path = None
    if want_subs:
        candidates = find_subtitle_files(video_path)
        if not candidates:
            raise RuntimeError("subtitle file not found in output dir")
        yt_subs = [c for c in candidates if ".whisper" not in c.name]
        subs_path = yt_subs[0] if yt_subs else candidates[0]
        if ".whisper" not in subs_path.name:
            try: clean_yt_rolling_vtt(subs_path)
            except Exception: pass

    if subs_path is None and wm_path is None:
        return None, None, ""

    # Build subtitles filter expr (no leading [label])
    subs_filter = None
    if subs_path:
        # ffmpeg subtitles filter wants forward slashes + escaped ':' on Win
        subs_arg = str(subs_path).replace("\\", "/").replace(":", "\\:")
        style = build_force_style(cfg)
        subs_filter = f"subtitles='{subs_arg}':force_style='{style}'"

    tmp = video_path.with_name(video_path.stem + ".__burning.mp4")

    # Subs-only branch: keep the original simple -vf path. No filter_complex
    # overhead, no 2nd input. Behaviour unchanged from before watermarks.
    if wm_path is None:
        cmd = [core.FFMPEG, "-y", "-loglevel", "error",
               "-i", str(video_path),
               "-vf", subs_filter,
               "-c:v", "libx264", "-preset", "medium", "-crf", "20",
               "-c:a", "copy",
               "-movflags", "+faststart",
               str(tmp)]
        return cmd, tmp, f"subs={subs_path.name}"

    # Watermark branch (with or without subs) → filter_complex. When
    # subs follow, the wm chain ends at [wmout] so the subs filter can
    # consume it; otherwise it ends directly at [out].
    out_w, out_h = video_dimensions(video_path)
    wm_chain_out = "wmout" if subs_filter else "out"
    wm_chain = build_watermark_filter_complex(
        out_w, out_h, wm_path, cfg,
        in_label="0:v", out_label=wm_chain_out)
    if subs_filter:
        # subs after watermark → captions sit on top, stay readable
        fc = wm_chain + f";[wmout]{subs_filter}[out]"
    else:
        fc = wm_chain

    cmd = [core.FFMPEG, "-y", "-loglevel", "error",
           "-i", str(video_path), "-i", str(wm_path),
           "-filter_complex", fc,
           "-map", "[out]", "-map", "0:a?",
           "-c:v", "libx264", "-preset", "medium", "-crf", "20",
           "-c:a", "copy",
           "-movflags", "+faststart",
           str(tmp)]
    msg = f"wm={wm_path.name} ({position})"
    if subs_path: msg = f"subs={subs_path.name} + " + msg
    return cmd, tmp, msg

def _run_overlay_cmd(
    video_path: Path,
    cmd: list,
    tmp: Path,
    on_proc: ProcCb,
    timeout_s: int,
) -> None:
    """Run a prepared ffmpeg overlay cmd, rename tmp → video on success."""
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                             stderr=subprocess.PIPE)
    if on_proc: on_proc(proc)
    try:
        _, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        try: proc.kill()
        except Exception: pass
        try: tmp.unlink()
        except Exception: pass
        raise RuntimeError("ffmpeg burn timed out")
    if proc.returncode != 0:
        try: tmp.unlink()
        except Exception: pass
        err = (stderr or b"").decode(errors="ignore")[:300]
        raise RuntimeError(f"ffmpeg failed: {err}")
    try: video_path.unlink()
    except Exception: pass
    tmp.rename(video_path)

# ---------- Burn-in ----------
def burn_subtitles(
    video_path: Path,
    cfg: dict,
    on_phase: PhaseCb = None,
    on_proc: ProcCb = None,
    timeout_s: int = 60 * 60,
) -> None:
    """Re-encode `video_path` with subtitles burned into the picture
    (plus optional watermark composited in the same pass when cfg has
    `watermark_name`/`watermark_path` set). Picks the best .vtt sidecar
    (yt-source > whisper-generated) and applies user style prefs.
    Replaces the original file in place. Raises if no .vtt found or
    ffmpeg fails."""
    cmd, tmp, msg = _build_overlay_pass(video_path, cfg, want_subs=True)
    if cmd is None:
        # Should never happen for burn_subtitles (subs required) — but
        # _build_overlay_pass would've raised. Guard anyway.
        raise RuntimeError("burn_subtitles: no subs and no watermark")
    if on_phase: on_phase(f"вшиваю ({msg}) — re-encoding")
    _run_overlay_cmd(video_path, cmd, tmp, on_proc, timeout_s)
    if on_phase: on_phase("")

def apply_watermark(
    video_path: Path,
    cfg: dict,
    on_phase: PhaseCb = None,
    on_proc: ProcCb = None,
    timeout_s: int = 60 * 60,
) -> bool:
    """Re-encode `video_path` to composite a watermark image (without
    touching subtitles). Used when the user wants a watermark but no
    subs burn-in. Returns True if applied, False if no watermark was
    configured (no-op)."""
    if _resolve_watermark(cfg) is None:
        return False
    cmd, tmp, msg = _build_overlay_pass(video_path, cfg, want_subs=False)
    if cmd is None: return False
    if on_phase: on_phase(f"наношу watermark ({msg}) — re-encoding")
    _run_overlay_cmd(video_path, cmd, tmp, on_proc, timeout_s)
    if on_phase: on_phase("")
    return True

# ---------- File-name sanitizer (used by upload to make safe paths) ----------
_UNSAFE_FN = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
def safe_filename(name: str, fallback: str = "uploaded.mp4") -> str:
    """Strip path separators and OS-illegal chars from a user-supplied
    filename; collapse whitespace; cap length."""
    if not name: return fallback
    cleaned = _UNSAFE_FN.sub("_", name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.lstrip(".") or fallback
    if len(cleaned) > 180:
        # Keep extension
        p = Path(cleaned)
        cleaned = p.stem[:170] + p.suffix
    return cleaned or fallback
