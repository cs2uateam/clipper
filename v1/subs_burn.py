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

# ---------- Burn-in ----------
def burn_subtitles(
    video_path: Path,
    cfg: dict,
    on_phase: PhaseCb = None,
    on_proc: ProcCb = None,
    timeout_s: int = 60 * 60,
) -> None:
    """Re-encode `video_path` with subs burned into the picture. Picks
    the best .vtt sidecar (yt-source > whisper-generated) and applies
    user style preferences. Replaces the original file in place — the
    ffmpeg output goes to a temp file then renames over the original.
    Raises if no .vtt found or ffmpeg fails."""
    candidates = find_subtitle_files(video_path)
    if not candidates:
        raise RuntimeError("subtitle file not found in output dir")
    yt_subs = [c for c in candidates if ".whisper" not in c.name]
    subs = yt_subs[0] if yt_subs else candidates[0]
    # YouTube auto-caption rolling format → clean before burning.
    if ".whisper" not in subs.name:
        try: clean_yt_rolling_vtt(subs)
        except Exception: pass
    if on_phase: on_phase(f"вшиваю субтитры ({subs.name}) — re-encoding")

    # ffmpeg subtitles filter wants forward slashes + escaped ':' on Windows
    subs_arg = str(subs).replace("\\", "/").replace(":", "\\:")
    style = build_force_style(cfg)
    vf = f"subtitles='{subs_arg}':force_style='{style}'"
    tmp = video_path.with_name(video_path.stem + ".__burning.mp4")
    cmd = [core.FFMPEG, "-y", "-loglevel", "error",
           "-i", str(video_path),
           "-vf", vf,
           "-c:v", "libx264", "-preset", "medium", "-crf", "20",
           "-c:a", "copy",
           "-movflags", "+faststart",
           str(tmp)]
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
    if on_phase: on_phase("")

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
