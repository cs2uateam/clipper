"""Tool-path resolver + tiny shared constants. Lifted from
`youtube-analyzer/v1/core.py` but stripped to the surface this project
actually uses (FFMPEG / FFPROBE / YTDLP paths, Whisper config, the
YouTube ID regex). Keeping this module independent so `clipper` doesn't
import from the analyzer project across project boundaries."""
import os, re, shutil, subprocess
from pathlib import Path

V1 = Path(__file__).resolve().parent
BIN = V1 / "bin"


def _find_exec(name, env_var=None, fallbacks=()):
    """Prefer env var → bundled ./bin → sibling vidos-analyzer/v4 → PATH
    → user-supplied fallbacks. Same lookup order as the analyzer side
    so both projects pick up the same toolchain on this machine."""
    if env_var:
        v = os.environ.get(env_var)
        if v and os.path.exists(v): return v
    is_win = os.name == "nt"
    suffix = ".exe" if is_win else ""
    bin_local = BIN / (name + suffix)
    if bin_local.exists(): return str(bin_local)
    # Same fallback the analyzer uses — keeps the existing winget /
    # bundled binaries reachable from this project too.
    sibling_bin = V1.parent.parent / "vidos-analyzer" / "v4" / "bin" / (name + suffix)
    if sibling_bin.exists(): return str(sibling_bin)
    found = shutil.which(name + suffix) or shutil.which(name)
    if found: return found
    for f in fallbacks:
        f = os.path.expanduser(f)
        if os.path.exists(f): return f
    return name


FFMPEG = _find_exec("ffmpeg", "FFMPEG_PATH")
FFPROBE = _find_exec("ffprobe", "FFPROBE_PATH")
YTDLP = _find_exec("yt-dlp", "YTDLP_PATH",
                    fallbacks=("~/Library/Python/3.9/bin/yt-dlp",
                               "~/.local/bin/yt-dlp"))


def yt_id_from_url(url):
    m = re.search(
        r"(?:youtu\.be/|youtube\.com/(?:watch\?(?:[^#]*&)?v=|live/|shorts/|embed/|v/))"
        r"([A-Za-z0-9_-]{11})", url)
    if not m: raise ValueError(f"bad YouTube url: {url}")
    return m.group(1)


# Aspect-ratio post-process for downloaded video. Output dimensions match
# the analyzer side (kept in sync so a clip from analyzer and a download
# from clipper land at the same canvas).
ASPECT_PRESETS = {
    "9:16":  (1080, 1920),
    "16:9":  (1920, 1080),
    "1:1":   (1080, 1080),
    "4:5":   (1080, 1350),
    "21:9":  (1920, 822),
}

def parse_aspect(aspect: str):
    """Return (out_w, out_h, rw, rh) for a known preset OR for any
    `W:H` string. Bad input falls back to 9:16."""
    if not aspect: return (1080, 1920, 9, 16)
    if aspect in ASPECT_PRESETS:
        w, h = ASPECT_PRESETS[aspect]
        rw, rh = aspect.split(":")
        return (w, h, int(rw), int(rh))
    m = re.match(r"^\s*(\d+)\s*:\s*(\d+)\s*$", aspect)
    if not m: return (1080, 1920, 9, 16)
    rw, rh = int(m.group(1)), int(m.group(2))
    if rw >= rh:
        out_w, out_h = 1920, max(360, int(round(1920 * rh / rw)))
    else:
        out_h, out_w = 1920, max(360, int(round(1920 * rw / rh)))
    out_w -= out_w % 2; out_h -= out_h % 2
    return (out_w, out_h, rw, rh)

def aspect_vf(aspect: str) -> str:
    """ffmpeg -vf string: pure CENTER CROP to target aspect — no scale.
    Source ratio is unknown at filter-build time (could be 16:9 horizontal
    OR 9:16 vertical Shorts), so we let ffmpeg branch at decode time
    via `if(gte(iw/ih, R), …, …)`:
    - source wider than target  → keep ih, crop sides
    - source taller than target → keep iw, crop top+bottom
    `floor(.../2)*2` keeps both output dims even — libx264/yuv420p chokes
    on odd values."""
    _, _, rw, rh = parse_aspect(aspect)
    # ffmpeg expression parser is left-associative: `iw/16/9` evaluates
    # as `(iw/16)/9`, not `iw/(16/9)`. We MUST wrap the ratio in parens
    # whenever we divide BY it, or the crop dimensions collapse to a
    # 2-pixel-tall sliver. (Real bug we hit — once.)
    R = f"({rw}/{rh})"
    # Inside a filter arg, `,` separates parameters — inner commas in
    # `if(...)` must be escaped with `\,` so ffmpeg sees them as part
    # of the expression. In Python string we write `\\,`.
    cw = f"if(gte(iw/ih\\,{R})\\,floor(ih*{R}/2)*2\\,floor(iw/2)*2)"
    ch = f"if(gte(iw/ih\\,{R})\\,floor(ih/2)*2\\,floor(iw/{R}/2)*2)"
    cx = f"if(gte(iw/ih\\,{R})\\,(iw-ih*{R})/2\\,0)"
    cy = f"if(gte(iw/ih\\,{R})\\,0\\,(ih-iw/{R})/2)"
    return f"crop={cw}:{ch}:{cx}:{cy}"


# Whisper model config — defaults match the analyzer side so the cache
# directory (~/.cache/...) stays shared and we don't re-download models.
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "large-v3")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE = os.environ.get("WHISPER_COMPUTE",
                                 "float16" if WHISPER_DEVICE == "cuda" else "int8")


def hide_file(path):
    """Mark a sidecar file as hidden so it doesn't clutter the user's
    File Explorer view next to the media. The file still exists and is
    readable by code — we set the Windows hidden attribute via attrib.
    No-op on non-Windows; on Linux/macOS the convention is leading-dot
    filenames, but our sidecar names already match those of the media
    file so we can't rename without breaking lookup logic."""
    if os.name != "nt": return
    try:
        subprocess.run(["attrib", "+H", str(path)],
                       capture_output=True, check=False)
    except Exception: pass
