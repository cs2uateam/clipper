"""Tool-path resolver + tiny shared constants. Lifted from
`youtube-analyzer/v1/core.py` but stripped to the surface this project
actually uses (FFMPEG / FFPROBE / YTDLP paths, Whisper config, the
YouTube ID regex). Keeping this module independent so `clipper` doesn't
import from the analyzer project across project boundaries."""
import os, re, shutil
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


# Whisper model config — defaults match the analyzer side so the cache
# directory (~/.cache/...) stays shared and we don't re-download models.
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "tiny")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE = os.environ.get("WHISPER_COMPUTE",
                                 "float16" if WHISPER_DEVICE == "cuda" else "int8")
