"""YouTube live-stream section grabber.

Mirror of `download.py` but for currently-running YT live broadcasts:
user pastes a stream URL + From/To absolute timecodes (since broadcast
start), we pull that HLS slice from YT's DVR buffer and write a mp4.

Implementation: streamlink reads the DVR manifest with `--hls-live-restart`
and seeks via `--hls-start-offset` / `--hls-duration`. Firefox YT cookies
are injected via `--http-cookie KEY=VALUE` (one flag per cookie) so the
download is treated as an authed session by googlevideo (otherwise YT
throttles to ~30 KB/s). Streamlink writes a .ts file; ffmpeg then
transmuxes to .mp4 with stream-copy (no re-encode, seconds-fast).

YT throttles live-DVR even with auth to ~2× real-time — that's a
server-side cap, no client trick gets around it. Expect ~10 minutes
of wall-clock for a 20-minute window.

Files land in `clipper/v1/downloads/youtube_live/<vid>_<start>-<end>.mp4`
and are served via the existing `/downloads/{path}` route (sibling of
`youtube/`, `insta/`, `tiktok/`)."""
import os, re, shutil, subprocess, threading, time, urllib.parse
from pathlib import Path
from typing import Dict, Optional

import core


V1 = Path(__file__).resolve().parent
DOWNLOADS_ROOT = V1 / "downloads" / "youtube_live"
DOWNLOADS_ROOT.mkdir(parents=True, exist_ok=True)

_LOCK = threading.Lock()
JOBS: Dict[str, "LiveDlJob"] = {}


def _find_streamlink() -> str:
    env = os.environ.get("STREAMLINK_PATH")
    if env and os.path.exists(env):
        return env
    name = "streamlink.exe" if os.name == "nt" else "streamlink"
    found = shutil.which(name) or shutil.which("streamlink")
    return found or "streamlink"


STREAMLINK = _find_streamlink()


def _hms(sec) -> str:
    sec = max(0, int(sec))
    return f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


def _firefox_yt_cookie_args() -> list:
    """Read YouTube cookies from Firefox via browser_cookie3 and build
    streamlink `--http-cookie KEY=VALUE` flags. Empty list if Firefox
    isn't installed or no cookies for youtube.com — streamlink will run
    unauthenticated and YT will throttle hard."""
    try:
        import browser_cookie3
        jar = browser_cookie3.firefox(domain_name=".youtube.com")
        out: list = []
        for c in jar:
            out += ["--http-cookie", f"{c.name}={c.value}"]
        return out
    except Exception:
        return []


_FIREFOX_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:137.0) "
               "Gecko/20100101 Firefox/137.0")


# ----- probe (yt-dlp -J + Firefox cookies → now_s + title) -----

def probe_live_duration(url: str) -> dict:
    """Return {ok, is_live, now_s, title, diag} for a YT live URL.
    Mirror of youtube-analyzer/v1/live_section.py::probe_live, kept
    in this module so the clipper doesn't have to import from a
    sibling project.

    now_s = current DVR-buffer length in seconds. Strategy:
      1. meta.duration / meta.live_duration if populated (rare for live).
      2. now() - meta.release_timestamp / timestamp (canonical for YT live).
      3. Cap at 12h — typical YT DVR window. For 24/7 streams this is
         the real maximum we can rewind anyway.

    NEVER blocks for long: 90s timeout on yt-dlp."""
    import json as _json
    out = {
        "ok": False, "is_live": False, "now_s": None,
        "title": None, "diag": {},
    }
    cmd = [
        core.YTDLP, "-J", "--no-warnings", "--skip-download",
        "--cookies-from-browser", "firefox",
        url,
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=90)
    except subprocess.TimeoutExpired:
        out["diag"]["error"] = "yt-dlp probe timed out"
        return out
    except FileNotFoundError:
        out["diag"]["error"] = "yt-dlp not installed"
        return out
    raw = (p.stdout or b"").decode(errors="ignore").strip()
    err = (p.stderr or b"").decode(errors="ignore").strip()
    if p.returncode != 0 or not raw or raw == "null":
        err_line = (err.splitlines() or [""])[-1][:300]
        out["diag"]["error"] = err_line or "yt-dlp returned no JSON"
        return out
    try:
        meta = _json.loads(raw)
    except _json.JSONDecodeError as e:
        out["diag"]["error"] = f"yt-dlp JSON parse: {e}"
        return out
    out["title"] = (meta.get("title") or "")[:200] or None
    is_live = bool(meta.get("is_live")
                   or (meta.get("live_status") == "is_live"))
    out["is_live"] = is_live
    if not is_live:
        out["diag"]["error"] = "не идущий live-стрим"
        out["diag"]["live_status"] = meta.get("live_status")
        return out
    # Compute now_s.
    now_s = None
    for k in ("duration", "live_duration"):
        v = meta.get(k)
        if isinstance(v, (int, float)) and v > 0:
            now_s = int(v); break
    if now_s is None:
        now = int(time.time())
        for k in ("release_timestamp", "timestamp"):
            v = meta.get(k)
            if isinstance(v, (int, float)) and v > 0 and v < now:
                cand = now - int(v)
                if cand >= 60:
                    now_s = min(cand, 12 * 3600)
                    break
    if not now_s or now_s < 30:
        out["diag"]["error"] = "не смог определить длину стрима"
        out["diag"]["release_timestamp"] = meta.get("release_timestamp")
        out["diag"]["timestamp"] = meta.get("timestamp")
        return out
    out["ok"] = True
    out["now_s"] = int(now_s)
    return out


# ----- streamlink quality name mapping -----

def _streamlink_quality(q: str) -> str:
    """Build a comma-separated quality fallback list for streamlink.

    YT live's streamlink plugin names variants by WIDTH (not height) +
    optional fps suffix: `1920p60` = 1080p60, `1280p60` = 720p60.
    Older streams use height-based names (`1080p`, `720p`). We pass both
    candidates plus `best` so streamlink picks whichever is available."""
    q = (q or "1080").lower()
    if q in ("best", "max"): return "best"
    if q in ("worst", "min"): return "worst"
    if q in ("2160", "2160p", "4k"):
        return "3840p60,3840p,2160p60,2160p,best"
    if q in ("1440", "1440p", "qhd"):
        return "2560p60,2560p,1440p60,1440p,best"
    if q in ("1080", "1080p", "fhd"):
        return "1920p60,1920p,1080p60,1080p,best"
    if q in ("720", "720p"):
        return "1280p60,1280p,720p60,720p,best"
    if q in ("480", "480p"):
        return "854p,480p,best"
    if q in ("360", "360p"):
        return "640p,360p,best"
    if q in ("240", "240p"):
        return "426p,240p,worst"
    if q in ("144", "144p"):
        return "256p,144p,worst"
    return "1920p60,1920p,1080p60,1080p,best"


class LiveDlJob:
    def __init__(self, url: str, cfg: dict):
        self.url = url
        self.cfg = {
            "quality":      "1080",
            "start_t":      0,         # seconds since broadcast start (absolute)
            "end_t":        None,      # None = streamlink can't run, must error
            **(cfg or {}),
        }
        try:
            self.vid = core.yt_id_from_url(url)
        except ValueError:
            self.vid = "live"
        self.id = f"livedl_{self.vid}_{int(time.time() * 1000)}"

        self.created_at = int(time.time())
        self.started_at: Optional[int] = None
        self.finished_at: Optional[int] = None
        self.status = "queued"           # queued | downloading | transmuxing | done | failed | cancelled
        self.progress_pct = 0.0
        self.size_total = ""             # estimated MB
        self.speed = ""                  # MB/s avg
        self.eta = ""
        self.title = ""
        self.file: Optional[Path] = None
        self.file_size: Optional[int] = None
        self.error: Optional[str] = None
        self.phase_msg = ""

        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None

        s_t = int(self.cfg.get("start_t") or 0)
        e_t = int(self.cfg.get("end_t") or 0)
        self._stem = f"{self.vid}_live_{s_t}-{e_t}"
        self._span_s = max(1, e_t - s_t)
        self._dvr_offset = max(0, s_t)
        # Diagnostics surfaced via status_dict — lets the UI show
        # exactly what time-window streamlink was told to grab.
        self.live_now_probed: Optional[int] = None
        self.behind_from_live: Optional[int] = None

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
        if self.status in ("queued", "downloading", "transmuxing"):
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
            "live_now_probed":  self.live_now_probed,
            "behind_from_live": self.behind_from_live,
        }

    # ---------- internal ----------

    def _run(self):
        try:
            self.started_at = int(time.time())
            self.status = "downloading"

            ts_path = DOWNLOADS_ROOT / f"{self._stem}.ts"
            raw_path = DOWNLOADS_ROOT / f"{self._stem}.raw"
            mp4_path = DOWNLOADS_ROOT / f"{self._stem}.mp4"
            log_path = DOWNLOADS_ROOT / f"{self._stem}.streamlink.log"

            # Wipe any leftover from a previous failed run.
            for p in (ts_path, raw_path, mp4_path, log_path):
                try: p.unlink()
                except FileNotFoundError: pass
                except OSError: pass

            # APPROACH (verified working): YT live segment URLs follow
            # `…/sq/<N>/…/seg.ts` where N is the sequence number. The
            # `sig=` signature on the URL covers everything EXCEPT the
            # sq path component, so substituting past N values returns
            # past segments directly. YT DVR keeps ~5000 segments
            # (~2.7h at 2s/seg). We:
            #   1. yt-dlp -g → HLS playlist URL
            #   2. Curl the playlist, grab any segment URL + the current sq
            #   3. Compute target_sq = current_sq - (behind_from_live/2)
            #   4. Download segments [target_sq, target_sq + span/2]
            #      by sq-substitution (parallel for speed)
            #   5. concat .ts and transmux to .mp4 via ffmpeg
            quality_h = {"720":720, "1080":1080, "1440":1440, "2160":2160,
                         "best":9999}.get(str(self.cfg.get("quality") or "1080"), 1080)

            self.phase_msg = "probing live edge…"
            probe = probe_live_duration(self.url)
            if not probe.get("ok") or not probe.get("is_live") or not probe.get("now_s"):
                self.status = "failed"
                self.error = f"live probe failed: {probe.get('error') or 'not live'}"
                return
            live_now = int(probe["now_s"])
            behind_from_live = max(0, live_now - int(self.cfg.get("start_t") or 0))
            self.live_now_probed = live_now
            self.behind_from_live = behind_from_live

            self.phase_msg = "resolving HLS URL…"
            url_proc = subprocess.run([
                core.YTDLP, "--cookies-from-browser", "firefox",
                "--no-warnings", "-g",
                "-f", f"best[height<={quality_h}]/best",
                self.url,
            ], capture_output=True, timeout=90)
            if url_proc.returncode != 0:
                err = (url_proc.stderr or b"").decode(errors="ignore")[-300:]
                raise RuntimeError(f"yt-dlp -g failed: {err}")
            hls_url = (url_proc.stdout or b"").decode(errors="ignore").strip().splitlines()
            hls_url = hls_url[0] if hls_url else ""
            if not hls_url.startswith("http"):
                raise RuntimeError(f"yt-dlp -g returned no URL: {hls_url[:200]}")

            # Pull a sample segment URL + current sq from the live playlist.
            import urllib.request
            self.phase_msg = "resolving DVR segment template…"
            req = urllib.request.Request(hls_url, headers={"User-Agent": _FIREFOX_UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                playlist_text = r.read().decode("utf-8", errors="ignore")
            seg_lines = [ln.strip() for ln in playlist_text.splitlines()
                         if ln.strip().startswith("http")]
            if not seg_lines:
                raise RuntimeError("playlist has no segment URLs")
            sample_url = seg_lines[0]
            cur_sq_m = re.search(r"/sq/(\d+)/", sample_url)
            if not cur_sq_m:
                raise RuntimeError(f"no /sq/N/ in segment URL: {sample_url[:200]}")
            cur_sq = int(cur_sq_m.group(1))

            # YT-live HLS segments are 2 sec each.
            SEG_SECONDS = 2
            n_segs = max(1, (self._span_s + SEG_SECONDS - 1) // SEG_SECONDS)
            target_start_sq = max(0, cur_sq - (behind_from_live // SEG_SECONDS))
            target_sqs = list(range(target_start_sq, target_start_sq + n_segs))

            print(f"[livedl] {self.id}: q<={quality_h}p "
                  f"live_now={live_now}s behind={behind_from_live}s "
                  f"cur_sq={cur_sq} target_start_sq={target_start_sq} "
                  f"n_segs={n_segs}", flush=True)

            # Build URL template — replace just the sq.
            url_template = re.sub(r"/sq/\d+/", "/sq/{SQ}/", sample_url, count=1)

            # Download all target segments concurrently, write .ts chunks.
            seg_dir = DOWNLOADS_ROOT / f"{self._stem}.parts"
            seg_dir.mkdir(parents=True, exist_ok=True)
            for f in seg_dir.glob("*.ts"):
                try: f.unlink()
                except OSError: pass

            import concurrent.futures as _cf
            self.status = "downloading"
            started_dl = time.time()
            fetched_count = [0]
            fetch_errors = []
            est_total_s = float(self._span_s)

            def _fetch(idx: int, sq: int) -> Optional[Path]:
                u = url_template.replace("{SQ}", str(sq))
                out = seg_dir / f"{idx:06d}.ts"
                try:
                    req = urllib.request.Request(
                        u, headers={"User-Agent": _FIREFOX_UA})
                    with urllib.request.urlopen(req, timeout=60) as r:
                        data = r.read()
                    if not data:
                        fetch_errors.append((sq, "empty"))
                        return None
                    out.write_bytes(data)
                    fetched_count[0] += 1
                    return out
                except Exception as e:
                    fetch_errors.append((sq, str(e)[:80]))
                    return None

            with _cf.ThreadPoolExecutor(max_workers=8) as pool:
                futures = {pool.submit(_fetch, i, sq): (i, sq)
                           for i, sq in enumerate(target_sqs)}
                while futures:
                    done, _ = _cf.wait(futures, timeout=0.8,
                                       return_when=_cf.FIRST_COMPLETED)
                    for f in done: del futures[f]
                    if self.status == "cancelled": break
                    elapsed = max(0.1, time.time() - started_dl)
                    mb = sum(p.stat().st_size for p in seg_dir.glob("*.ts")) / (1024*1024)
                    mbps = mb / elapsed
                    pct = min(98.0, 100.0 * fetched_count[0] / n_segs)
                    self.progress_pct = pct
                    self.size_total = f"{mb:.1f} MB"
                    self.speed = f"{mbps:.2f} MB/s"
                    self.phase_msg = (
                        f"sq-fetch: {fetched_count[0]}/{n_segs} segs · "
                        f"{mb:.1f} MB · {mbps:.2f} MB/s")

            if self.status == "cancelled":
                return

            ok_segs = sorted(seg_dir.glob("*.ts"))
            if len(ok_segs) < n_segs * 0.8:
                raise RuntimeError(
                    f"only {len(ok_segs)}/{n_segs} segments fetched. "
                    f"first 3 errors: {fetch_errors[:3]}")

            # Concat .ts files (binary append) and transmux to mp4.
            self.status = "transmuxing"
            self.phase_msg = f"concat + transmux {len(ok_segs)} segs → mp4"
            concat_ts = seg_dir / "concat.ts"
            with open(concat_ts, "wb") as out_f:
                for p in ok_segs:
                    out_f.write(p.read_bytes())

            tm_cmd = [
                core.FFMPEG, "-y", "-hide_banner", "-loglevel", "warning",
                "-i", str(concat_ts),
                "-c", "copy",
                "-bsf:a", "aac_adtstoasc",
                "-movflags", "+faststart",
                str(mp4_path),
            ]
            tm = subprocess.run(tm_cmd, capture_output=True, timeout=300)
            if tm.returncode != 0 or not mp4_path.exists() or mp4_path.stat().st_size < 1024:
                err = (tm.stderr or b"").decode(errors="ignore").strip()[-500:]
                raise RuntimeError(f"ffmpeg transmux failed: {err}")

            # Cleanup parts dir.
            try:
                for p in seg_dir.glob("*.ts"): p.unlink()
                seg_dir.rmdir()
            except OSError: pass

            try: raw_path.unlink()
            except (OSError, FileNotFoundError): pass

            self.file = mp4_path
            self.file_size = mp4_path.stat().st_size
            self.title = f"live {_hms(self._dvr_offset)}–{_hms(self._dvr_offset + self._span_s)}"
            self.status = "done"
            self.progress_pct = 100.0
            self.phase_msg = f"done · {self.file_size // (1024*1024)} MB mp4"
            self.finished_at = int(time.time())

        except Exception as e:
            self.error = str(e)
            self.status = "failed"
            self.finished_at = int(time.time())
            print(f"[livedl] {self.id} failed: {e}", flush=True)


# ---------- module-level API used by app.py ----------

def start_download(url: str, cfg: dict) -> "LiveDlJob":
    return LiveDlJob(url, cfg).start()


def get_job(jid: str) -> Optional["LiveDlJob"]:
    with _LOCK:
        return JOBS.get(jid)


def stop_job(jid: str) -> bool:
    with _LOCK:
        j = JOBS.get(jid)
    if not j: return False
    j.stop()
    return True


def list_jobs() -> list:
    """Return active + recent live-section jobs. Persists by also scanning
    the downloads folder so files survive server restarts."""
    out = []
    seen_files = set()
    with _LOCK:
        for j in list(JOBS.values()):
            d = j.status_dict()
            out.append(d)
            if d.get("file_name"):
                seen_files.add(d["file_name"])
    # Hydrate from disk for any mp4 not in the in-memory job map (server
    # restart, previous-session downloads).
    for p in sorted(DOWNLOADS_ROOT.glob("*.mp4"),
                    key=lambda x: -x.stat().st_mtime):
        if p.name in seen_files: continue
        st = p.stat()
        out.append({
            "id": f"livedl_disk_{p.name}",
            "vid": p.stem.split("_live_")[0] if "_live_" in p.stem else p.stem,
            "url": "",
            "title": p.stem.replace("_live_", " "),
            "status": "done",
            "progress_pct": 100.0,
            "size_total": f"{st.st_size // (1024*1024)} MB",
            "speed": "", "eta": "",
            "phase_msg": "loaded from disk",
            "created_at": int(st.st_mtime),
            "started_at": int(st.st_mtime),
            "finished_at": int(st.st_mtime),
            "file_url": f"/downloads/{urllib.parse.quote(p.name)}",
            "file_name": p.name,
            "file_size_b": st.st_size,
            "error": None,
            "cfg": {},
        })
    return out


def delete_job(jid: str) -> bool:
    """Remove the mp4 + any sidecars, drop from JOBS."""
    target_file: Optional[Path] = None
    with _LOCK:
        j = JOBS.pop(jid, None)
    if j and j.file:
        target_file = j.file
    elif jid.startswith("livedl_disk_"):
        target_file = DOWNLOADS_ROOT / jid[len("livedl_disk_"):]
    if target_file and target_file.exists():
        try: target_file.unlink()
        except OSError: pass
        # Also nuke the .streamlink.log sidecar if present.
        for sidecar in (".streamlink.log", ".ts", ".raw"):
            sp = target_file.with_suffix(sidecar)
            if sp.exists():
                try: sp.unlink()
                except OSError: pass
        return True
    return False
