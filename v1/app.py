"""FastAPI backend for clipper v1 — public-video grabber + Idea generator.

Routes only what Project B owns: YouTube/Instagram/TikTok download,
user upload, and the Gemini-powered Idea modal. The clip-finding
analyzer (heatmap/captions/audio + Live mode) lives in the sibling
`youtube-analyzer/` project on port 8766. This one runs on 8767."""
import sys, time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import download as dl_mod
import upload as up_mod
import insta as ig_mod
import tiktok as tt_mod
import idea as idea_mod
import cutter as cut_mod
import subs_burn

V1 = Path(__file__).resolve().parent
STATIC = V1 / "static"

app = FastAPI(title="clipper v1")


# ---------- Pydantic models ----------
# Watermark fields are shared by all four ingest tabs — declared once
# here, mixed into each request model below. `watermark_name` is the
# filename under WATERMARKS_DIR (resolved to an absolute path before
# the cfg reaches subs_burn). Empty name = no watermark.
class _WatermarkOpts(BaseModel):
    watermark_name:     str = ""           # filename or "" to disable
    watermark_size:     int = 20           # 10..100, % of output width
    watermark_opacity:  int = 70           # 10..100, %
    watermark_position: str = "center"     # center/tl/tr/bl/br/fill/none

class DownloadStartReq(_WatermarkOpts):
    url: str
    quality: str = "1080"
    audio_only: bool = False
    audio_format: str = "mp3"
    subtitles: bool = False
    burn_subs: bool = False
    subs_size: str = "medium"
    subs_color: str = "white"
    subs_bg: str = "none"
    subs_position: str = "bottom"
    start_t: Optional[int] = None
    end_t: Optional[int] = None
    aspect: str = "source"   # source / 9:16 / 16:9 / 1:1 / 4:5 / W:H

class InstaStartReq(_WatermarkOpts):
    url: str
    audio_only: bool = False
    audio_format: str = "mp3"
    subtitles: bool = False
    burn_subs: bool = False
    subs_size: str = "medium"
    subs_color: str = "white"
    subs_bg: str = "none"
    subs_position: str = "bottom"

class TikTokStartReq(_WatermarkOpts):
    url: str
    audio_only: bool = False
    audio_format: str = "mp3"
    subtitles: bool = False
    burn_subs: bool = False
    subs_size: str = "medium"
    subs_color: str = "white"
    subs_bg: str = "none"
    subs_position: str = "bottom"

class IdeaStartReq(BaseModel):
    source: str   # "youtube" | "upload" | "insta" | "tiktok"
    name: str
    api_key: str = ""    # may be empty when a cache hit is expected
    force: bool = False  # True = bypass cache and call Gemini fresh
    hint: str = ""       # optional refinement note → forces fresh call


class CutSegment(BaseModel):
    start_s: float
    end_s: float
    label: str = ""

class CutStartReq(_WatermarkOpts):
    source: str       # "youtube" | "upload" | "insta" | "tiktok" | "cuts"
    name: str         # filename under that root (matches Idea modal's name)
    segments: list[CutSegment]
    precise: bool = False
    label: str = ""

class CutMergeReq(BaseModel):
    filenames: list[str]      # base filenames under cuts root, in concat order
    output_name: str = ""     # optional; auto-named when blank

class CutAudioReq(BaseModel):
    filename: str             # base filename under cuts root
    fmt: str = "mp3"          # mp3 | m4a | wav


# ---------- Watermark helpers (shared by all 4 ingest tabs) ----------
def _inject_watermark_path(cfg: dict) -> dict:
    """Resolve cfg['watermark_name'] (filename under WATERMARKS_DIR) to
    an absolute filesystem path stored in cfg['watermark_path']. Validates
    that the file exists + lives under the watermarks dir (no path
    traversal). Mutates and returns the same cfg dict for convenience.
    Empty/missing name = no-op (cfg flows through unchanged)."""
    name = (cfg.get("watermark_name") or "").strip()
    if not name:
        cfg["watermark_path"] = ""
        return cfg
    root = subs_burn.WATERMARKS_DIR.resolve()
    p = (root / name).resolve()
    try: p.relative_to(root)
    except ValueError:
        raise HTTPException(400, "watermark path escape")
    if not p.exists():
        raise HTTPException(400, f"watermark not found: {name}")
    cfg["watermark_path"] = str(p)
    return cfg


# ---------- Activity events → Digest (:8772) ----------
def _digest_event(source: str, kind: str, name: str, meta=None) -> None:
    """Fire-and-forget POST to digest /api/event. Best-effort: never raises,
    never blocks the request handler. Stores activity so the daily report
    still sees uploads/downloads/cuts even after files were deleted."""
    import json as _json
    import threading
    import urllib.request
    def _post():
        try:
            req = urllib.request.Request(
                "http://127.0.0.1:8772/api/event",
                data=_json.dumps({"source": source, "kind": kind,
                                  "name": name, "meta": meta or {}}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=1.0)
        except Exception:
            pass
    threading.Thread(target=_post, daemon=True).start()


# ---------- YouTube download ----------
@app.post("/api/download/start")
def api_download_start(req: DownloadStartReq):
    cfg = req.model_dump()
    cfg.pop("url", None)
    _inject_watermark_path(cfg)
    try:
        job = dl_mod.start_download(req.url, cfg)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"failed to start: {e}")
    _digest_event("clipper", "download_youtube", req.url,
                  {"id": job.id, "quality": cfg.get("quality"),
                   "aspect": cfg.get("aspect")})
    return {"id": job.id, "status": job.status}

@app.get("/api/download/status/{job_id}")
def api_download_status(job_id: str):
    job = dl_mod.get_job(job_id)
    if not job:
        raise HTTPException(404, "unknown download job")
    return job.status_dict()

@app.post("/api/download/stop/{job_id}")
def api_download_stop(job_id: str):
    job = dl_mod.stop_job(job_id)
    if not job:
        raise HTTPException(404, "unknown download job")
    return {"id": job.id, "status": job.status}

@app.get("/api/download/list")
def api_download_list():
    return {"jobs": dl_mod.list_jobs()}

@app.delete("/api/download/{job_id:path}")
def api_download_delete(job_id: str):
    ok = dl_mod.delete_job(job_id)
    if not ok:
        raise HTTPException(404, "unknown or already deleted")
    return {"deleted": job_id}

@app.get("/downloads/{filename:path}")
def serve_download(filename: str):
    # `/downloads/cuts/*` belongs to the cutter (separate root); dispatch
    # there first so a cut segment doesn't get mis-routed to the YouTube
    # downloads folder and 404.
    if filename.startswith("cuts/"):
        p = cut_mod.CUTS_ROOT / filename[len("cuts/"):]
        if not p.exists(): raise HTTPException(404)
        return FileResponse(p, filename=Path(filename).name)
    p = dl_mod.DOWNLOADS_ROOT / filename
    if not p.exists(): raise HTTPException(404)
    return FileResponse(p, filename=filename)


# ---------- Upload ----------
@app.post("/api/upload/start")
async def api_upload_start(
    file: UploadFile = File(...),
    subs_size: str = Form("medium"),
    subs_color: str = Form("white"),
    subs_bg: str = Form("none"),
    subs_position: str = Form("bottom"),
    burn_subs: bool = Form(True),
    watermark_name: str = Form(""),
    watermark_size: int = Form(20),
    watermark_opacity: int = Form(70),
    watermark_position: str = Form("center"),
):
    run_id = f"u_{int(time.time() * 1000)}"
    folder = up_mod.UPLOADS_ROOT / run_id
    folder.mkdir(parents=True, exist_ok=True)
    safe_name = subs_burn.safe_filename(file.filename or "uploaded.mp4")
    target = folder / safe_name
    try:
        with open(target, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk: break
                out.write(chunk)
    finally:
        await file.close()
    cfg = {
        "subtitles":          True,
        "burn_subs":          burn_subs,
        "subs_size":          subs_size,
        "subs_color":         subs_color,
        "subs_bg":            subs_bg,
        "subs_position":      subs_position,
        "watermark_name":     watermark_name,
        "watermark_size":     watermark_size,
        "watermark_opacity":  watermark_opacity,
        "watermark_position": watermark_position,
    }
    _inject_watermark_path(cfg)
    try:
        job = up_mod.start_upload(target, cfg)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"failed to start: {e}")
    _digest_event("clipper", "upload", safe_name,
                  {"id": job.id, "run_id": job.run_id,
                   "size_b": target.stat().st_size if target.exists() else 0})
    return {"id": job.id, "run_id": job.run_id, "status": job.status}

@app.get("/api/upload/status/{job_id:path}")
def api_upload_status(job_id: str):
    job = up_mod.get_job(job_id)
    if not job:
        raise HTTPException(404, "unknown upload job")
    return job.status_dict()

@app.post("/api/upload/stop/{job_id:path}")
def api_upload_stop(job_id: str):
    job = up_mod.stop_job(job_id)
    if not job:
        raise HTTPException(404, "unknown upload job")
    return {"id": job.id, "status": job.status}

@app.get("/api/upload/list")
def api_upload_list():
    return {"jobs": up_mod.list_jobs()}

@app.delete("/api/upload/{job_id:path}")
def api_upload_delete(job_id: str):
    ok = up_mod.delete_job(job_id)
    if not ok:
        raise HTTPException(404, "unknown or already deleted")
    return {"deleted": job_id}

@app.get("/uploads/{run_id}/{filename:path}")
def serve_upload(run_id: str, filename: str):
    p = up_mod.UPLOADS_ROOT / run_id / filename
    if not p.exists(): raise HTTPException(404)
    return FileResponse(p, filename=filename)


# ---------- Instagram ----------
@app.post("/api/insta/start")
def api_insta_start(req: InstaStartReq):
    cfg = req.model_dump(); cfg.pop("url", None)
    _inject_watermark_path(cfg)
    try:
        job = ig_mod.start_insta(req.url, cfg)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"failed to start: {e}")
    _digest_event("clipper", "download_insta", req.url,
                  {"id": job.id, "reel_id": getattr(job, "reel_id", "")})
    return {"id": job.id, "reel_id": job.reel_id, "status": job.status}

@app.get("/api/insta/status/{job_id:path}")
def api_insta_status(job_id: str):
    job = ig_mod.get_job(job_id)
    if not job: raise HTTPException(404, "unknown insta job")
    return job.status_dict()

@app.post("/api/insta/stop/{job_id:path}")
def api_insta_stop(job_id: str):
    job = ig_mod.stop_job(job_id)
    if not job: raise HTTPException(404, "unknown insta job")
    return {"id": job.id, "status": job.status}

@app.get("/api/insta/list")
def api_insta_list():
    return {"jobs": ig_mod.list_jobs()}

@app.delete("/api/insta/{job_id:path}")
def api_insta_delete(job_id: str):
    ok = ig_mod.delete_job(job_id)
    if not ok: raise HTTPException(404, "unknown or already deleted")
    return {"deleted": job_id}

@app.get("/insta/{filename:path}")
def serve_insta(filename: str):
    p = ig_mod.INSTA_ROOT / filename
    if not p.exists(): raise HTTPException(404)
    return FileResponse(p, filename=filename)


# ---------- TikTok ----------
@app.post("/api/tiktok/start")
def api_tiktok_start(req: TikTokStartReq):
    cfg = req.model_dump(); cfg.pop("url", None)
    _inject_watermark_path(cfg)
    try:
        job = tt_mod.start_tiktok(req.url, cfg)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"failed to start: {e}")
    _digest_event("clipper", "download_tiktok", req.url,
                  {"id": job.id, "video_id": getattr(job, "video_id", "")})
    return {"id": job.id, "video_id": job.video_id, "status": job.status}

@app.get("/api/tiktok/status/{job_id:path}")
def api_tiktok_status(job_id: str):
    job = tt_mod.get_job(job_id)
    if not job: raise HTTPException(404, "unknown tiktok job")
    return job.status_dict()

@app.post("/api/tiktok/stop/{job_id:path}")
def api_tiktok_stop(job_id: str):
    job = tt_mod.stop_job(job_id)
    if not job: raise HTTPException(404, "unknown tiktok job")
    return {"id": job.id, "status": job.status}

@app.get("/api/tiktok/list")
def api_tiktok_list():
    return {"jobs": tt_mod.list_jobs()}


# ---------- All sources aggregated ----------
@app.get("/api/all/list")
def api_all_list():
    """Read-only хронологический фид из всех 4 источников. Каждый item
    получает поле `source` (yt|insta|tiktok|upload) — UI по нему
    маршрутизирует actions (delete/idea/serve) на per-source endpoint'ы.
    Никакого нового хранилища: тянем существующие list_jobs()."""
    merged = []
    # Источники тегируем теми же значениями что используют per-tab
    # вкладки и /api/idea/* / cut endpoint'ы. Особенно важно "youtube"
    # (а не "yt") — backend _resolve_idea_path принимает только эти.
    for src, mod in (("youtube", dl_mod), ("insta", ig_mod),
                     ("tiktok", tt_mod), ("upload", up_mod)):
        try:
            jobs = mod.list_jobs() or []
        except Exception as e:
            print(f"all/list: {src} skipped: {e}")
            jobs = []
        for j in jobs:
            j2 = dict(j)
            j2["source"] = src
            merged.append(j2)
    merged.sort(key=lambda j: -(j.get("finished_at")
                                or j.get("created_at") or 0))
    return {"jobs": merged}

@app.delete("/api/tiktok/{job_id:path}")
def api_tiktok_delete(job_id: str):
    ok = tt_mod.delete_job(job_id)
    if not ok: raise HTTPException(404, "unknown or already deleted")
    return {"deleted": job_id}

@app.get("/tiktok/{filename:path}")
def serve_tiktok(filename: str):
    p = tt_mod.TIKTOK_ROOT / filename
    if not p.exists(): raise HTTPException(404)
    return FileResponse(p, filename=filename)


# ---------- Cutter (multi-segment trim of any source mp4) ----------
@app.post("/api/cut/start")
def api_cut_start(req: CutStartReq):
    try:
        source_path = cut_mod.resolve_source(req.source, req.name)
    except ValueError as e:
        raise HTTPException(400, str(e))
    # Resolve watermark name → absolute path (or empty if disabled).
    # The cutter's CutJob inspects this cfg to decide whether to take
    # the single-pass re-encode + overlay branch.
    wm_cfg = {
        "watermark_name":     req.watermark_name,
        "watermark_size":     req.watermark_size,
        "watermark_opacity":  req.watermark_opacity,
        "watermark_position": req.watermark_position,
    }
    _inject_watermark_path(wm_cfg)
    try:
        job = cut_mod.start_cut(
            source_path,
            [s.model_dump() for s in req.segments],
            precise=req.precise,
            label=req.label,
            wm_cfg=wm_cfg,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"failed to start: {e}")
    _digest_event("clipper", "cut", req.name or req.source,
                  {"id": job.id, "segments": len(req.segments),
                   "precise": req.precise, "label": req.label})
    return {"id": job.id, "status": job.status}

@app.get("/api/cut/status/{job_id:path}")
def api_cut_status(job_id: str):
    job = cut_mod.get_job(job_id)
    if not job: raise HTTPException(404, "unknown cut job")
    return job.status_dict()

@app.post("/api/cut/stop/{job_id:path}")
def api_cut_stop(job_id: str):
    job = cut_mod.stop_job(job_id)
    if not job: raise HTTPException(404, "unknown cut job")
    return {"id": job.id, "status": job.status}

@app.get("/api/cut/list")
def api_cut_list():
    return {"jobs": cut_mod.list_jobs()}

# NOTE: order matters — the more-specific `/segment/...` route MUST be
# declared BEFORE the catch-all `/{job_id:path}`, otherwise FastAPI
# matches `/api/cut/segment/foo.mp4` against the catch-all and routes
# it to delete_job(), which doesn't understand segment filenames and
# returns 404. (Took a while to track down this bug.)
@app.delete("/api/cut/segment/{filename:path}")
def api_cut_delete_segment(filename: str):
    """Delete one segment file (per-clip 🗑 button)."""
    ok = cut_mod.delete_segment(filename)
    if not ok: raise HTTPException(404, "segment not found")
    return {"deleted": filename}

@app.delete("/api/cut/{job_id:path}")
def api_cut_delete(job_id: str):
    ok = cut_mod.delete_job(job_id)
    if not ok: raise HTTPException(404, "unknown or already deleted")
    return {"deleted": job_id}

@app.post("/api/cut/merge")
def api_cut_merge(req: CutMergeReq):
    """Concat 2+ existing cut segments into a single mp4. All inputs
    must live under the cuts/ root. Output filename is sanitized."""
    try:
        out = cut_mod.merge_segments(req.filenames, req.output_name)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"merge failed: {e}")
    import urllib.parse
    return {
        "file": out.name,
        "url":  f"/downloads/cuts/{urllib.parse.quote(out.name)}",
        "size_b": out.stat().st_size if out.exists() else 0,
    }

@app.post("/api/cut/extract-audio")
def api_cut_extract_audio(req: CutAudioReq):
    """Save audio track of a cut segment as standalone mp3/m4a/wav file
    in the same cuts/ folder."""
    try:
        out = cut_mod.extract_audio(req.filename, req.fmt)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"extract failed: {e}")
    import urllib.parse
    return {
        "file": out.name,
        "url":  f"/downloads/cuts/{urllib.parse.quote(out.name)}",
        "size_b": out.stat().st_size if out.exists() else 0,
    }


# ---------- Reveal in File Explorer ----------
@app.post("/api/reveal")
def api_reveal(url: str):
    """Open the OS file manager focused on the given media file. The
    `url` is the server-relative serve URL we already hand out (e.g.
    `/insta/foo.mp4`) — we resolve it back to disk path under one of
    the known roots, then call the platform-native reveal-in-folder.
    Pure-cosmetic feature: failures are non-fatal, just return ok=False."""
    import os, subprocess
    from urllib.parse import unquote
    url = unquote(url or "")
    if not url.startswith("/"):
        raise HTTPException(400, "url must be absolute path")
    parts = url.lstrip("/").split("/", 1)
    if len(parts) < 2:
        raise HTTPException(400, "bad url")
    prefix, rest = parts
    # `cuts` lives under /downloads/cuts/ on the URL side, but on disk
    # it's its own root (cut_mod.CUTS_ROOT). Strip the `cuts/` prefix
    # from `rest` and dispatch to the cutter root in that case.
    if prefix == "downloads" and rest.startswith("cuts/"):
        prefix, rest = "cuts", rest[len("cuts/"):]
    roots = {
        "downloads": dl_mod.DOWNLOADS_ROOT,
        "insta":     ig_mod.INSTA_ROOT,
        "tiktok":    tt_mod.TIKTOK_ROOT,
        "uploads":   up_mod.UPLOADS_ROOT,
        "cuts":      cut_mod.CUTS_ROOT,
    }
    root = roots.get(prefix)
    if root is None:
        raise HTTPException(404, "unknown root")
    media = (root / rest).resolve()
    try: media.relative_to(root.resolve())
    except ValueError: raise HTTPException(400, "path escape")
    if not media.exists():
        raise HTTPException(404, "media missing")
    try:
        if os.name == "nt":
            # Open the parent folder via the Windows shell — bulletproof
            # across spaces, unicode, and quoting. We dropped
            # `explorer /select,<path>` because it silently mis-parsed
            # paths with spaces (the upload folder uses the original
            # filename, which often has spaces) and opened the user's
            # home folder instead.
            os.startfile(str(media.parent))  # type: ignore[attr-defined]
        elif os.uname().sysname == "Darwin":
            subprocess.Popen(["open", "-R", str(media)])
        else:
            subprocess.Popen(["xdg-open", str(media.parent)])
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "path": str(media)}


# ---------- Thumbnails ----------
@app.get("/api/thumb")
def api_thumb(url: str):
    """Generate (or serve cached) thumbnail for a media file referenced
    by its server URL. The cache lives next to the media as
    `<stem>.thumb.jpg` and is hidden via attrib +H so it doesn't litter
    the user's File Explorer."""
    import core, subprocess
    from urllib.parse import unquote
    # Map serve-URL → on-disk path under one of the four roots.
    url = unquote(url or "")
    if not url.startswith("/"):
        raise HTTPException(400, "url must be absolute path")
    parts = url.lstrip("/").split("/", 1)
    if len(parts) < 2:
        raise HTTPException(400, "bad url")
    prefix, rest = parts
    # `cuts` lives under /downloads/cuts/ on the URL side, but on disk
    # it's its own root (cut_mod.CUTS_ROOT). Strip the `cuts/` prefix
    # from `rest` and dispatch to the cutter root in that case.
    if prefix == "downloads" and rest.startswith("cuts/"):
        prefix, rest = "cuts", rest[len("cuts/"):]
    roots = {
        "downloads": dl_mod.DOWNLOADS_ROOT,
        "insta":     ig_mod.INSTA_ROOT,
        "tiktok":    tt_mod.TIKTOK_ROOT,
        "uploads":   up_mod.UPLOADS_ROOT,
        "cuts":      cut_mod.CUTS_ROOT,
    }
    root = roots.get(prefix)
    if root is None:
        raise HTTPException(404, "unknown root")
    media = (root / rest).resolve()
    try: media.relative_to(root.resolve())
    except ValueError: raise HTTPException(400, "path escape")
    if not media.exists():
        raise HTTPException(404, "media missing")
    # New filename suffix (`.thumb640.jpg`) makes old 240px cached thumbs
    # stale automatically — they sit next to a media file with the OLD
    # name, this lookup misses them and we regenerate at higher quality.
    cache = media.with_suffix(".thumb640.jpg")
    if not cache.exists() or cache.stat().st_mtime < media.stat().st_mtime:
        # 640px wide, ffmpeg quality 2 (near-best JPEG, 2..31 lower = better).
        # Up from 240px/q5 — old thumbs looked blurry once card grid widened.
        cmd = [core.FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
               "-ss", "1", "-i", str(media),
               "-frames:v", "1", "-vf", "scale=640:-2",
               "-q:v", "2", str(cache)]
        try:
            subprocess.run(cmd, capture_output=True, timeout=30)
            core.hide_file(cache)
        except Exception:
            pass
    if not cache.exists(): raise HTTPException(404, "thumb gen failed")
    return FileResponse(cache, media_type="image/jpeg")


# ---------- Idea (Gemini) ----------
def _resolve_idea_path(source: str, name: str) -> Path:
    roots = {
        "insta":   ig_mod.INSTA_ROOT,
        "tiktok":  tt_mod.TIKTOK_ROOT,
        "upload":  up_mod.UPLOADS_ROOT,
        "youtube": dl_mod.DOWNLOADS_ROOT,
    }
    if source not in roots:
        raise HTTPException(400, f"unknown source: {source}")
    root = roots[source].resolve()
    candidate = (root / name).resolve()
    try: candidate.relative_to(root)
    except ValueError: raise HTTPException(400, "path escapes root")
    if candidate.exists():
        return candidate
    # Upload fallback: files live at uploads/<run_id>/<file_name>. If the
    # caller only knew the bare filename (e.g. an external agent without
    # run_id context), search one level deep for a match.
    if source == "upload" and "/" not in name and "\\" not in name:
        for run_dir in root.iterdir():
            if not run_dir.is_dir(): continue
            hit = run_dir / name
            if hit.exists():
                return hit.resolve()
    raise HTTPException(404, f"file not found: {source}/{name}")

@app.post("/api/idea/start")
def api_idea_start(req: IdeaStartReq):
    path = _resolve_idea_path(req.source, req.name)
    try:
        job = idea_mod.start_idea(path, req.api_key,
                                   force=req.force, hint=req.hint)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"id": job.id, "status": job.status}

@app.get("/api/idea/cached")
def api_idea_cached(source: str, name: str):
    """UI calls this before opening the modal to know whether a cache
    hit is available — lets us hide the API-key prompt on cached files."""
    path = _resolve_idea_path(source, name)
    return {"cached": idea_mod.has_cached(path)}

@app.get("/api/idea/status/{job_id}")
def api_idea_status(job_id: str):
    job = idea_mod.get_job(job_id)
    if not job: raise HTTPException(404, "unknown idea job")
    return job.status_dict()

@app.delete("/api/idea/{job_id}")
def api_idea_drop(job_id: str):
    ok = idea_mod.drop_job(job_id)
    if not ok: raise HTTPException(404, "unknown idea job")
    return {"dropped": job_id}


class IdeaPrepareReq(BaseModel):
    source: str
    name: str
    n_frames: int = 0   # 0 → auto: clamp(12, dur/2, 30)


@app.post("/api/idea/prepare")
def api_idea_prepare(req: IdeaPrepareReq):
    """Extract frames + transcript for a video without calling Gemini.
    Used by external agents (curator) that want to write the caption
    themselves via Claude/whatever. Returns absolute frame paths +
    transcript text; the caller composes the caption."""
    path = _resolve_idea_path(req.source, req.name)
    nf = req.n_frames if req.n_frames > 0 else 0  # 0 → auto in idea_mod
    if nf > 60: nf = 60                            # hard ceiling regardless
    try:
        out = idea_mod.prepare_inputs(path, n_frames=nf)
    except Exception as e:
        raise HTTPException(500, f"prepare failed: {type(e).__name__}: {e}")
    return out


# ---------- Caption (Claude SDK, curator-style 2-phase flow) ----------
import caption as caption_mod


class CaptionScoutReq(BaseModel):
    source: str
    name: str


class CaptionGenerateReq(BaseModel):
    source: str
    name: str
    angle: str = ""


@app.post("/api/caption/scout")
async def api_caption_scout(req: CaptionScoutReq):
    """Phase 1 — extract 12-30 keyframes + transcript, ask Claude for
    PART A (factual read) + PART B (angle question with 2-3 options).
    No captions yet — user picks the angle, then calls /generate."""
    path = _resolve_idea_path(req.source, req.name)
    try:
        out = await caption_mod.scout_clip(path)
    except Exception as e:
        raise HTTPException(500, f"scout failed: {type(e).__name__}: {e}")
    return out


@app.post("/api/caption/generate")
async def api_caption_generate(req: CaptionGenerateReq):
    """Phase 2 — with the chosen angle, compose all 4 cards (Figma title +
    3 platforms). Each generation is appended to per-clip history."""
    path = _resolve_idea_path(req.source, req.name)
    try:
        out = await caption_mod.generate_captions(path, angle=req.angle)
    except Exception as e:
        raise HTTPException(500, f"generate failed: {type(e).__name__}: {e}")
    return out


@app.get("/api/caption/history")
def api_caption_history(source: str, name: str):
    """Returns past generations for this clip — { items: [{ts, angle, text}] }.
    Most recent first."""
    path = _resolve_idea_path(source, name)
    items = caption_mod.load_history(path)
    items_sorted = sorted(items, key=lambda x: -float(x.get("ts", 0)))
    return {"items": items_sorted, "count": len(items_sorted)}


class CaptionHistoryDeleteReq(BaseModel):
    source: str
    name:   str
    ts:     float


@app.post("/api/caption/history/delete")
def api_caption_history_delete(req: CaptionHistoryDeleteReq):
    path = _resolve_idea_path(req.source, req.name)
    ok = caption_mod.delete_history_entry(path, req.ts)
    if not ok:
        raise HTTPException(404, "entry not found")
    return {"ok": True}


# ---------- Watermarks (uploadable PNG/JPG library) ----------
# Storage lives at `clipper/v1/watermarks/` (created by subs_burn). Each
# file is a freestanding image the user can pick per render via the
# shared watermark controls in every ingest tab. The same image flows
# through all 4 pipelines (download/upload/insta/tiktok).
_WM_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
_WM_NAME_UNSAFE = __import__("re").compile(r'[<>:"/\\|?*\x00-\x1f]')

def _safe_wm_name(name: str) -> str:
    """Strip path separators / OS-illegal chars from a watermark filename
    and ensure an image extension. Same shape as subs_burn.safe_filename
    but extension-aware (default .png)."""
    if not name: name = "watermark.png"
    name = Path(name).name  # strip dirs
    name = _WM_NAME_UNSAFE.sub("_", name)
    name = name.strip().lstrip(".") or "watermark.png"
    if Path(name).suffix.lower() not in _WM_EXTS:
        name += ".png"
    if len(name) > 180:
        p = Path(name); name = p.stem[:170] + p.suffix
    return name

@app.post("/api/watermarks")
async def api_wm_upload(file: UploadFile = File(...)):
    name = _safe_wm_name(file.filename or "watermark.png")
    target = subs_burn.WATERMARKS_DIR / name
    try:
        with open(target, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk: break
                out.write(chunk)
    finally:
        await file.close()
    try: size = target.stat().st_size
    except Exception: size = 0
    return {"name": target.name,
            "url": f"/watermarks/{target.name}",
            "size_b": size}

@app.get("/api/watermarks")
def api_wm_list():
    """All images currently in WATERMARKS_DIR. Sorted newest first so
    the most-recent upload shows up first in the UI picker."""
    items = []
    if subs_burn.WATERMARKS_DIR.exists():
        for f in subs_burn.WATERMARKS_DIR.iterdir():
            if not f.is_file(): continue
            if f.suffix.lower() not in _WM_EXTS: continue
            try: stat = f.stat()
            except Exception: continue
            items.append({"name": f.name,
                          "url": f"/watermarks/{f.name}",
                          "size_b": stat.st_size,
                          "mtime": int(stat.st_mtime)})
    items.sort(key=lambda x: -x["mtime"])
    return {"items": items}

@app.delete("/api/watermarks/{name}")
def api_wm_delete(name: str):
    safe = _safe_wm_name(name)
    p = (subs_burn.WATERMARKS_DIR / safe).resolve()
    try: p.relative_to(subs_burn.WATERMARKS_DIR.resolve())
    except ValueError: raise HTTPException(400, "path escape")
    if not p.exists():
        raise HTTPException(404, "not found")
    try: p.unlink()
    except Exception as e:
        raise HTTPException(500, f"delete failed: {e}")
    return {"deleted": safe}

@app.get("/watermarks/{name}")
def serve_watermark(name: str):
    safe = _safe_wm_name(name)
    p = subs_burn.WATERMARKS_DIR / safe
    if not p.exists(): raise HTTPException(404)
    return FileResponse(p, filename=safe)


# ---------- Static / index ----------
_NO_CACHE_HEADERS = {
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Pragma": "no-cache",
    "Expires": "0",
}

@app.get("/")
def index():
    # Localhost dev tool — browser must never cache the HTML. Without
    # this every iteration on index.html requires Ctrl+Shift+R.
    return FileResponse(STATIC / "index.html", headers=_NO_CACHE_HEADERS)

app.mount("/static", StaticFiles(directory=STATIC), name="static")


# ---------- Startup: sweep orphan sidecars ----------
# Sidecars (`<stem>.thumb.jpg`, `<stem>.idea.json`, `<stem>.en.vtt`,
# `<stem>.whisper.en.vtt`, `<stem>.ytdlp.log`, `<stem>.info.json`) live
# next to a media mp4. When the mp4 disappears (manual rm, server crash
# mid-delete, etc.) the sidecar can survive — over time it accumulates.
# This runs once at app startup and deletes any sidecar whose `<stem>`
# has no matching media file in the same folder.
_SIDECAR_SUFFIXES = (
    ".thumb.jpg", ".idea.json", ".en.vtt", ".whisper.en.vtt",
    ".ytdlp.log", ".info.json", ".srt", ".ttml",
)
_MEDIA_SUFFIXES = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v",
                   ".flv", ".ts", ".mp3", ".m4a", ".wav")

def _sweep_orphan_sidecars():
    import time as _time
    t0 = _time.time()
    # Scan flat single-level folders + upload's <run_id>/<file> structure.
    roots = [dl_mod.DOWNLOADS_ROOT, ig_mod.INSTA_ROOT, tt_mod.TIKTOK_ROOT,
             cut_mod.CUTS_ROOT]
    upload_roots = []
    if up_mod.UPLOADS_ROOT.exists():
        for sub in up_mod.UPLOADS_ROOT.iterdir():
            if sub.is_dir(): upload_roots.append(sub)
    n = 0
    for d in roots + upload_roots:
        if not d.exists(): continue
        names = {f.name for f in d.iterdir() if f.is_file()}
        # Stems of every real media file in this dir. An orphan sidecar
        # is one whose `<stem>` has no media-extension sibling.
        media_stems = {Path(nm).stem for nm in names
                       if Path(nm).suffix.lower() in _MEDIA_SUFFIXES}
        for nm in list(names):
            for suf in _SIDECAR_SUFFIXES:
                if nm.endswith(suf):
                    stem = nm[: -len(suf)]
                    if stem and stem not in media_stems:
                        try:
                            (d / nm).unlink(); n += 1
                            print(f"orphan-sweep: rm {d.name}/{nm}", file=sys.stderr)
                        except Exception:
                            pass
                    break
    print(f"orphan-sweep done: removed {n} files in {_time.time()-t0:.2f}s", file=sys.stderr)

@app.on_event("startup")
def _on_startup():
    try: _sweep_orphan_sidecars()
    except Exception as e:
        print(f"orphan-sweep failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8767)
