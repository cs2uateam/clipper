"""FastAPI backend for clipper v1 — public-video grabber + Idea generator.

Routes only what Project B owns: YouTube/Instagram/TikTok download,
user upload, and the Gemini-powered Idea modal. The clip-finding
analyzer (heatmap/captions/audio + Live mode) lives in the sibling
`youtube-analyzer/` project on port 8766. This one runs on 8767."""
import time
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
import subs_burn

V1 = Path(__file__).resolve().parent
STATIC = V1 / "static"

app = FastAPI(title="clipper v1")


# ---------- Pydantic models ----------
class DownloadStartReq(BaseModel):
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

class InstaStartReq(BaseModel):
    url: str
    subtitles: bool = False
    burn_subs: bool = False
    subs_size: str = "medium"
    subs_color: str = "white"
    subs_bg: str = "none"
    subs_position: str = "bottom"

class TikTokStartReq(BaseModel):
    url: str
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


# ---------- YouTube download ----------
@app.post("/api/download/start")
def api_download_start(req: DownloadStartReq):
    cfg = req.model_dump()
    cfg.pop("url", None)
    try:
        job = dl_mod.start_download(req.url, cfg)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"failed to start: {e}")
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
        "subtitles":     True,
        "burn_subs":     burn_subs,
        "subs_size":     subs_size,
        "subs_color":    subs_color,
        "subs_bg":       subs_bg,
        "subs_position": subs_position,
    }
    try:
        job = up_mod.start_upload(target, cfg)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"failed to start: {e}")
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
    try:
        job = ig_mod.start_insta(req.url, cfg)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"failed to start: {e}")
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
    try:
        job = tt_mod.start_tiktok(req.url, cfg)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"failed to start: {e}")
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
    roots = {
        "downloads": dl_mod.DOWNLOADS_ROOT,
        "insta":     ig_mod.INSTA_ROOT,
        "tiktok":    tt_mod.TIKTOK_ROOT,
        "uploads":   up_mod.UPLOADS_ROOT,
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
            # explorer /select,<path> opens the parent folder with the
            # target file pre-selected. The trailing comma is part of
            # the syntax — no space allowed between /select and ,.
            subprocess.Popen(["explorer", f"/select,{media}"])
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
    roots = {
        "downloads": dl_mod.DOWNLOADS_ROOT,
        "insta":     ig_mod.INSTA_ROOT,
        "tiktok":    tt_mod.TIKTOK_ROOT,
        "uploads":   up_mod.UPLOADS_ROOT,
    }
    root = roots.get(prefix)
    if root is None:
        raise HTTPException(404, "unknown root")
    media = (root / rest).resolve()
    try: media.relative_to(root.resolve())
    except ValueError: raise HTTPException(400, "path escape")
    if not media.exists():
        raise HTTPException(404, "media missing")
    cache = media.with_suffix(".thumb.jpg")
    if not cache.exists() or cache.stat().st_mtime < media.stat().st_mtime:
        # Pull a frame at 1s, scale to 240px wide, JPEG-encode small.
        # Failing here is fine — caller has a fallback emoji.
        cmd = [core.FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
               "-ss", "1", "-i", str(media),
               "-frames:v", "1", "-vf", "scale=240:-2",
               "-q:v", "5", str(cache)]
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
    if not candidate.exists():
        raise HTTPException(404, f"file not found: {source}/{name}")
    return candidate

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


# ---------- Static / index ----------
@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")

app.mount("/static", StaticFiles(directory=STATIC), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8767)
