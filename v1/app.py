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
    api_key: str


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

@app.get("/downloads/{vid}/{filename:path}")
def serve_download(vid: str, filename: str):
    p = dl_mod.DOWNLOADS_ROOT / vid / filename
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
        job = idea_mod.start_idea(path, req.api_key)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"id": job.id, "status": job.status}

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
