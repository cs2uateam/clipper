"""User-uploaded video → Whisper transcription → burn subs into picture.

Mirrors the download tab's burn pipeline but starts from a file the user
already has on disk (no yt-dlp). All subtitle logic comes from subs_burn,
so a fix there propagates here AND to the download path.

State on disk: `youtube-analyzer/v1/uploads/<run_id>/<filename>` plus the
generated `.whisper.en.vtt` sidecar. The original mp4 is replaced
in-place by the burned version when the user enabled burn_subs."""
import shutil, subprocess, threading, time, urllib.parse
from pathlib import Path
from typing import Dict, Optional

import core
import subs_burn

V1 = Path(__file__).resolve().parent
UPLOADS_ROOT = V1 / "uploads"
UPLOADS_ROOT.mkdir(exist_ok=True)

# Recognised media extensions — input that comes through the upload form.
# yt-dlp / ffmpeg can decode any of them via ffmpeg internally.
_MEDIA_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".flv",
               ".ts", ".mpg", ".mpeg", ".wmv"}

_LOCK = threading.Lock()
JOBS: Dict[str, "UploadJob"] = {}

class UploadJob:
    """Process an already-on-disk video: Whisper-transcribe → optionally
    burn subtitles. Same status taxonomy as DownloadJob so the UI can
    use the same render code."""

    def __init__(self, file_path: Path, cfg: dict):
        self.file: Path = file_path
        self.cfg = {
            "subtitles":          True,    # always — that's the entire point
            "burn_subs":          True,    # default ON
            "subs_size":          "medium",
            "subs_color":         "white",
            "subs_bg":            "none",
            "subs_position":      "bottom",
            # Optional watermark — composited in the same burn pass when
            # burn_subs is on. Resolved via app.py before the cfg lands here.
            "watermark_name":     "",
            "watermark_path":     "",
            "watermark_size":     20,
            "watermark_opacity":  70,
            "watermark_position": "center",
            **(cfg or {}),
        }
        # run_id is the UPLOADS_ROOT/<run_id>/ folder name; reused as
        # job id since uploads can be reattached after server restart.
        self.run_id = file_path.parent.name
        self.id = f"up_{self.run_id}"

        self.created_at = int(time.time())
        self.started_at: Optional[int] = None
        self.finished_at: Optional[int] = None
        self.status = "queued"   # queued | transcribing | burning | done | failed | cancelled
        self.phase_msg = ""
        self.title = file_path.stem
        self.file_size: Optional[int] = None
        try:
            self.file_size = file_path.stat().st_size
        except Exception: pass
        self.error: Optional[str] = None

        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None

    # ---------- public ----------
    def start(self):
        with _LOCK:
            JOBS[self.id] = self
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
                try: self._proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    try: self._proc.kill()
                    except Exception: pass
            except Exception: pass
        if self.status in ("queued", "transcribing", "burning"):
            self.status = "cancelled"

    def status_dict(self) -> dict:
        file_url = None
        if self.file and self.file.exists():
            file_url = (f"/uploads/{self.run_id}/"
                        f"{urllib.parse.quote(self.file.name)}")
        return {
            "id":           self.id,
            "run_id":       self.run_id,
            "title":        self.title,
            "status":       self.status,
            "phase_msg":    self.phase_msg,
            "created_at":   self.created_at,
            "started_at":   self.started_at,
            "finished_at":  self.finished_at,
            "file_url":     file_url,
            "file_name":    self.file.name if self.file else None,
            "file_size_b":  self.file_size,
            "error":        self.error,
            "cfg":          self.cfg,
        }

    # ---------- internal ----------
    def _run(self):
        try:
            self.started_at = int(time.time())
            # Whisper is on-demand: only runs when the user actually
            # wants subs burned in (`burn_subs=True`). Upload is often
            # a "put it here for later" step — running large-v3 for
            # minutes on every drop is wasteful. If the user later goes
            # into Cut / IDEA / Prokop, each of those paths runs Whisper
            # with the correct language.
            if self.cfg.get("burn_subs"):
                self.status = "transcribing"
                self.phase_msg = "generating subtitles via Whisper"
                lang = (self.cfg.get("subs_lang") or "en").lower()
                subs_burn.generate_subs_via_whisper(
                    self.file,
                    on_phase=lambda m: setattr(self, "phase_msg", m),
                    language=lang)
                self.status = "burning"
                self.phase_msg = "вшиваю субтитры в видео — re-encoding"
                subs_burn.burn_subtitles(
                    self.file, self.cfg,
                    on_phase=lambda m: setattr(self, "phase_msg", m),
                    on_proc=lambda p: setattr(self, "_proc", p))
                try: self.file_size = self.file.stat().st_size
                except Exception: pass
            elif self.cfg.get("watermark_name"):
                # No subs burn, but the user still wants the overlay.
                self.status = "burning"
                self.phase_msg = "наношу watermark"
                subs_burn.apply_watermark(
                    self.file, self.cfg,
                    on_phase=lambda m: setattr(self, "phase_msg", m),
                    on_proc=lambda p: setattr(self, "_proc", p))
                try: self.file_size = self.file.stat().st_size
                except Exception: pass

            self.status = "done"
            self.phase_msg = ""
        except Exception as e:
            if self.status != "cancelled":
                self.status = "failed"
                self.error = f"{type(e).__name__}: {e}"
        finally:
            self.finished_at = int(time.time())
            self._proc = None


# ---------- API helpers ----------
def start_upload(file_path: Path, cfg: dict) -> UploadJob:
    if not file_path.exists():
        raise ValueError(f"upload file missing: {file_path}")
    if file_path.suffix.lower() not in _MEDIA_EXTS:
        raise ValueError(f"unsupported video extension: {file_path.suffix}")
    job = UploadJob(file_path, cfg or {})
    job.start()
    return job

def get_job(job_id: str) -> Optional["UploadJob"]:
    with _LOCK:
        return JOBS.get(job_id)

def stop_job(job_id: str) -> Optional["UploadJob"]:
    job = get_job(job_id)
    if job: job.stop()
    return job

def list_jobs() -> list:
    """In-memory jobs + disk archive of past uploads. Sidecar .vtt files
    are hidden so the user only sees one entry per upload."""
    with _LOCK:
        live_all = [j.status_dict() for j in JOBS.values()]
    # Призраки: live-джоб ссылается на отсутствующий файл (переименован
    # в Explorer'е) → скрываем чтоб не было двойника с disk-сканом ниже.
    live = []
    for j in live_all:
        fn = j.get("file_name"); rid = j.get("run_id")
        if (fn and rid and j.get("status") == "done"
                and not (UPLOADS_ROOT / rid / fn).exists()):
            continue
        live.append(j)
    seen = {(j["run_id"], j.get("file_name")) for j in live if j.get("file_name")}
    out = list(live)
    if UPLOADS_ROOT.exists():
        for run_dir in UPLOADS_ROOT.iterdir():
            if not run_dir.is_dir(): continue
            for f in run_dir.iterdir():
                if not f.is_file(): continue
                if f.suffix.lower() in (".vtt", ".srt", ".ttml"): continue
                if f.suffix.lower() not in _MEDIA_EXTS: continue
                if (run_dir.name, f.name) in seen: continue
                try: stat = f.stat()
                except Exception: continue
                out.append({
                    "id":          f"up_disk:{run_dir.name}/{f.name}",
                    "run_id":      run_dir.name,
                    "title":       f.stem,
                    "status":      "done",
                    "phase_msg":   "",
                    "file_url":    (f"/uploads/{run_dir.name}/"
                                    f"{urllib.parse.quote(f.name)}"),
                    "file_name":   f.name,
                    "file_size_b": stat.st_size,
                    "finished_at": int(stat.st_mtime),
                    "has_idea":    f.with_suffix(".idea.json").exists(),
                })
    out.sort(key=lambda j: -(j.get("finished_at") or j.get("created_at") or 0))
    return out

def delete_job(job_id: str) -> bool:
    """Stop in-memory job + remove its run-folder from disk (mp4 +
    sidecars + the run folder itself)."""
    job = get_job(job_id)
    target_dir: Optional[Path] = None
    if job:
        job.stop()
        if job.file:
            target_dir = job.file.parent
        with _LOCK:
            JOBS.pop(job_id, None)
    elif job_id.startswith("up_disk:"):
        rest = job_id[len("up_disk:"):]
        if "/" in rest:
            run_id = rest.split("/", 1)[0]
            target_dir = UPLOADS_ROOT / run_id
    elif job_id.startswith("up_"):
        # Job from a previous in-memory session; fall back to run-id match
        run_id = job_id[len("up_"):]
        candidate = UPLOADS_ROOT / run_id
        if candidate.exists(): target_dir = candidate
    if target_dir is None or not target_dir.exists():
        return False
    try:
        shutil.rmtree(target_dir)
        return True
    except Exception:
        return False
