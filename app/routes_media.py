"""Media routes: upload staging, ROI configuration, annotated-video download."""

from __future__ import annotations

import tempfile
import time
import uuid
from pathlib import Path

import cv2
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.media import frame_to_data_uri_jpeg, resize_bgr_max_width
from app.runtime import JOBS, STAGED
from utils.roi_launcher import launch_roi_tool, load_session_roi, roi_status, session_roi_path  # noqa: F401

router = APIRouter()


@router.post("/api/stage-upload")
async def api_stage_upload(video: UploadFile = File(...)):
    raw = await video.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")
    session_id = str(uuid.uuid4())
    job_root = Path(tempfile.mkdtemp(prefix="vlstage_"))
    suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"
    in_path = job_root / f"input{suffix}"
    in_path.write_bytes(raw)
    preview = ""
    try:
        cap = cv2.VideoCapture(str(in_path))
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            small = resize_bgr_max_width(frame, 480)
            preview = frame_to_data_uri_jpeg(small, 80)
    except Exception:
        preview = ""
    STAGED[session_id] = {
        "path": str(in_path),
        "name": video.filename or "upload",
        "root": str(job_root),
        "created": time.time(),
    }
    return {
        "session_id": session_id,
        "filename": video.filename,
        "preview": preview,
    }


@router.get("/api/roi/{session_id}")
def api_roi_status(session_id: str):
    if session_id not in STAGED:
        raise HTTPException(status_code=404, detail="Unknown session.")
    return roi_status(session_id)


@router.post("/api/roi/configure")
async def api_roi_configure(session_id: str = Form(...), mode: str = Form(...)):
    st = STAGED.get(session_id)
    if not st:
        raise HTTPException(status_code=404, detail="Unknown session — upload again in step 1.")
    if mode not in ("violation", "signal", "no_parking"):
        raise HTTPException(status_code=400, detail="Invalid ROI mode.")
    ok, msg = launch_roi_tool(st["path"], session_id, mode)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    return {"ok": True, "message": msg, "roi": roi_status(session_id)}


@router.get("/api/download/{job_id}")
def download(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job")
    vp = job.get("video")
    if not vp or not Path(vp).is_file():
        raise HTTPException(404, "No output video for this job")
    return FileResponse(
        vp,
        media_type="video/mp4",
        filename="violane_annotated.mp4",
    )
