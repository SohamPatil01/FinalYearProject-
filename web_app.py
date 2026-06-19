"""
VioLane web UI: FastAPI + HTML/JS (uploaded video only).

Live preview: POST /api/run-stream — Server-Sent Events with per-frame annotated JPEGs,
violations, and plate track state while the clip is processed.

Run: uvicorn web_app:app --reload --host 127.0.0.1 --port 8765
"""

from __future__ import annotations

import base64
import json
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import config
from utils.pipeline import TrafficPipeline
from utils.roi_launcher import launch_roi_tool, load_session_roi, roi_status, session_roi_path
from utils.rule_validation import count_yolo_loads, parse_csv_ids, validate_selection
from utils.ui_common import paths_from_model_ids
from utils.video_decode import iter_decode_media

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="VioLane")
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
JOBS: Dict[str, Dict[str, Any]] = {}
STAGED: Dict[str, Dict[str, Any]] = {}


def _expand_rules(model_ids: List[str], rules_raw: str) -> Set[str]:
    rules: Set[str] = set(parse_csv_ids(rules_raw))
    mids = set(model_ids)
    if "helmet" in mids:
        rules.add("helmet")
    if "triple" in mids:
        rules.add("triple")
    if "truck" in mids:
        rules.add("truck_restricted")
    if "plate" in mids:
        rules.add("plate_ocr")
    return rules


def _build_pipeline(
    model_ids: List[str],
    rules_raw: str,
    roi_config: Dict[str, Any],
    truck_start: int,
    truck_end: int,
) -> TrafficPipeline:
    enabled = _expand_rules(model_ids, rules_raw)
    paths = paths_from_model_ids(model_ids) if model_ids else {}
    return TrafficPipeline(
        model_paths=paths if paths else {},
        truck_violation_active_start_hour=int(truck_start),
        truck_violation_active_end_hour=int(truck_end),
        enabled_rules=enabled,
        roi_config=roi_config,
    )


def _thumb_data_uri(rgb: np.ndarray) -> str:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _frame_to_data_uri_jpeg(bgr: np.ndarray, quality: int = 82) -> str:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _resize_bgr_max_width(bgr: np.ndarray, max_w: int) -> np.ndarray:
    if max_w <= 0:
        return bgr
    h, w = bgr.shape[:2]
    if w <= max_w:
        return bgr
    nh = max(1, int(h * (max_w / w)))
    return cv2.resize(bgr, (max_w, nh), interpolation=cv2.INTER_AREA)


def _build_summary_dict(
    job_id: str,
    out_path: Path,
    captures: List[Dict[str, Any]],
    last_bgr: Optional[np.ndarray],
    n_frames: int,
    v_events: int,
    fps: float,
    dec_skip: int,
    est_decoded: int,
    pipeline: TrafficPipeline,
    zone_events: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    unique_tids = len({int(c["tid"]) for c in captures})
    summary: Dict[str, Any] = {
        "job_id": job_id,
        "models": sorted(pipeline.active_models),
        "frames_processed": n_frames,
        "violation_events": v_events,
        "fps": fps,
        "decode_stride": dec_skip,
        "frames_total_est": est_decoded,
        "stats": {
            "violations": v_events,
            "plates_locked": len(captures),
            "unique_plate_tracks": unique_tids,
            "frame": n_frames,
            "frame_total": est_decoded,
        },
        "rules": [
            {"id": "R1", "name": "Restricted hours (truck)", "key": "truck_restricted"},
            {"id": "R2", "name": "Triple seat", "key": "triple"},
            {"id": "R3", "name": "Helmet", "key": "helmet"},
            {"id": "R4", "name": "Plate OCR", "key": "plate_ocr"},
            {"id": "R5", "name": "Red light", "key": "red_light"},
            {"id": "R6", "name": "No parking", "key": "no_parking"},
        ],
        "poster": _frame_to_data_uri_jpeg(last_bgr) if last_bgr is not None else "",
        "plates": [
            {
                "text": c["text"],
                "frame": c["frame"],
                "tid": c["tid"],
                "ocr": c["ocr"],
                "yolo": c["yolo"],
                "thumb": _thumb_data_uri(c["thumb_rgb"]),
            }
            for c in captures
        ],
    }
    recent: List[Dict[str, Any]] = []
    for c in captures:
        fi = int(c["frame"])
        t_sec = round((fi * dec_skip) / max(fps, 1e-6), 1)
        recent.append(
            {
                "t_sec": t_sec,
                "vid": f"V{int(c['tid'])}",
                "zone": "—",
                "plate": c["text"],
                "frame": fi,
                "kind": "plate",
            }
        )
    for zev in zone_events or []:
        recent.append(
            {
                "t_sec": float(zev.get("t_sec", 0)),
                "vid": zev.get("vid") or "—",
                "zone": zev.get("zone") or "—",
                "plate": zev.get("plate") or "",
                "frame": int(zev.get("frame", 0)),
                "kind": zev.get("kind") or "zone",
                "summary": zev.get("summary") or "",
            }
        )
    recent.sort(key=lambda r: (float(r.get("t_sec", 0)), int(r.get("frame", 0))))
    summary["recent_events"] = list(reversed(recent))[:40]
    video_out = str(out_path) if out_path.is_file() and out_path.stat().st_size > 0 else None
    summary["download_ready"] = bool(video_out)
    return summary


def _run_full_pass(
    in_path: Path,
    out_path: Path,
    pipeline: TrafficPipeline,
) -> Tuple[int, int, List[Dict[str, Any]], Optional[np.ndarray], float, int, int]:
    done: Optional[Dict[str, Any]] = None
    last_frame: Optional[Dict[str, Any]] = None
    for ev in iter_decode_media(in_path, out_path, pipeline):
        if ev["kind"] == "frame":
            last_frame = ev
        else:
            done = ev
    if not done:
        raise RuntimeError("Processing produced no frames")
    n_frames = int(done["frame_idx"])
    est = int(done["est_decoded"])
    return (
        n_frames,
        int(done["cum_viol"]),
        done["captures"],
        done["last_bgr"],
        float(done["fps"]),
        int(done["dec_skip"]),
        est,
    )


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("violane.html", {"request": request})


@app.get("/api/catalog")
def api_catalog():
    entries = []
    paths = config.catalog_model_paths()
    for e in config.MODEL_CATALOG:
        p = paths.get(e["id"], "")
        ok = bool(p and config.is_model_file_usable(p))
        entries.append(
            {
                "id": e["id"],
                "title": e["title"],
                "summary": e.get("summary", ""),
                "ready": ok,
            }
        )
    rules = [
        {
            "id": r["id"],
            "title": r["title"],
            "summary": r.get("summary", ""),
            "needs_roi": r.get("needs_roi") or [],
            "requires_model": r.get("requires_model"),
        }
        for r in config.RULE_CATALOG
    ]
    return {
        "models": entries,
        "rules": rules,
        "heavy_combo_threshold": 3,
        "heavy_combo_hints": [
            "3+ YOLO weights selected — consider unchecking unused detectors.",
            "Red light + no parking together load yolov10s and yolov8n in addition to lane models.",
        ],
    }


@app.post("/api/stage-upload")
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
            small = _resize_bgr_max_width(frame, 480)
            preview = _frame_to_data_uri_jpeg(small, 80)
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


@app.get("/api/roi/{session_id}")
def api_roi_status(session_id: str):
    if session_id not in STAGED:
        raise HTTPException(status_code=404, detail="Unknown session.")
    return roi_status(session_id)


@app.post("/api/roi/configure")
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


@app.get("/api/download/{job_id}")
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


def _sse_pack(obj: Dict[str, Any]) -> bytes:
    return f"data: {json.dumps(obj, default=str)}\n\n".encode("utf-8")


@app.post("/api/run-stream")
async def api_run_stream(
    models: str = Form(""),
    rules: str = Form(""),
    session_id: str = Form(""),
    video: Optional[UploadFile] = File(None),
    truck_start: int = Form(6),
    truck_end: int = Form(22),
):
    mids = parse_csv_ids(models)
    rules_list = parse_csv_ids(rules)
    roi_config = load_session_roi(session_id) if session_id else {}
    errors, warnings = validate_selection(mids, rules_list, roi_config)
    if errors:
        raise HTTPException(status_code=400, detail="; ".join(errors))

    if session_id and session_id in STAGED:
        in_path = Path(STAGED[session_id]["path"])
        job_root = Path(STAGED[session_id]["root"])
    elif video is not None:
        raw = await video.read()
        job_id = str(uuid.uuid4())
        job_root = Path(tempfile.mkdtemp(prefix="vljob_"))
        suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"
        in_path = job_root / f"input{suffix}"
        in_path.write_bytes(raw)
        session_id = job_id
    else:
        raise HTTPException(status_code=400, detail="Upload a file in step 1 or provide session_id.")

    job_id = session_id or str(uuid.uuid4())
    out_path = job_root / "annotated.mp4"

    if not in_path.is_file() or not in_path.stat().st_size:
        raise HTTPException(status_code=400, detail="Empty or missing media file.")

    try:
        pipeline = _build_pipeline(mids, rules, roi_config, truck_start, truck_end)
    except Exception as e:
        shutil.rmtree(job_root, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Pipeline: {e}") from e

    if pipeline.use_plate:
        try:
            pipeline._get_ocr_reader()
        except Exception as e:
            shutil.rmtree(job_root, ignore_errors=True)
            raise HTTPException(status_code=500, detail=f"EasyOCR: {e}") from e

    stream_max = int(getattr(config, "WEB_STREAM_PREVIEW_MAX_WIDTH", 960))
    pace_stream = bool(getattr(config, "WEB_STREAM_REALTIME_PACE", False))
    stream_every = max(1, int(getattr(config, "WEB_STREAM_FRAME_EVERY_N", 1)))
    if pace_stream:
        stream_every = 1

    def event_gen() -> Generator[bytes, None, None]:
        try:
            yield _sse_pack(
                {
                    "type": "start",
                    "job_id": job_id,
                    "models": sorted(pipeline.active_models),
                    "models_loaded": pipeline.models_loaded,
                    "engines_active": pipeline.engines_active,
                    "warnings": warnings,
                    "yolo_load_count": count_yolo_loads(mids, rules_list),
                }
            )
            done_ev: Optional[Dict[str, Any]] = None
            seen_zone_keys: Set[Tuple[Any, ...]] = set()
            zone_recent: List[Dict[str, Any]] = []
            for ev in iter_decode_media(in_path, out_path, pipeline):
                if ev["kind"] == "frame":
                    _fi = int(ev["frame_idx"])
                    for eng_ev in ev["meta"].get("engine_events") or []:
                        tid = eng_ev.get("track_id")
                        if tid is not None:
                            zkey = (
                                eng_ev.get("violation_type"),
                                eng_ev.get("zone"),
                                int(tid),
                            )
                        else:
                            zkey = (
                                eng_ev.get("violation_type"),
                                eng_ev.get("zone"),
                                tuple(eng_ev.get("bbox") or []),
                            )
                        if zkey in seen_zone_keys:
                            continue
                        seen_zone_keys.add(zkey)
                        t_sec = round((_fi * ev["dec_skip"]) / max(ev["fps"], 1e-6), 1)
                        vid = f"V{int(tid)}" if tid is not None else "—"
                        summary_txt = str(eng_ev.get("summary") or eng_ev.get("violation_type") or "Violation")
                        zone = str(eng_ev.get("zone") or "—")
                        plate = str(eng_ev.get("plate") or "")
                        zone_recent.append(
                            {
                                "t_sec": t_sec,
                                "vid": vid,
                                "zone": zone,
                                "plate": plate,
                                "frame": _fi,
                                "kind": str(eng_ev.get("violation_type") or "zone"),
                                "summary": summary_txt,
                            }
                        )
                        yield _sse_pack(
                            {
                                "type": "violation_new",
                                "summary": summary_txt,
                                "zone": zone,
                                "vid": vid,
                                "plate": plate,
                                "frame": _fi,
                                "t_sec": t_sec,
                                "violation_type": eng_ev.get("violation_type"),
                            }
                        )
                    for c in ev["new_captures"]:
                        t_sec = round((ev["frame_idx"] * ev["dec_skip"]) / max(ev["fps"], 1e-6), 1)
                        yield _sse_pack(
                            {
                                "type": "plate_new",
                                "text": c["text"],
                                "tid": int(c["tid"]),
                                "frame": ev["frame_idx"],
                                "t_sec": t_sec,
                                "thumb": _thumb_data_uri(c["thumb_rgb"]),
                                "ocr": float(c["ocr"]),
                            }
                        )
                    if ev["frame_idx"] % stream_every == 0:
                        small = _resize_bgr_max_width(ev["processed"], stream_max)
                        plates_live = []
                        for p in ev["meta"].get("plates") or []:
                            plates_live.append(
                                {
                                    "tid": p.get("track_id"),
                                    "text": (p.get("text") or "")[:40],
                                    "pending": bool(p.get("pending")),
                                    "yolo": round(float(p.get("yolo_conf", 0)), 2),
                                    "sharp": round(float(p.get("sharpness", 0)), 0),
                                }
                            )
                        yield _sse_pack(
                            {
                                "type": "frame",
                                "frame": ev["frame_idx"],
                                "frame_total_est": ev["frame_total_est"],
                                "fps": ev["fps"],
                                "violations": ev["violations"],
                                "violations_total": ev["cum_viol"],
                                "plates": plates_live,
                                "plates_locked": ev["plates_locked_count"],
                                "unique_tracks": ev["unique_plate_tracks"],
                                "image": _frame_to_data_uri_jpeg(small, 78),
                            }
                        )
                    if pace_stream:
                        fpsv = max(float(ev["fps"]), 1e-6)
                        dsk = int(ev["dec_skip"])
                        cap_pf = float(getattr(config, "WEB_PACE_MAX_FPS", 0.0) or 0.0)
                        eff_fps = min(fpsv, cap_pf) if cap_pf > 0 else fpsv
                        want_s = dsk / max(eff_fps, 1e-6)
                        # Deadline from start of frame processing (see video_decode._pace_t0), not post-decode.
                        pace_t0 = float(ev.get("_pace_t0", time.perf_counter()))
                        delay = (pace_t0 + want_s) - time.perf_counter()
                        if delay > 0:
                            time.sleep(delay)
                else:
                    done_ev = ev

            if not done_ev:
                yield _sse_pack({"type": "error", "message": "No frames decoded"})
                return

            summary = _build_summary_dict(
                job_id,
                out_path,
                done_ev["captures"],
                done_ev["last_bgr"],
                int(done_ev["frame_idx"]),
                int(done_ev["cum_viol"]),
                float(done_ev["fps"]),
                int(done_ev["dec_skip"]),
                int(done_ev["est_decoded"]),
                pipeline,
                zone_events=zone_recent,
            )
            video_p = str(out_path) if summary.get("download_ready") else None
            JOBS[job_id] = {"root": str(job_root), "video": video_p, "created": time.time()}
            try:
                in_path.unlink(missing_ok=True)
            except OSError:
                pass
            yield _sse_pack({"type": "done", **summary})
        except Exception as e:
            shutil.rmtree(job_root, ignore_errors=True)
            JOBS.pop(job_id, None)
            yield _sse_pack({"type": "error", "message": str(e)})

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_gen(), media_type="text/event-stream", headers=headers)


@app.post("/api/run")
async def api_run(
    models: str = Form(""),
    rules: str = Form(""),
    session_id: str = Form(""),
    video: Optional[UploadFile] = File(None),
    truck_start: int = Form(6),
    truck_end: int = Form(22),
):
    mids = parse_csv_ids(models)
    rules_list = parse_csv_ids(rules)
    roi_config = load_session_roi(session_id) if session_id else {}
    errors, _warnings = validate_selection(mids, rules_list, roi_config)
    if errors:
        raise HTTPException(status_code=400, detail="; ".join(errors))

    if session_id and session_id in STAGED:
        in_path = Path(STAGED[session_id]["path"])
        job_root = Path(STAGED[session_id]["root"])
        job_id = session_id
    elif video is not None:
        job_id = str(uuid.uuid4())
        job_root = Path(tempfile.mkdtemp(prefix="vljob_"))
        suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"
        in_path = job_root / f"input{suffix}"
        in_path.write_bytes(await video.read())
    else:
        raise HTTPException(status_code=400, detail="Upload a file or provide session_id.")

    out_path = job_root / "annotated.mp4"

    if not in_path.is_file() or not in_path.stat().st_size:
        shutil.rmtree(job_root, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Empty file.")

    try:
        pipeline = _build_pipeline(mids, rules, roi_config, truck_start, truck_end)
    except Exception as e:
        shutil.rmtree(job_root, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Pipeline: {e}") from e

    if pipeline.use_plate:
        try:
            pipeline._get_ocr_reader()
        except Exception as e:
            shutil.rmtree(job_root, ignore_errors=True)
            raise HTTPException(status_code=500, detail=f"EasyOCR: {e}") from e

    video_out: str | None = None
    try:
        n_frames, v_events, captures, last_bgr, fps, dec_skip, est_decoded = _run_full_pass(
            in_path, out_path, pipeline
        )
        summary = _build_summary_dict(
            job_id,
            out_path,
            captures,
            last_bgr,
            n_frames,
            v_events,
            fps,
            dec_skip,
            est_decoded,
            pipeline,
        )
        if out_path.is_file() and out_path.stat().st_size > 0:
            video_out = str(out_path)
        JOBS[job_id] = {"root": str(job_root), "video": video_out, "created": time.time()}
    except Exception as e:
        shutil.rmtree(job_root, ignore_errors=True)
        JOBS.pop(job_id, None)
        raise HTTPException(status_code=500, detail=str(e)) from e

    try:
        in_path.unlink(missing_ok=True)
    except OSError:
        pass

    return JSONResponse(summary)
