"""Run routes: streaming (SSE) and one-shot analysis.

Logic moved verbatim from ``web_app.py`` — identical events, summary, and
job lifecycle.
"""

from __future__ import annotations

import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

import config
from app.media import (
    build_summary_dict,
    frame_to_data_uri_jpeg,
    resize_bgr_max_width,
    thumb_data_uri,
)
from app.pipeline_factory import build_pipeline, run_full_pass, sse_pack
from app.runtime import JOBS, STAGED
from utils.rule_validation import count_yolo_loads, parse_csv_ids, validate_selection
from utils.roi_launcher import load_session_roi
from utils.video_decode import iter_decode_media

router = APIRouter()


@router.post("/api/run-stream")
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
        pipeline = build_pipeline(mids, rules, roi_config, truck_start, truck_end)
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
            yield sse_pack(
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
                        yield sse_pack(
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
                        yield sse_pack(
                            {
                                "type": "plate_new",
                                "text": c["text"],
                                "tid": int(c["tid"]),
                                "frame": ev["frame_idx"],
                                "t_sec": t_sec,
                                "thumb": thumb_data_uri(c["thumb_rgb"]),
                                "ocr": float(c["ocr"]),
                            }
                        )
                    if ev["frame_idx"] % stream_every == 0:
                        small = resize_bgr_max_width(ev["processed"], stream_max)
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
                        yield sse_pack(
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
                                "image": frame_to_data_uri_jpeg(small, 78),
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
                yield sse_pack({"type": "error", "message": "No frames decoded"})
                return

            summary = build_summary_dict(
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
            yield sse_pack({"type": "done", **summary})
        except Exception as e:
            shutil.rmtree(job_root, ignore_errors=True)
            JOBS.pop(job_id, None)
            yield sse_pack({"type": "error", "message": str(e)})

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_gen(), media_type="text/event-stream", headers=headers)


@router.post("/api/run")
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
        pipeline = build_pipeline(mids, rules, roi_config, truck_start, truck_end)
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
        n_frames, v_events, captures, last_bgr, fps, dec_skip, est_decoded = run_full_pass(
            in_path, out_path, pipeline
        )
        summary = build_summary_dict(
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
