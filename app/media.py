"""Frame/thumbnail encoding and result-summary serialization.

Moved verbatim from ``web_app.py`` — same encodings, same summary shape.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from utils.pipeline import TrafficPipeline


def thumb_data_uri(rgb: np.ndarray) -> str:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def frame_to_data_uri_jpeg(bgr: np.ndarray, quality: int = 82) -> str:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def resize_bgr_max_width(bgr: np.ndarray, max_w: int) -> np.ndarray:
    if max_w <= 0:
        return bgr
    h, w = bgr.shape[:2]
    if w <= max_w:
        return bgr
    nh = max(1, int(h * (max_w / w)))
    return cv2.resize(bgr, (max_w, nh), interpolation=cv2.INTER_AREA)


def build_summary_dict(
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
        "poster": frame_to_data_uri_jpeg(last_bgr) if last_bgr is not None else "",
        "plates": [
            {
                "text": c["text"],
                "frame": c["frame"],
                "tid": c["tid"],
                "ocr": c["ocr"],
                "yolo": c["yolo"],
                "thumb": thumb_data_uri(c["thumb_rgb"]),
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
