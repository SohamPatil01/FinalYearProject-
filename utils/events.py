"""Normalized violation events across lane, red-light, and no-parking engines."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class ViolationEvent(TypedDict, total=False):
    violation_type: str
    time_sec: float
    frame_index: int
    summary: str
    zone: str
    track_id: int
    evidence_path: str
    plate: str


def normalize_engine_events(raw_events: List[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    for ev in raw_events or []:
        summary = str(ev.get("summary") or "").strip()
        if not summary:
            vtype = str(ev.get("violation_type") or "violation")
            summary = vtype.replace("_", " ").title()
        zone = ev.get("zone")
        if zone:
            summary = f"{summary} · {zone}"
        plate = ev.get("plate")
        if plate:
            summary = f"{summary} | {plate}"
        lines.append(summary)
    return lines


def merge_snapshots(meta: Dict[str, Any], new_snaps: List[Dict[str, Any]]) -> None:
    existing = list(meta.get("violation_snapshots") or [])
    existing.extend(new_snaps or [])
    meta["violation_snapshots"] = existing


def event_to_snapshot(ev: Dict[str, Any], frame_bgr, default_message: str) -> Optional[Dict[str, Any]]:
    bbox = ev.get("bbox")
    if not bbox or frame_bgr is None:
        return None
    try:
        import cv2
        import numpy as np

        from utils.ui_common import resize_preview_rgb

        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame_bgr[y1:y2, x1:x2]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        thumb = resize_preview_rgb(rgb, 140)
        return {
            "message": str(ev.get("summary") or default_message),
            "bbox": [x1, y1, x2, y2],
            "thumb_rgb": thumb,
            "frame": int(ev.get("frame_index", 0)),
        }
    except Exception:
        return None
