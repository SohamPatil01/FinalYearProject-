"""UI-agnostic helpers shared by Streamlit, FastAPI, or other front-ends."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import cv2

import config
from utils.plate_ocr import _safe_crop


def model_options() -> Tuple[List[str], List[str], Dict[str, str]]:
    all_paths = config.catalog_model_paths()
    labels: List[str] = []
    label_to_id: Dict[str, str] = {}
    defaults: List[str] = []

    for entry in config.MODEL_CATALOG:
        mid = entry["id"]
        path = all_paths[mid]
        if not config.is_model_file_usable(path):
            continue
        lab = f"{entry['title']} ({mid})"
        labels.append(lab)
        label_to_id[lab] = mid

    # No detectors on by default — user picks models on the first screen, then uploads.
    return labels, defaults, label_to_id


def resize_preview_rgb(rgb, max_width: int):
    if max_width <= 0:
        return rgb
    h, w = rgb.shape[0], rgb.shape[1]
    if w <= max_width:
        return rgb
    nh = max(1, int(h * (max_width / w)))
    return cv2.resize(rgb, (max_width, nh), interpolation=cv2.INTER_AREA)


def paths_from_labels(selected: List[str], label_to_id: Dict[str, str]) -> Dict[str, str]:
    all_paths = config.catalog_model_paths()
    out: Dict[str, str] = {}
    for lab in selected or []:
        mid = label_to_id.get(lab)
        if mid and mid in all_paths:
            out[mid] = all_paths[mid]
    return out


def paths_from_model_ids(model_ids: List[str]) -> Dict[str, str]:
    """Build `model_paths` for TrafficPipeline from catalog ids (e.g. truck, plate)."""
    all_paths = config.catalog_model_paths()
    out: Dict[str, str] = {}
    for mid in model_ids or []:
        mid = str(mid).strip()
        p = all_paths.get(mid)
        if p and config.is_model_file_usable(p):
            out[mid] = p
    return out


def write_upload_to_temp(uploaded) -> str:
    """Streamlit UploadedFile: `.name` and `.getvalue()`."""
    suffix = Path(uploaded.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        return tmp.name


def write_path_copy(src_path: str) -> str:
    """Copy an on-disk file (e.g. Gradio temp path) to a NamedTemporaryFile."""
    p = Path(src_path)
    suffix = p.suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(p.read_bytes())
        return tmp.name


def append_plate_capture_from_frame(
    frame_bgr,
    plates: List[Dict[str, Any]],
    *,
    frame_idx: int,
    seen: Set[Any],
    captures: List[Dict[str, Any]],
    max_items: int,
    thumb_w: int,
) -> None:
    if frame_bgr is None:
        return
    h, w = frame_bgr.shape[:2]
    for p in plates:
        if p.get("pending"):
            continue
        text = (str(p.get("text") or "")).strip()
        if len(text) < 1:
            continue
        tid = p.get("track_id", -1)
        key = (tid, text)
        if key in seen:
            continue
        seen.add(key)
        x1, y1, x2, y2 = [int(x) for x in p["bbox"]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        # Same padded crop as EasyOCR uses in ``read_plate_from_crop`` (``_safe_crop``).
        crop = _safe_crop(frame_bgr, x1, y1, x2, y2)
        if crop.size == 0:
            continue
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tw = max(48, int(thumb_w))
        thumb = resize_preview_rgb(crop_rgb, tw) if crop_rgb.shape[1] > tw else crop_rgb
        captures.append(
            {
                "text": text,
                "frame": frame_idx,
                "tid": tid,
                "thumb_rgb": thumb,
                "ocr": float(p.get("confidence", 0.0)),
                "yolo": float(p.get("yolo_conf", 0.0)),
            }
        )
        while len(captures) > max_items:
            captures.pop(0)

