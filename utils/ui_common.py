"""UI-agnostic helpers shared by Streamlit, FastAPI, or other front-ends."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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
    store_crop: bool = False,
    track_best: Optional[Dict[Any, float]] = None,
) -> None:
    """Collect plate crop thumbnails into ``captures``.

    Normal mode keeps the first decent crop per (track[, text]). When ``track_best``
    is provided (deferred-OCR mode), it instead keeps the single *highest quality*
    crop per track -- replacing the earlier crop in place when a sharper/larger one
    appears -- so the batch OCR at the end reads from the best available image.
    ``store_crop`` additionally retains the full-resolution BGR crop (``crop_bgr``)
    needed by that later OCR pass.
    """
    if frame_bgr is None:
        return
    include_pending = bool(getattr(config, "PLATE_GALLERY_INCLUDE_PENDING", False))
    pending_min_yolo = float(getattr(config, "PLATE_GALLERY_PENDING_MIN_YOLO", 0.45))
    h, w = frame_bgr.shape[:2]
    for p in plates:
        tid = p.get("track_id", -1)
        text = (str(p.get("text") or "")).strip()
        is_read = (not p.get("pending")) and len(text) >= 1
        yolo_conf = float(p.get("yolo_conf", 0.0))
        # Deferred OCR: pick the best crop per track (no text yet, so always "pending").
        best_mode = track_best is not None and not is_read

        if is_read:
            # Confirmed OCR read: one capture per (track, text).
            key: Any = (tid, text)
            if key in seen:
                continue
        elif best_mode or include_pending:
            # Plate detected but not yet read -> still show the crop so the
            # extracted-plate image is visible even before/without an OCR read.
            if yolo_conf < pending_min_yolo:
                continue
            if not best_mode:
                key = ("pending", tid)
                if key in seen:
                    continue
        else:
            continue

        x1, y1, x2, y2 = [int(x) for x in p["bbox"]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        if best_mode:
            # Quality proxy: confident, large, sharp crops read best.
            area = (x2 - x1) * (y2 - y1)
            sharp = float(p.get("sharpness", 0.0)) or 1.0
            quality = yolo_conf * (area ** 0.5) * sharp
            prev_q = track_best.get(tid)
            if prev_q is not None and quality <= prev_q:
                continue

        # Same padded crop as EasyOCR uses in ``read_plate_from_crop`` (``_safe_crop``).
        crop = _safe_crop(frame_bgr, x1, y1, x2, y2)
        if crop.size == 0:
            continue
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tw = max(48, int(thumb_w))
        thumb = resize_preview_rgb(crop_rgb, tw) if crop_rgb.shape[1] > tw else crop_rgb
        item: Dict[str, Any] = {
            "text": text if is_read else "reading…",
            "frame": frame_idx,
            "tid": tid,
            "thumb_rgb": thumb,
            "ocr": float(p.get("confidence", 0.0)),
            "yolo": yolo_conf,
        }
        if store_crop:
            item["crop_bgr"] = crop

        if best_mode:
            if track_best.get(tid) is not None:
                for i, c in enumerate(captures):
                    if c.get("tid") == tid:
                        captures[i] = item
                        break
                else:
                    captures.append(item)
            else:
                captures.append(item)
            track_best[tid] = quality
        else:
            seen.add(key)
            captures.append(item)
        while len(captures) > max_items:
            captures.pop(0)

