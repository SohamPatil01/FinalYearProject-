"""Detection utilities for running multiple YOLO models.

Plate boxes used for OCR always come from the dedicated ``plate.pt`` pass (``infer_plate``), not from
truck, triple, or helmet checkpoints even if those define plate-like classes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
from ultralytics import YOLO

import config


def _iou_xyxy(a: Sequence[int], b: Sequence[int]) -> float:
    ax1, ay1, ax2, ay2 = (int(a[0]), int(a[1]), int(a[2]), int(a[3]))
    bx1, by1, bx2, by2 = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    bb = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def _nms_plate_dets(dets: List[dict], iou_thresh: float) -> List[dict]:
    """Greedy NMS on plate detections (higher confidence first)."""
    if len(dets) <= 1:
        return dets
    sorted_d = sorted(dets, key=lambda d: float(d.get("confidence", 0.0)), reverse=True)
    keep: List[dict] = []
    for d in sorted_d:
        bb = d["bbox"]
        if any(_iou_xyxy(bb, k["bbox"]) >= iou_thresh for k in keep):
            continue
        keep.append(d)
    return keep


def _expand_bbox_xyxy(
    bbox: Sequence[int], pad_frac: float, frame_h: int, frame_w: int
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    px, py = int(bw * pad_frac), int(bh * pad_frac)
    nx1 = max(0, x1 - px)
    ny1 = max(0, y1 - py)
    nx2 = min(frame_w, x2 + px)
    ny2 = min(frame_h, y2 + py)
    return nx1, ny1, nx2, ny2


def expand_bbox_xyxy(
    bbox: Sequence[int], pad_frac: float, frame_h: int, frame_w: int
) -> Tuple[int, int, int, int]:
    """Public helper: expand a box by ``pad_frac`` of its width/height, clamped to the frame."""
    return _expand_bbox_xyxy(bbox, pad_frac, frame_h, frame_w)


class MultiModelDetector:
    """Loads and runs multiple YOLO models, returning a unified output format."""

    def __init__(self, model_paths: Dict[str, str]) -> None:
        self.models: Dict[str, YOLO] = {}

        for model_name, model_path in model_paths.items():
            if not Path(model_path).exists():
                print(f"[WARN] Model file not found, skipping: {model_path}")
                continue

            try:
                self.models[model_name] = YOLO(model_path)
                print(f"[INFO] Loaded model '{model_name}' from {model_path}")
            except Exception as exc:  # pragma: no cover - runtime protection
                print(f"[ERROR] Could not load model '{model_name}': {exc}")

    def infer(self, frame, skip_models: Optional[Set[str]] = None) -> List[dict]:
        """
        Run inference for each loaded model on one frame.

        skip_models: optional set of model keys to skip this frame (e.g. reuse cached plate boxes).

        Returns list of detections in unified format:
        {
            "model": "truck",
            "class": int,
            "confidence": float,
            "bbox": [x1, y1, x2, y2]
        }
        """
        all_detections: List[dict] = []

        skip = skip_models or set()
        default_sz = int(getattr(config, "YOLO_IMGSZ", 640))
        plate_key = getattr(config, "PLATE_MODEL_KEY", "plate")
        plate_sz = int(getattr(config, "YOLO_PLATE_IMGSZ", default_sz))
        use_half = bool(getattr(config, "YOLO_HALF_PRECISION", False)) and torch.cuda.is_available()

        for model_name, model in self.models.items():
            if model_name in skip:
                continue
            imgsz = plate_sz if model_name == plate_key else default_sz
            try:
                results = model(frame, verbose=False, imgsz=imgsz, half=use_half)
            except Exception:
                results = model(frame, verbose=False, imgsz=imgsz, half=False)
            if not results:
                continue

            result = results[0]
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                conf = float(boxes.conf[i].cpu().numpy().item())
                cls_id = int(boxes.cls[i].cpu().numpy().item())

                all_detections.append(
                    {
                        "model": model_name,
                        "class": cls_id,
                        "confidence": conf,
                        "bbox": [int(v) for v in xyxy],
                    }
                )

        return all_detections

    def infer_plate(
        self,
        frame,
        truck_boxes: Optional[List[Sequence[int]]] = None,
        *,
        use_truck_roi: bool = False,
        truck_roi_pad_frac: float = 0.15,
        include_full_frame_when_roi: bool = False,
    ) -> List[dict]:
        """
        Run only the plate YOLO on the full frame, or on crops around each truck box.

        When ``use_truck_roi`` is True and ``truck_boxes`` is non-empty, each truck ROI
        is expanded by ``truck_roi_pad_frac``. Optionally (see ``TRUCK_PLATE_ROI_BOTTOM_HALF_ONLY``)
        plate YOLO runs only on the bottom portion of that ROI (reference: bumper / plate strip).
        Detections are mapped back to full-frame coordinates and merged with NMS.
        """
        plate_key = getattr(config, "PLATE_MODEL_KEY", "plate")
        model = self.models.get(plate_key)
        if model is None:
            return []

        h, w = frame.shape[:2]
        default_sz = int(getattr(config, "YOLO_IMGSZ", 640))
        plate_sz = int(getattr(config, "YOLO_PLATE_IMGSZ", default_sz))
        use_half = bool(getattr(config, "YOLO_HALF_PRECISION", False)) and torch.cuda.is_available()
        merge_iou = float(getattr(config, "PLATE_ROI_MERGE_IOU", 0.45))
        min_pc = float(getattr(config, "PLATE_YOLO_MIN_CONF", 0.0))

        def _run_on_bgr(bgr, ox: int = 0, oy: int = 0) -> List[dict]:
            out: List[dict] = []
            try:
                results = model(bgr, verbose=False, imgsz=plate_sz, half=use_half)
            except Exception:
                results = model(bgr, verbose=False, imgsz=plate_sz, half=False)
            if not results:
                return out
            result = results[0]
            boxes = result.boxes
            if boxes is None:
                return out
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                conf = float(boxes.conf[i].cpu().numpy().item())
                if conf < min_pc:
                    continue
                cls_id = int(boxes.cls[i].cpu().numpy().item())
                gx1 = int(xyxy[0]) + ox
                gy1 = int(xyxy[1]) + oy
                gx2 = int(xyxy[2]) + ox
                gy2 = int(xyxy[3]) + oy
                out.append(
                    {
                        "model": plate_key,
                        "class": cls_id,
                        "confidence": conf,
                        "bbox": [gx1, gy1, gx2, gy2],
                    }
                )
            return out

        collected: List[dict] = []
        if use_truck_roi and truck_boxes:
            bottom_only = bool(getattr(config, "TRUCK_PLATE_ROI_BOTTOM_HALF_ONLY", False))
            v0 = float(getattr(config, "TRUCK_PLATE_ROI_VERTICAL_START_FRAC", 0.5) or 0.0)
            v0 = max(0.0, min(0.95, v0))

            if include_full_frame_when_roi:
                collected.extend(_run_on_bgr(frame, 0, 0))

            for tb in truck_boxes:
                x1, y1, x2, y2 = _expand_bbox_xyxy(tb, truck_roi_pad_frac, h, w)
                if x2 <= x1 or y2 <= y1:
                    continue
                if bottom_only:
                    rh = max(1, y2 - y1)
                    y_top = y1 + int(rh * v0)
                    if y_top >= y2:
                        y_top = y1 + rh // 2
                    crop = frame[y_top:y2, x1:x2]
                    oy = int(y_top)
                else:
                    crop = frame[y1:y2, x1:x2]
                    oy = y1
                if crop.size == 0:
                    continue
                collected.extend(_run_on_bgr(crop, ox=x1, oy=oy))
            collected = _nms_plate_dets(collected, merge_iou)
            return collected

        return _run_on_bgr(frame, 0, 0)
