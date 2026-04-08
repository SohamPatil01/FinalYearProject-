"""
Track plate boxes from **plate YOLO** and run OCR **only on the plate crop image**
(``_safe_crop`` → ``read_plate_from_crop``). The full video frame is never sent to OCR.

Plate detection and violation YOLOs run in ``TrafficPipeline.process_frame`` before this;
OCR attempts are gated by quality + stability so we do not OCR blurry or jumping crops every frame.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

import config
from utils.plate_ocr import _safe_crop, read_plate_from_crop
from utils.tracker import CentroidTracker


def _iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = a + b - inter
    return inter / union if union > 0 else 0.0


def _centroid(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _crop_sharpness(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    if gray.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _plate_geometry_ok(bbox: Tuple[int, int, int, int]) -> Tuple[bool, float, int]:
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    area = bw * bh
    ar = bw / float(bh)
    min_a = int(getattr(config, "PLATE_OCR_MIN_AREA", 1800))
    min_ar = float(getattr(config, "PLATE_OCR_MIN_ASPECT", 1.7))
    max_ar = float(getattr(config, "PLATE_OCR_MAX_ASPECT", 6.0))
    ok = area >= min_a and min_ar <= ar <= max_ar
    return ok, ar, area


class PlateOCRGate:
    """Centroid-tracks plate boxes; runs EasyOCR only when quality + stability gates pass."""

    def __init__(self) -> None:
        md = int(getattr(config, "PLATE_TRACK_MAX_DISTANCE", 95))
        miss = int(getattr(config, "PLATE_TRACK_MAX_DISAPPEARED", 18))
        self.tracker = CentroidTracker(max_disappeared=miss, max_distance=md)
        self._meta: Dict[int, Dict[str, Any]] = {}
        self._frame_count = 0

    def _best_det_for_bbox(self, bbox: Tuple[int, int, int, int], plate_dets: List[dict]) -> Optional[dict]:
        best, best_iou = None, 0.0
        t = tuple(int(x) for x in bbox)
        iou_need = float(getattr(config, "PLATE_OCR_DET_IOU_MIN", 0.08))
        for d in plate_dets:
            db = tuple(int(x) for x in d["bbox"])
            v = _iou(t, db)
            if v > best_iou:
                best_iou = v
                best = d
        return best if best_iou >= iou_need else None

    def update(
        self,
        frame: np.ndarray,
        plate_dets: List[dict],
        reader: Union[Any, Callable[[], Any]],
    ) -> List[dict]:
        self._frame_count += 1

        if plate_dets:
            rects = [tuple(int(x) for x in d["bbox"]) for d in plate_dets]
            tracked = self.tracker.update(rects)
        else:
            # Let CentroidTracker decay IDs; do not wipe tracks/meta on one missed frame.
            tracked = self.tracker.update([])

        active_ids = set(tracked.keys())
        for tid in list(self._meta.keys()):
            if tid not in active_ids:
                del self._meta[tid]

        if not tracked:
            return []

        min_yolo = float(getattr(config, "PLATE_OCR_MIN_YOLO_CONF", 0.52))
        min_sharp = float(getattr(config, "PLATE_OCR_MIN_SHARPNESS", 35.0))
        stable_need = int(getattr(config, "PLATE_OCR_STABLE_FRAMES", 4))
        drift = float(getattr(config, "PLATE_OCR_MAX_CENTROID_DRIFT", 14.0))
        min_chars = int(getattr(config, "PLATE_OCR_MIN_TEXT_LEN", 4))
        allow_upgrade = bool(getattr(config, "PLATE_OCR_ALLOW_UPGRADE", False))
        upgrade_ratio = float(getattr(config, "PLATE_OCR_UPGRADE_QUALITY_RATIO", 1.12))
        retry_gap = int(getattr(config, "PLATE_OCR_RETRY_MIN_FRAMES", 6))
        max_fails = max(1, int(getattr(config, "PLATE_OCR_MAX_TRIES_PER_TRACK", 2)))
        one_shot = bool(getattr(config, "PLATE_OCR_ONE_SHOT_PER_TRACK", False))
        frame_ocr_budget = max(0, int(getattr(config, "PLATE_OCR_MAX_TRIES_PER_FRAME", 1)))
        ocr_stride = max(1, int(getattr(config, "PLATE_OCR_ATTEMPT_EVERY_N_FRAMES", 1)))
        on_ocr_frame = (self._frame_count % ocr_stride) == 0

        out: List[dict] = []

        smooth_a = float(getattr(config, "PLATE_BBOX_SMOOTH_ALPHA", 0.0))

        for tid, bbox in tracked.items():
            raw_bbox = tuple(float(x) for x in bbox)

            if tid not in self._meta:
                self._meta[tid] = {
                    "stable": 0,
                    "last_c": None,
                    "text": "",
                    "text_hist": [],
                    "ocr_conf": 0.0,
                    "has_ocr": False,
                    "best_quality": 0.0,
                    "ocr_error": False,
                    "ocr_fail_count": 0,
                    "ocr_attempted": False,
                    "last_ocr_try_frame": -999,
                    "last_yolo_conf": 0.0,
                    "smooth_bbox": None,
                    "_gates_ok_prev": False,
                }
            m = self._meta[tid]
            if smooth_a > 0.0 and smooth_a <= 1.0:
                prev = m.get("smooth_bbox")
                if prev is None:
                    m["smooth_bbox"] = list(raw_bbox)
                else:
                    a = smooth_a
                    for i in range(4):
                        prev[i] = (1.0 - a) * float(prev[i]) + a * raw_bbox[i]
                sx1, sy1, sx2, sy2 = m["smooth_bbox"]
                bbox_use = (int(round(sx1)), int(round(sy1)), int(round(sx2)), int(round(sy2)))
            else:
                m["smooth_bbox"] = None
                bbox_use = tuple(int(x) for x in raw_bbox)

            bbox_i = [bbox_use[0], bbox_use[1], bbox_use[2], bbox_use[3]]

            cx, cy = _centroid(bbox_use)
            lc = m["last_c"]
            if lc is not None:
                dist = math.hypot(cx - lc[0], cy - lc[1])
                if dist <= drift:
                    m["stable"] = int(m["stable"]) + 1
                else:
                    m["stable"] = 0
            m["last_c"] = (cx, cy)

            det = self._best_det_for_bbox(bbox_use, plate_dets) if plate_dets else None
            if det is not None:
                m["last_yolo_conf"] = float(det.get("confidence", 0.0))
            yolo_conf = float(det["confidence"]) if det is not None else float(m.get("last_yolo_conf", 0.0))

            geom_ok, aspect, area = _plate_geometry_ok(bbox_use)
            sharp = _crop_sharpness(frame, bbox_use)
            quality = yolo_conf * math.sqrt(float(area) / 4000.0) * (sharp / max(min_sharp, 1.0))
            quality = min(quality, 3.0)

            # Strong view: allow OCR even if aspect filter disagrees (angled / vertical plates).
            geom_or_sharp_bypass = (
                geom_ok
                or (
                    sharp >= float(getattr(config, "PLATE_OCR_BYPASS_GEOM_SHARPNESS", 120.0))
                    and yolo_conf >= min_yolo
                    and m["stable"] >= stable_need
                )
            )

            gates_ok = (
                yolo_conf >= min_yolo
                and geom_or_sharp_bypass
                and sharp >= min_sharp
                and m["stable"] >= stable_need
            )
            prev_gate = bool(m.get("_gates_ok_prev", False))
            gate_rise = gates_ok and not prev_gate
            m["_gates_ok_prev"] = gates_ok

            skip_edge = bool(getattr(config, "PLATE_OCR_SKIP_STRIDE_ON_STABLE_EDGE", False))
            if not m["has_ocr"]:
                stride_ok = (gate_rise or on_ocr_frame) if skip_edge else on_ocr_frame
            else:
                stride_ok = on_ocr_frame

            def run_ocr() -> None:
                try:
                    rdr = reader() if callable(reader) else reader
                    crop = _safe_crop(frame, bbox_i[0], bbox_i[1], bbox_i[2], bbox_i[3])
                    if crop.size == 0:
                        m["ocr_fail_count"] = int(m.get("ocr_fail_count", 0)) + 1
                        return
                    txt, ocf = read_plate_from_crop(rdr, crop)
                    if txt and len(txt) >= min_chars:
                        hist_n = max(3, int(getattr(config, "PLATE_TEXT_STABILIZE_WINDOW", 7)))
                        hist = list(m.get("text_hist", []))
                        hist.append(str(txt))
                        if len(hist) > hist_n:
                            hist = hist[-hist_n:]
                        m["text_hist"] = hist
                        m["text"] = Counter(hist).most_common(1)[0][0]
                        m["ocr_conf"] = ocf
                        m["has_ocr"] = True
                        m["best_quality"] = quality
                        m["ocr_error"] = False
                        m["ocr_fail_count"] = 0
                    else:
                        m["ocr_fail_count"] = int(m.get("ocr_fail_count", 0)) + 1
                except Exception:
                    m["ocr_error"] = True
                    m["ocr_fail_count"] = int(m.get("ocr_fail_count", 0)) + 1

            if not m["has_ocr"]:
                fail_count = int(m.get("ocr_fail_count", 0))
                # Exponential retry backoff for failed tracks: retry_gap, 2x, 4x...
                effective_retry_gap = retry_gap * (2 ** min(fail_count, 2))
                if (
                    frame_ocr_budget > 0
                    and gates_ok
                    and stride_ok
                    and (not one_shot or not bool(m.get("ocr_attempted", False)))
                    and fail_count < max_fails
                    and (self._frame_count - int(m["last_ocr_try_frame"]) >= effective_retry_gap)
                ):
                    m["last_ocr_try_frame"] = self._frame_count
                    m["ocr_attempted"] = True
                    frame_ocr_budget -= 1
                    run_ocr()
            elif allow_upgrade and gates_ok and quality >= float(m["best_quality"]) * upgrade_ratio:
                if (
                    frame_ocr_budget > 0
                    and on_ocr_frame
                    and self._frame_count - int(m.get("last_ocr_try_frame", -999)) >= max(3, retry_gap // 2)
                ):
                    m["last_ocr_try_frame"] = self._frame_count
                    frame_ocr_budget -= 1
                    run_ocr()

            pending = not bool(m["has_ocr"])
            raw_box = None
            if det is not None and det.get("bbox_raw") is not None:
                raw_box = [int(x) for x in det["bbox_raw"]]
            out.append(
                {
                    "track_id": tid,
                    "bbox": bbox_i,
                    "bbox_raw": raw_box,
                    "text": str(m.get("text", "")),
                    "confidence": float(m.get("ocr_conf", 0.0)),
                    "yolo_conf": yolo_conf,
                    "pending": pending,
                    "ocr_error": bool(m.get("ocr_error", False)),
                    "sharpness": sharp,
                    "stable_frames": int(m["stable"]),
                    "near_truck": False,
                }
            )

        out.sort(key=lambda r: r["bbox"][0])
        return out
