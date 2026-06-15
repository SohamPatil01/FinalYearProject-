"""
License plate localization with a custom Ultralytics YOLO `.pt` model,
then character recognition with EasyOCR on the cropped plate region.
"""

from __future__ import annotations

import os
import re
from typing import Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


def _sanitize_plate_text(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", s).upper()


class PlateReader:
    def __init__(
        self,
        weights_path: str,
        det_conf: float = 0.25,
        ocr_langs: Optional[list] = None,
    ):
        if not weights_path or not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Plate model not found: {weights_path}")
        self.model = YOLO(weights_path)
        self.det_conf = det_conf
        self.ocr_langs = ocr_langs or ["en"]
        self._ocr_reader = None

    def _get_ocr(self):
        if self._ocr_reader is None:
            import easyocr

            self._ocr_reader = easyocr.Reader(self.ocr_langs, gpu=False)
        return self._ocr_reader

    def _ocr_plate_crop(self, plate_crop: np.ndarray):
        reader = self._get_ocr()
        variants = [plate_crop]
        g = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        variants.append(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))
        variants.append(cv2.cvtColor(cv2.equalizeHist(g), cv2.COLOR_GRAY2BGR))
        best_plate, best_raw = None, ""
        for v in variants:
            rgb = cv2.cvtColor(v, cv2.COLOR_BGR2RGB)
            lines = reader.readtext(rgb, detail=1, paragraph=False)
            raw_parts = [t for _b, t, _c in lines if t]
            raw = " ".join(raw_parts).strip()
            cand = _sanitize_plate_text(raw)
            if len(cand) >= 6 and (best_plate is None or len(cand) > len(best_plate)):
                best_plate, best_raw = cand, raw
        return best_plate, best_raw

    def read_from_vehicle_roi(
        self,
        frame_bgr: np.ndarray,
        vehicle_xyxy: Tuple[int, int, int, int],
    ) -> Tuple[Optional[str], Optional[Tuple[int, int, int, int]], str]:
        """
        Run plate detector on the vehicle crop, then EasyOCR on the best plate box.

        Returns:
            plate_text: alphanumeric string or None
            plate_xyxy: axis-aligned plate box in **full frame** coordinates, or None
            raw_ocr: concatenated OCR strings before sanitization (for debugging)
        """
        x1, y1, x2, y2 = map(int, vehicle_xyxy)
        h, w = frame_bgr.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(x1 + 1, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(y1 + 1, min(y2, h))
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None, None, ""

        res = self.model(crop, conf=self.det_conf, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return None, None, ""

        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        best = int(np.argmax(confs))
        px1, py1, px2, py2 = map(int, xyxy[best])

        ph, pw = crop.shape[:2]
        px1 = max(0, min(px1, pw - 1))
        px2 = max(px1 + 1, min(px2, pw))
        py1 = max(0, min(py1, ph - 1))
        py2 = max(py1 + 1, min(py2, ph))

        plate_crop = crop[py1:py2, px1:px2]
        if plate_crop.size == 0:
            return None, None, ""

        plate, raw = self._ocr_plate_crop(plate_crop)
        plate = plate or None

        gx1, gy1 = x1 + px1, y1 + py1
        gx2, gy2 = x1 + px2, y1 + py2
        return plate, (gx1, gy1, gx2, gy2), raw


def safe_filename_plate(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"[^A-Za-z0-9_-]", "", text)[:40]


def default_plate_weights() -> Optional[str]:
    root = os.path.dirname(os.path.abspath(__file__))
    env = os.environ.get("PLATE_MODEL_PATH")
    if env and os.path.isfile(env):
        return env
    local = os.path.join(root, "plate_best.pt")
    if os.path.isfile(local):
        return local
    return None


    def read_from_vehicle_roi_robust(
        self,
        frame_bgr: np.ndarray,
        vehicle_xyxy: Tuple[int, int, int, int],
    ) -> Tuple[Optional[str], Optional[Tuple[int, int, int, int]], str]:
        """Try original ROI and a slightly expanded ROI to improve plate recall."""
        plate, box, raw = self.read_from_vehicle_roi(frame_bgr, vehicle_xyxy)
        if plate:
            return plate, box, raw

        x1, y1, x2, y2 = map(int, vehicle_xyxy)
        h, w = frame_bgr.shape[:2]
        vw, vh = max(1, x2 - x1), max(1, y2 - y1)
        ex = int(0.12 * vw)
        ey_top = int(0.08 * vh)
        ey_bot = int(0.22 * vh)
        exp = (max(0, x1 - ex), max(0, y1 - ey_top), min(w, x2 + ex), min(h, y2 + ey_bot))
        return self.read_from_vehicle_roi(frame_bgr, exp)
