"""Headless no-parking engine — zone entry + stationary dwell before violation."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
from ultralytics import YOLO

import config
from modules.red_light.tracker import Tracker
from utils.logging_config import get_logger

log = get_logger("vl.no_parking")

_VEHICLE_CLASS_IDS = [2, 3, 5, 7]
_LABEL_TEXT = "WRONG PARKING"


def _zone_corners(zone: List[int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    x1, y1, x2, y2 = [int(v) for v in zone]
    corner_a = (min(x1, x2), min(y1, y2))
    corner_b = (max(x1, x2), max(y1, y2))
    return corner_a, corner_b


def _top_left_inside(x1: int, y1: int, corner_a: Tuple[int, int], corner_b: Tuple[int, int]) -> bool:
    return corner_a[0] < x1 < corner_b[0] and corner_a[1] < y1 < corner_b[1]


class NoParkingEngine:
    def __init__(self, model_path: Optional[str] = None) -> None:
        path = model_path or config.NO_PARKING_MODEL_PATH
        self._model = YOLO(path)
        self._tracker = Tracker(max_match_dist=90)
        self._violated_track_ids: set = set()
        self._dwell_state: Dict[int, Dict[str, float]] = {}
        self._min_stand_sec = float(getattr(config, "NO_PARKING_MIN_STAND_SEC", 5.0))
        self._max_move_px = int(getattr(config, "NO_PARKING_MAX_MOVE_PX", 15))
        log.info("Loaded no-parking model: %s", path)

    def reset(self) -> None:
        self._tracker = Tracker(max_match_dist=90)
        self._violated_track_ids = set()
        self._dwell_state = {}

    def process_frame(
        self,
        frame_bgr,
        zone: List[int],
        frame_idx: int,
        time_sec: float,
    ) -> Tuple[Any, List[Dict[str, Any]]]:
        events: List[Dict[str, Any]] = []
        if not zone or len(zone) != 4:
            return frame_bgr, events

        frame = frame_bgr if frame_bgr.flags.writeable else frame_bgr.copy()
        corner_a, corner_b = _zone_corners(zone)

        results = self._model(frame, device=getattr(config, "YOLO_DEVICE", "cpu"))
        cv2.rectangle(
            frame,
            (corner_a[0], corner_a[1]),
            (corner_b[0], corner_b[1]),
            (0, 0, 255),
            1,
        )

        detections: List[List[int]] = []
        in_zone_ids: set = set()

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                clss = int(box.cls[0].item())
                if clss not in _VEHICLE_CLASS_IDS:
                    continue
                w, h = x2 - x1, y2 - y1
                if w <= 0 or h <= 0:
                    continue
                detections.append([x1, y1, w, h])

        tracked = self._tracker.update(detections)

        for bbox in tracked:
            x1, y1, w, h, track_id = bbox
            x2, y2 = x1 + w, y1 + h
            tid = int(track_id)

            if not _top_left_inside(x1, y1, corner_a, corner_b):
                continue

            in_zone_ids.add(tid)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            state = self._dwell_state.get(tid)

            if state is None:
                self._dwell_state[tid] = {"start_sec": float(time_sec), "cx": float(cx), "cy": float(cy)}
                continue

            move = math.hypot(cx - state["cx"], cy - state["cy"])
            if move > self._max_move_px:
                self._dwell_state[tid] = {"start_sec": float(time_sec), "cx": float(cx), "cy": float(cy)}
                continue

            state["cx"] = float(cx)
            state["cy"] = float(cy)
            stood_for = float(time_sec) - float(state["start_sec"])
            if stood_for < self._min_stand_sec:
                continue

            ty = max(y1 - 10, 16)
            cv2.putText(
                frame,
                _LABEL_TEXT,
                (x1, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            if tid in self._violated_track_ids:
                continue
            self._violated_track_ids.add(tid)
            events.append(
                {
                    "violation_type": "no_parking",
                    "time_sec": time_sec,
                    "frame_index": frame_idx,
                    "summary": _LABEL_TEXT,
                    "zone": "no_parking",
                    "track_id": tid,
                    "bbox": [x1, y1, x2, y2],
                    "stood_sec": round(stood_for, 1),
                }
            )

        for tid in list(self._dwell_state.keys()):
            if tid not in in_zone_ids:
                del self._dwell_state[tid]

        return frame, events
