"""Headless red-light processor — same per-frame logic as pmain1.py / original red_light_engine."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import cvzone
import pandas as pd
from ultralytics import YOLO

import config
from modules.red_light import pmain1 as pm
from modules.red_light.test1 import process_frame as tf_process_frame
from utils.logging_config import get_logger

log = get_logger("vl.red_light")


class RedLightPipelineEngine:
    """Single-frame red-light violation detection (shares model/helpers with pmain1)."""

    def __init__(
        self,
        roi_config: dict,
        model_path: Optional[str] = None,
        coco_path: Optional[str] = None,
        evidence_dir: Optional[str] = None,
    ) -> None:
        self.violation_rois = list(
            roi_config.get("violation_rois") or roi_config.get("rois") or []
        )
        sr = roi_config.get("signal_roi")
        self.signal_roi = list(sr) if sr and len(sr) == 4 else None
        if not self.violation_rois or not self.signal_roi:
            raise ValueError("roi_config must define violation_rois and signal_roi")

        coco_file = Path(coco_path or config.RED_LIGHT_CONFIG_DIR / "coco.txt")
        self.class_list = coco_file.read_text(encoding="utf-8").split("\n")

        pm._model = None
        if model_path:
            pm._model = YOLO(model_path)
        else:
            pm._get_model()
        log.info("Loaded red-light model via pmain1")

        today = datetime.now().strftime("%Y-%m-%d")
        base = evidence_dir or str(config.BASE_DIR / "saved_images" / today)
        self.output_dir = base
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        self._plate_w = pm.default_plate_weights()
        self.plate_reader = None
        if self._plate_w:
            try:
                self.plate_reader = pm.PlateReader(self._plate_w, ocr_langs=["en"])
            except Exception:
                self.plate_reader = None

        self.reset()

    def reset(self) -> None:
        self.violated_ids: set = set()
        self.prev_centers: dict = {}
        self.was_in_violation_roi: dict = {}
        self.pending_red_crossing: dict = {}
        self.last_violation_roi: dict = {}
        self._last_red_frame = -10**9
        self._sig_ema = None
        pm._sig_ema = None
        self.tracker = pm.Tracker()

    def process_frame(
        self, frame_bgr, frame_idx: int, time_sec: float
    ) -> Tuple[Any, List[Dict[str, Any]]]:
        pm._sig_ema = self._sig_ema
        events: List[Dict[str, Any]] = []
        src_h, src_w = frame_bgr.shape[:2]
        frame = cv2.resize(frame_bgr, (1020, 600))
        fh, fw = frame.shape[0], frame.shape[1]

        results = pm.model(
            frame, conf=pm.VEHICLE_CONF, iou=0.5, verbose=False,
            device=getattr(config, "YOLO_DEVICE", "cpu"),
        )
        a = results[0].boxes.data.cpu()
        px = pd.DataFrame(a).astype("float")

        tl_box = pm._pick_traffic_light_bbox(px, self.class_list, self.signal_roi, fw, fh)
        current_signal_roi = pm._smooth_signal_roi(tl_box, self.signal_roi, fw, fh)
        processed_frame, detected_label = tf_process_frame(
            frame, signal_roi=current_signal_roi
        )
        frame = processed_frame
        if detected_label == "RED":
            self._last_red_frame = frame_idx

        def _red_ok():
            if detected_label == "RED":
                return True
            if detected_label == "GREEN":
                return False
            return (frame_idx - self._last_red_frame) <= pm.RED_LIGHT_GRACE_FRAMES

        detections = []
        det_meta = []
        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            d = int(row[5])
            c = self.class_list[d]
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            detections.append([x1, y1, w, h])
            det_meta.append({"bbox": [x1, y1, x2, y2], "class": c})

        bbox_idx = self.tracker.update(detections)

        for bbox in bbox_idx:
            x3, y3, w, h, oid = bbox
            x4 = x3 + w
            y4 = y3 + h
            cx = int(x3 + x4) // 2
            cy = int(y3 + y4) // 2

            obj_class = "vehicle"
            best_iou = 0.0
            for det in det_meta:
                dx1, dy1, dx2, dy2 = det["bbox"]
                ix1 = max(x3, dx1)
                iy1 = max(y3, dy1)
                ix2 = min(x4, dx2)
                iy2 = min(y4, dy2)
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                inter = (ix2 - ix1) * (iy2 - iy1)
                area1 = max(1, (x4 - x3) * (y4 - y3))
                area2 = max(1, (dx2 - dx1) * (dy2 - dy1))
                iou = inter / float(area1 + area2 - inter)
                if iou > best_iou:
                    best_iou = iou
                    obj_class = det["class"]

            if best_iou < 0.2:
                best_dist = 1e9
                best_cls = obj_class
                for det in det_meta:
                    if det["class"] not in pm.VIOLATION_CLASSES:
                        continue
                    dx1, dy1, dx2, dy2 = det["bbox"]
                    dcx = (dx1 + dx2) // 2
                    dcy = (dy1 + dy2) // 2
                    dist2 = (dcx - cx) ** 2 + (dcy - cy) ** 2
                    if dist2 < best_dist:
                        best_dist = dist2
                        best_cls = det["class"]
                if best_dist <= 165 * 165:
                    obj_class = best_cls

            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
            prev_center = self.prev_centers.get(oid)
            move_ok = True
            if prev_center is not None:
                dx = cx - prev_center[0]
                dy = cy - prev_center[1]
                if pm.TRACK_DIRECTION == "down":
                    move_ok = dy > pm.MIN_MOVE_PIXELS
                elif pm.TRACK_DIRECTION == "up":
                    move_ok = dy < -pm.MIN_MOVE_PIXELS
                elif pm.TRACK_DIRECTION == "right":
                    move_ok = dx > pm.MIN_MOVE_PIXELS
                elif pm.TRACK_DIRECTION == "left":
                    move_ok = dx < -pm.MIN_MOVE_PIXELS

            matched_roi = (
                pm._best_matching_roi(cx, cy, x3, y3, x4, y4, self.violation_rois)
                if self.violation_rois
                else None
            )
            in_any_roi = matched_roi is not None
            is_target = obj_class in pm.VIOLATION_CLASSES
            inside_zone = in_any_roi and is_target
            prev_in = self.was_in_violation_roi.get(oid, False)

            def save_and_event():
                ok = pm._save_violation(
                    frame,
                    obj_class,
                    oid,
                    (x3, y3, x4, y4),
                    self.output_dir,
                    self.violated_ids,
                    self.plate_reader,
                )
                if ok:
                    events.append(
                        {
                            "violation_type": "red_light",
                            "time_sec": time_sec,
                            "frame_index": frame_idx,
                            "summary": f"Red-light violation {obj_class} id={oid}",
                            "zone": "red_light",
                            "track_id": int(oid),
                            "bbox": [x3, y3, x4, y4],
                            "evidence_path": self.output_dir,
                        }
                    )

            if in_any_roi and not is_target:
                cvzone.putTextRect(frame, f"{obj_class} {oid}", (x3, y3), 1, 1)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
            elif is_target:
                if inside_zone:
                    self.last_violation_roi[oid] = matched_roi
                    crossed_now = bool(
                        prev_center is not None
                        and matched_roi is not None
                        and pm._crossed_stop_line(
                            prev_center, (cx, cy), matched_roi, pm.TRACK_DIRECTION
                        )
                    )
                    if _red_ok() and crossed_now:
                        cvzone.putTextRect(
                            frame, f"{obj_class} {oid} VIOLATION", (x3, y3), 1, 1
                        )
                        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
                        save_and_event()
                        self.pending_red_crossing[oid] = False
                    elif _red_ok():
                        if move_ok:
                            self.pending_red_crossing[oid] = True
                            cvzone.putTextRect(
                                frame, f"{obj_class} {oid} APPROACH", (x3, y3), 1, 1
                            )
                            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 165, 255), 2)
                        else:
                            cvzone.putTextRect(
                                frame, f"{obj_class} {oid} IGNORE", (x3, y3), 1, 1
                            )
                            cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 0), 2)
                    else:
                        self.pending_red_crossing[oid] = False
                        cvzone.putTextRect(frame, f"{obj_class} {oid} OK", (x3, y3), 1, 1)
                        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                else:
                    did_violation = False
                    if prev_center is not None and _red_ok():
                        ref_roi = self.last_violation_roi.get(oid) or pm._roi_containing(
                            prev_center[0], prev_center[1], self.violation_rois
                        )
                        armed = bool(self.pending_red_crossing.get(oid))
                        crossed_line = bool(
                            ref_roi
                            and pm._crossed_stop_line(
                                prev_center, (cx, cy), ref_roi, pm.TRACK_DIRECTION
                            )
                        )
                        if (
                            ref_roi
                            and pm._exited_forward(
                                cx, cy, ref_roi, pm.TRACK_DIRECTION, prev_center
                            )
                            and (armed or crossed_line or prev_in)
                        ):
                            cvzone.putTextRect(
                                frame, f"{obj_class} {oid} VIOLATION", (x3, y3), 1, 1
                            )
                            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
                            save_and_event()
                            did_violation = True
                    if not did_violation:
                        cvzone.putTextRect(frame, f"{obj_class} {oid}", (x3, y3), 1, 1)
                        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                    self.pending_red_crossing[oid] = False
                    if oid in self.last_violation_roi:
                        del self.last_violation_roi[oid]
                self.was_in_violation_roi[oid] = inside_zone
            else:
                cvzone.putTextRect(frame, f"{obj_class} {oid}", (x3, y3), 1, 1)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)

            self.prev_centers[oid] = (cx, cy)

        for idx, roi in enumerate(self.violation_rois):
            rx1, ry1, rx2, ry2 = roi
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)
            cv2.putText(
                frame,
                f"V{idx + 1}",
                (rx1, max(20, ry1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

        cv2.putText(
            frame,
            f"Red-light pipeline | {pm.TRACK_DIRECTION} | f={frame_idx}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

        self._sig_ema = pm._sig_ema

        if (fw, fh) != (src_w, src_h):
            frame_out = cv2.resize(frame, (src_w, src_h))
        else:
            frame_out = frame
        return frame_out, events
