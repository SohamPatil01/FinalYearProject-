"""Shared frame-processing pipeline for CLI and dashboard apps."""

from __future__ import annotations

import re
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2

import config
from utils.detectors import MultiModelDetector, expand_bbox_xyxy
from utils.plate_ocr import get_easyocr_reader
from utils.plate_track_ocr import PlateOCRGate
from utils.tracker import CentroidTracker
from utils.violations import (
    TRIPLE_SEAT_VIOLATION_LABEL,
    ViolationManager,
    detect_signal_state,
    hour_in_half_open_window,
    infer_plate_like_class_ids_from_yolo_names,
    infer_truck_class_allowlist_from_yolo_names,
    infer_triple_class_allowlist_from_yolo_names,
    infer_triple_semantics_from_yolo_names,
)
from utils.zones import get_zones

MODEL_DRAW_COLORS: Dict[str, tuple] = {
    "truck": (0, 255, 255),
    "triple": (255, 128, 0),
    "helmet": (255, 0, 255),
    "signal": (0, 255, 0),
    "no_parking": (180, 180, 255),
    "plate": (180, 220, 255),
}
DEFAULT_DRAW_COLOR = (200, 200, 200)


def _expand_plate_detections(frame, plate_dets: List[dict]) -> List[dict]:
    """
    Attach ``bbox_raw`` (YOLO) and set ``bbox`` to an expanded box for OCR / display.
    """
    if not plate_dets:
        return []
    pad = float(getattr(config, "PLATE_BBOX_EXPAND_FRAC", 0.0) or 0.0)
    h, w = int(frame.shape[0]), int(frame.shape[1])
    out: List[dict] = []
    for d in plate_dets:
        e = dict(d)
        raw = list(e.get("bbox_raw") or e["bbox"])
        e["bbox_raw"] = [int(x) for x in raw]
        if pad > 0:
            x1, y1, x2, y2 = expand_bbox_xyxy(raw, pad, h, w)
            e["bbox"] = [int(x1), int(y1), int(x2), int(y2)]
        else:
            e["bbox"] = list(e["bbox_raw"])
        out.append(e)
    return out


def _draw_plate_boxes_on_frame(frame, plate_read: dict) -> None:
    """Draw expanded plate ROI (thick) and optional inner YOLO box (thin)."""
    x1, y1, x2, y2 = [int(v) for v in plate_read["bbox"]]
    inner = plate_read.get("bbox_raw")
    draw_inner = bool(getattr(config, "PLATE_DRAW_INNER_YOLO_BOX", True))
    pad = float(getattr(config, "PLATE_BBOX_EXPAND_FRAC", 0.0) or 0.0)
    color_outer = (0, 255, 255)  # BGR cyan — high contrast
    color_inner = (180, 200, 255)
    thick = max(3, int(getattr(config, "THICKNESS", 2)) + 1)

    if inner is not None and draw_inner and pad > 0:
        ix1, iy1, ix2, iy2 = [int(v) for v in inner]
        cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), color_inner, 1, lineType=cv2.LINE_AA)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color_outer, thick, lineType=cv2.LINE_AA)

    yolo_c = float(plate_read.get("yolo_conf", 0.0))
    tid = int(plate_read.get("track_id", 0))
    if plate_read.get("ocr_error") and not plate_read.get("text"):
        line = f"PLATE #{tid} · YOLO {yolo_c:.2f} · OCR?"
    elif plate_read.get("pending"):
        line = (
            f"PLATE #{tid} · YOLO {yolo_c:.2f} · OCR…"
            if plate_read.get("immediate_ocr")
            else f"PLATE #{tid} · YOLO {yolo_c:.2f} · track…"
        )
    else:
        txt = str(plate_read.get("text") or "?")[:18]
        ocf = float(plate_read.get("confidence", 0.0))
        line = f"PLATE #{tid} · {txt} · OCR {ocf:.2f}"

    fs = 0.55
    (tw, th), bl = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
    ty = max(y1 - 8, th + 10)
    tx = min(x1, max(4, frame.shape[1] - tw - 6))
    cv2.rectangle(frame, (tx - 2, ty - th - 6), (tx + tw + 2, ty + bl), (20, 20, 20), -1)
    cv2.putText(frame, line, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, color_outer, 2, cv2.LINE_AA)


def _hour_in_truck_violation_window(hour: int, start_h: int, end_h: int) -> bool:
    """True when `hour` is inside the configured truck-rules window (half-open; overnight supported)."""
    return hour_in_half_open_window(hour, start_h, end_h)


def _bbox_center(bbox: List[int]) -> Tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) // 2, (y1 + y2) // 2


def _point_in_bbox(pt: Tuple[int, int], bbox: List[int]) -> bool:
    x, y = pt
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


class TrafficPipeline:
    def __init__(
        self,
        model_paths: Optional[Dict[str, str]] = None,
        truck_violation_active_start_hour: Optional[int] = None,
        truck_violation_active_end_hour: Optional[int] = None,
    ) -> None:
        # Caller-provided paths (e.g. dashboard): run only those models — no silent extras.
        # Default `model_paths=None` uses config.MODEL_PATHS and may add plate when weights exist (CLI).
        if model_paths is not None:
            paths = dict(model_paths)
        else:
            paths = dict(config.MODEL_PATHS)
            plate_path = config.PLATE_MODEL_PATH
            if config.is_model_file_usable(plate_path) and config.PLATE_MODEL_KEY not in paths:
                paths[config.PLATE_MODEL_KEY] = plate_path

        if not paths:
            raise ValueError("At least one model path must be provided.")

        self.detector = MultiModelDetector(paths)
        self.active_models: set[str] = set(self.detector.models.keys())
        if not self.active_models:
            raise RuntimeError("No YOLO models loaded. Check that selected .pt files exist and are valid.")

        self.use_truck = "truck" in self.active_models
        self.use_triple = "triple" in self.active_models
        self.use_plate = config.PLATE_MODEL_KEY in self.active_models

        self.truck_viol_start = (
            truck_violation_active_start_hour
            if truck_violation_active_start_hour is not None
            else config.TRUCK_VIOLATIONS_ACTIVE_START_HOUR
        )
        self.truck_viol_end = (
            truck_violation_active_end_hour
            if truck_violation_active_end_hour is not None
            else config.TRUCK_VIOLATIONS_ACTIVE_END_HOUR
        )

        self.tracker = CentroidTracker(
            max_disappeared=config.TRACKER_MAX_DISAPPEARED,
            max_distance=config.TRACKER_MAX_DISTANCE,
        )
        triple_semantics = None
        triple_allow_from_model: Optional[List[int]] = None
        if self.use_triple and "triple" in self.detector.models:
            t_mdl = self.detector.models["triple"]
            t_names = getattr(t_mdl, "names", None)
            if len(getattr(config, "TRIPLE_VIOLATION_CLASS_IDS", [])) == 0:
                triple_semantics = infer_triple_semantics_from_yolo_names(t_names)
                if triple_semantics is None and bool(getattr(config, "TRIPLE_AUTO_CLASS_FILTER", True)):
                    triple_allow_from_model = infer_triple_class_allowlist_from_yolo_names(t_names)

        self._violation_snapshot_seen: Set[Tuple[Any, ...]] = set()

        if bool(getattr(config, "TRUCK_RESTRICTED_MATCH_VIOLATION_WINDOW", True)):
            restricted_s, restricted_e = self.truck_viol_start, self.truck_viol_end
        else:
            restricted_s = int(getattr(config, "TRUCK_RESTRICTED_START_HOUR", 0))
            restricted_e = int(getattr(config, "TRUCK_RESTRICTED_END_HOUR", 24))

        self.violation_manager = ViolationManager(
            no_parking_threshold_sec=config.NO_PARKING_TIME_THRESHOLD_SEC,
            truck_restricted_start=restricted_s,
            truck_restricted_end=restricted_e,
            triple_class_allowlist=triple_allow_from_model,
            triple_semantics=triple_semantics,
        )
        self.zones = None
        self._ocr_reader = None
        self._plate_gate = PlateOCRGate() if self.use_plate else None
        self._plate_yolo_frame_counter = 0
        self._cached_plate_dets: List[dict] = []
        # When dedicated plate YOLO is on, drop plate-like classes from truck/triple/etc. (same boxes from weaker head).
        self._aux_plate_class_ids_by_model: Dict[str, Set[int]] = {}
        if self.use_plate:
            pk = config.PLATE_MODEL_KEY
            for mname, mdl in self.detector.models.items():
                if mname == pk:
                    continue
                aux = infer_plate_like_class_ids_from_yolo_names(getattr(mdl, "names", None))
                if aux:
                    self._aux_plate_class_ids_by_model[mname] = aux

        self._truck_class_allowlist: Optional[Set[int]] = None
        if self.use_truck and "truck" in self.detector.models:
            cfg_ids = getattr(config, "TRUCK_CLASS_IDS", None)
            if cfg_ids is not None and len(cfg_ids) > 0:
                self._truck_class_allowlist = {int(x) for x in cfg_ids}
            elif bool(getattr(config, "TRUCK_AUTO_CLASS_FILTER", True)):
                inferred = infer_truck_class_allowlist_from_yolo_names(
                    getattr(self.detector.models["truck"], "names", None)
                )
                if inferred is not None:
                    self._truck_class_allowlist = set(inferred)

    def _filter_truck_model_detections(self, dets: List[dict]) -> List[dict]:
        """Keep only truck-head boxes that match allowed class IDs and min confidence."""
        if not self.use_truck:
            return dets
        min_c = float(getattr(config, "TRUCK_YOLO_MIN_CONF", 0.0) or 0.0)
        allow = self._truck_class_allowlist
        out: List[dict] = []
        for d in dets:
            if d.get("model") != "truck":
                out.append(d)
                continue
            if min_c > 0 and float(d.get("confidence", 0.0)) < min_c:
                continue
            if allow is not None and int(d.get("class", -1)) not in allow:
                continue
            out.append(d)
        return out

    def _without_auxiliary_plate_detections(self, dets: List[dict]) -> List[dict]:
        if not self.use_plate or not self._aux_plate_class_ids_by_model:
            return dets
        mp = self._aux_plate_class_ids_by_model
        out: List[dict] = []
        for d in dets:
            m = d["model"]
            c = int(d.get("class", -1))
            if m in mp and c in mp[m]:
                continue
            out.append(d)
        return out

    def _get_ocr_reader(self):
        """Lazy EasyOCR reader for plate crops."""
        if self._ocr_reader is not None:
            return self._ocr_reader
        self._ocr_reader = get_easyocr_reader(config.EASYOCR_LANGS)
        return self._ocr_reader

    def _now_for_truck_rules(self, reference_time: Optional[datetime] = None) -> datetime:
        """Clock used for truck violation windows (optional IANA tz in config)."""
        if reference_time is not None:
            return reference_time
        tz_name = getattr(config, "TRUCK_RULES_TIMEZONE", None)
        if tz_name:
            try:
                from zoneinfo import ZoneInfo

                return datetime.now(ZoneInfo(str(tz_name)))
            except Exception:
                pass
        return datetime.now()

    def truck_violations_time_active(self, now: Optional[datetime] = None) -> bool:
        """True if rule clock hour is inside the configured truck-rules window."""
        t = now if now is not None else self._now_for_truck_rules()
        return _hour_in_truck_violation_window(t.hour, self.truck_viol_start, self.truck_viol_end)

    def _crop_bbox_bgr(self, frame_bgr: Any, bbox: List[int]) -> Optional[Any]:
        h, w = frame_bgr.shape[:2]
        pad = float(getattr(config, "VIOLATION_SNAPSHOT_PAD_FRAC", 0.12))
        x1, y1, x2, y2 = [int(v) for v in bbox]
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        px = max(2, int(bw * pad))
        py = max(2, int(bh * pad))
        x1 = max(0, x1 - px)
        y1 = max(0, y1 - py)
        x2 = min(w, x2 + px)
        y2 = min(h, y2 + py)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return crop

    def _collect_violation_snapshots(
        self,
        frame_bgr: Any,
        viol_raw: List[str],
        tracked_objects: Dict[int, Tuple[int, int, int, int]],
        detections_for_rules: List[dict],
        triple_bbox_queue: "deque[List[int]]",
    ) -> List[Dict[str, Any]]:
        """
        One crop per *new* violation incident (deduped by rule + subject).
        Uses the same raw messages as violation checks (before string dedup).
        """
        out: List[Dict[str, Any]] = []
        truck_dets_sorted = sorted(
            [d for d in detections_for_rules if d["model"] == "truck"],
            key=lambda d: float(d["bbox"][0]),
        )
        ti = 0
        cell_px = max(16, int(getattr(config, "TRIPLE_STREAK_CELL_PX", 72)))

        for msg in viol_raw:
            key: Optional[Tuple[Any, ...]] = None
            bbox: Optional[List[int]] = None

            if msg.startswith("No parking violation:"):
                m = re.search(r"ID (\d+)", msg)
                if m:
                    oid = int(m.group(1))
                    key = ("no_parking", oid)
                    t = tracked_objects.get(oid)
                    if t is not None:
                        bbox = list(t)
            elif msg.startswith("Signal jump"):
                m = re.search(r"ID (\d+)", msg)
                if m:
                    oid = int(m.group(1))
                    key = ("signal_red", oid)
                    t = tracked_objects.get(oid)
                    if t is not None:
                        bbox = list(t)
            elif msg == "Truck in restricted hours":
                if ti < len(truck_dets_sorted):
                    bbox = list(truck_dets_sorted[ti]["bbox"])
                    b = bbox
                    key = (
                        "truck_hours",
                        int(b[0]) // 16,
                        int(b[1]) // 16,
                        int(b[2]) // 16,
                        int(b[3]) // 16,
                    )
                    ti += 1
            elif msg in (TRIPLE_SEAT_VIOLATION_LABEL, "Triple riding detected"):
                if triple_bbox_queue:
                    bbox = list(triple_bbox_queue.popleft())
                    cx, cy = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
                    key = ("triple", cx // cell_px, cy // cell_px)

            if key is None or bbox is None:
                continue
            if key in self._violation_snapshot_seen:
                continue
            crop_bgr = self._crop_bbox_bgr(frame_bgr, bbox)
            if crop_bgr is None:
                continue
            self._violation_snapshot_seen.add(key)
            rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            out.append({"message": msg, "bbox": list(bbox), "thumb_rgb": rgb})
        return out

    @classmethod
    def draw_detection(cls, frame, det):
        x1, y1, x2, y2 = det["bbox"]
        model_key = det["model"]
        color = MODEL_DRAW_COLORS.get(model_key, DEFAULT_DRAW_COLOR)
        label = f"{model_key} | cls:{det['class']} | {det['confidence']:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, config.THICKNESS)
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            config.FONT_SCALE,
            color,
            config.THICKNESS,
        )

    def process_frame(
        self,
        frame,
        *,
        force_immediate_plate_ocr: bool = False,
        reference_time: Optional[datetime] = None,
        force_full_frame_plate: bool = False,
    ):
        if self.zones is None:
            self.zones = get_zones(frame.shape)

        now = self._now_for_truck_rules(reference_time)
        plate_key = config.PLATE_MODEL_KEY
        plate_every = max(1, int(getattr(config, "PLATE_YOLO_EVERY_N_FRAMES", 1)))
        skip_plate_yolo = False
        if self.use_plate and plate_every > 1:
            skip_plate_yolo = self._plate_yolo_frame_counter % plate_every != 0
            self._plate_yolo_frame_counter += 1
        elif self.use_plate:
            self._plate_yolo_frame_counter += 1

        # Plate YOLO runs via `infer_plate` (full frame or truck ROI); main `infer` skips the plate head.
        detections = self.detector.infer(
            frame,
            skip_models=({plate_key} if self.use_plate else None) or None,
        )
        detections = self._filter_truck_model_detections(detections)

        plate_infer_mode = "off"
        if self.use_plate:
            if skip_plate_yolo:
                plate_dets = [dict(d) for d in self._cached_plate_dets]
                plate_infer_mode = "cached"
            else:
                min_tc = float(getattr(config, "TRUCK_PLATE_MIN_TRUCK_CONF", 0.0))
                truck_boxes = [
                    d["bbox"]
                    for d in detections
                    if d["model"] == "truck" and float(d.get("confidence", 0.0)) >= min_tc
                ]
                scoped = bool(
                    getattr(
                        config,
                        "TRUCK_SCOPED_PLATE_ONLY",
                        getattr(config, "PLATE_DETECT_INSIDE_TRUCK_ROI", True),
                    )
                )
                # Still images (uploaded photos): always run plate YOLO on the full frame so a plate-only
                # shot or a plate outside the truck bottom strip is not missed when truck+plate are on.
                use_roi = (
                    (not force_full_frame_plate)
                    and self.use_truck
                    and scoped
                    and len(truck_boxes) > 0
                )
                plate_infer_mode = "truck_roi" if use_roi else "full_frame"
                pad = float(
                    getattr(
                        config,
                        "TRUCK_PLATE_ROI_PAD_FRAC",
                        getattr(config, "TRUCK_ROI_PLATE_PAD_FRAC", 0.18),
                    )
                )
                plate_dets = self.detector.infer_plate(
                    frame,
                    truck_boxes,
                    use_truck_roi=use_roi,
                    truck_roi_pad_frac=pad,
                )
                for d in plate_dets:
                    d["bbox_raw"] = [int(x) for x in d["bbox"]]
                self._cached_plate_dets = [dict(d) for d in plate_dets]
            plate_dets = _expand_plate_detections(frame, plate_dets)
            other_dets = [d for d in detections if d["model"] != plate_key]
        else:
            plate_dets = []
            other_dets = detections

        # Do not draw or feed rules with plate-class boxes from truck/triple heads when plate.pt is loaded.
        other_dets = self._without_auxiliary_plate_detections(other_dets)

        detections_for_rules = other_dets + plate_dets

        truck_dets = [d for d in other_dets if d["model"] == "truck"]

        if self.use_truck:
            truck_rects = [tuple(d["bbox"]) for d in truck_dets]
            tracked_objects = self.tracker.update(truck_rects)
            for object_id, bbox in tracked_objects.items():
                x1, y1, x2, y2 = bbox
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)
                cv2.putText(
                    frame,
                    f"ID {object_id}",
                    (x1, y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )
        else:
            tracked_objects = {}

        for det in other_dets:
            self.draw_detection(frame, det)

        plate_reads: List[Dict[str, Any]] = []
        if self.use_plate and self._plate_gate is not None:
            try:
                if force_immediate_plate_ocr:
                    from utils.plate_ocr import ocr_plate_detections_one_shot

                    min_y = float(getattr(config, "SAMPLE_OCR_MIN_YOLO_CONF", 0.35))
                    shot = ocr_plate_detections_one_shot(
                        frame,
                        plate_dets,
                        self._get_ocr_reader(),
                        min_yolo_conf=min_y,
                        truck_boxes=[d["bbox"] for d in truck_dets],
                    )
                    plate_reads = []
                    for i, row in enumerate(shot):
                        plate_reads.append(
                            {
                                "track_id": i,
                                "bbox": list(row["bbox"]),
                                "bbox_raw": list(row["bbox_raw"]) if row.get("bbox_raw") is not None else None,
                                "text": str(row.get("text") or ""),
                                "confidence": float(row.get("confidence") or 0.0),
                                "yolo_conf": float(row.get("yolo_conf") or 0.0),
                                "pending": not bool(row.get("text")),
                                "ocr_error": bool(row.get("ocr_error", False)),
                                "sharpness": 0.0,
                                "stable_frames": 0,
                                "near_truck": bool(row.get("near_truck", False)),
                                "immediate_ocr": True,
                            }
                        )
                else:
                    plate_reads = self._plate_gate.update(frame, plate_dets, self._get_ocr_reader)
            except Exception:
                plate_reads = []
                for d in plate_dets:
                    plate_reads.append(
                        {
                            "track_id": -1,
                            "text": "",
                            "confidence": 0.0,
                            "bbox": list(d["bbox"]),
                            "bbox_raw": list(d["bbox_raw"]) if d.get("bbox_raw") is not None else None,
                            "yolo_conf": float(d.get("confidence", 0.0)),
                            "pending": True,
                            "ocr_error": True,
                            "near_truck": False,
                        }
                    )

            for pr in plate_reads:
                _draw_plate_boxes_on_frame(frame, pr)
                pc = _bbox_center(pr["bbox"])
                for td in truck_dets:
                    if _point_in_bbox(pc, td["bbox"]):
                        pr["near_truck"] = True
                        break
                else:
                    pr["near_truck"] = False

        viol_raw: List[str] = []
        truck_rules_active = False
        truck_tracking_only = False

        triple_bbox_queue: deque = deque()
        if self.use_triple:
            t_pairs = self.violation_manager.check_triple_riding_pairs(detections_for_rules)
            viol_raw.extend(m for m, _ in t_pairs)
            for _, b in t_pairs:
                triple_bbox_queue.append(b)

        if self.use_truck:
            truck_rules_active = self.truck_violations_time_active(now)
            truck_tracking_only = not truck_rules_active

            if truck_rules_active:
                signal_state = detect_signal_state(frame, self.zones["signal_light"])
                viol_raw.extend(self.violation_manager.check_truck_restriction(detections_for_rules, now))
                viol_raw.extend(
                    self.violation_manager.update_no_parking(tracked_objects, self.zones["no_parking"], now)
                )
                viol_raw.extend(
                    self.violation_manager.check_signal_line_violation(
                        tracked_objects,
                        self.zones["signal_line"],
                        signal_state,
                    )
                )
                cv2.putText(
                    frame,
                    f"Signal: {signal_state} | Truck rules: ON",
                    (20, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 100),
                    2,
                )
            else:
                cv2.putText(
                    frame,
                    f"Truck rules: OFF (window {self.truck_viol_start:02d}:00–{self.truck_viol_end:02d}:00) — track + plate only",
                    (20, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (0, 220, 255),
                    2,
                )
        else:
            cv2.putText(
                frame,
                f"Models: {', '.join(sorted(m for m in self.active_models if m != plate_key))}",
                (20, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (220, 220, 220),
                2,
            )

        violations = list(dict.fromkeys(viol_raw))
        violation_snapshots = self._collect_violation_snapshots(
            frame, viol_raw, tracked_objects, detections_for_rules, triple_bbox_queue
        )

        y = 52 if self.use_truck else 50
        if violations:
            for message in violations[:6]:
                cv2.putText(
                    frame,
                    message,
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
                y += 24
        else:
            cv2.putText(
                frame,
                "No violations",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 0),
                2,
            )

        meta: Dict[str, Any] = {
            "plates": plate_reads,
            "truck_rules_active": self.use_truck and truck_rules_active,
            "truck_tracking_only": self.use_truck and truck_tracking_only,
            "truck_violation_window": (self.truck_viol_start, self.truck_viol_end),
            "truck_rules_clock_hour": now.hour,
            "truck_rules_tz": getattr(config, "TRUCK_RULES_TIMEZONE", None),
            "plate_infer_mode": plate_infer_mode,
            "plate_yolo_boxes": len(plate_dets),
            "violation_snapshots": violation_snapshots,
        }

        return frame, violations, meta
