"""Shared frame-processing pipeline for CLI and dashboard apps."""

from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2

import config
from utils.detectors import MultiModelDetector, expand_bbox_xyxy
from utils.plate_ocr import (
    get_plate_reader,
    looks_like_indian_plate,
    ocr_plate_detections_one_shot,
    read_plate_from_crop,
)
from utils.plate_track_ocr import PlateOCRGate
from utils.tracker import CentroidTracker
from utils.events import merge_snapshots, normalize_engine_events
from utils.logging_config import get_logger
from utils.violations import (
    HELMET_VIOLATION_LABEL,
    TRIPLE_SEAT_VIOLATION_LABEL,
    ViolationManager,
    hour_in_half_open_window,
    infer_helmet_present_class_ids,
    infer_helmet_rider_class_ids,
    infer_helmet_violation_class_ids,
    infer_plate_like_class_ids_from_yolo_names,
    infer_truck_class_allowlist_from_yolo_names,
    infer_triple_class_allowlist_from_yolo_names,
    infer_triple_semantics_from_yolo_names,
)

_log = get_logger("vl.pipeline")
MODEL_DRAW_COLORS: Dict[str, tuple] = {
    "truck": (0, 255, 255),
    "triple": (255, 128, 0),
    "helmet": (255, 0, 255),
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
    locked = not bool(plate_read.get("pending"))
    # Green once the plate is identified/locked; cyan while still tracking/reading.
    color_outer = (0, 200, 0) if locked else (0, 255, 255)  # BGR
    color_inner = (180, 200, 255)
    thick = max(3, int(getattr(config, "THICKNESS", 2)) + 1)

    if inner is not None and draw_inner and pad > 0:
        ix1, iy1, ix2, iy2 = [int(v) for v in inner]
        cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), color_inner, 1, lineType=cv2.LINE_AA)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color_outer, thick, lineType=cv2.LINE_AA)

    yolo_c = float(plate_read.get("yolo_conf", 0.0))
    tid = int(plate_read.get("track_id", 0))
    # ASCII-only label (OpenCV's Hershey font renders non-ASCII like "·"/"…" as "?").
    if plate_read.get("ocr_error") and not plate_read.get("text"):
        line = f"PLATE #{tid} | YOLO {yolo_c:.2f} | OCR?"
    elif plate_read.get("pending"):
        line = (
            f"PLATE #{tid} | YOLO {yolo_c:.2f} | reading..."
            if plate_read.get("immediate_ocr")
            else f"PLATE #{tid} | YOLO {yolo_c:.2f} | tracking..."
        )
    else:
        txt = str(plate_read.get("text") or "?")[:18]
        ocf = float(plate_read.get("confidence", 0.0))
        line = f"PLATE #{tid} | {txt} | LOCKED {ocf:.2f}"

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


def _bbox_iou(a: List[int], b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = int(a[0]), int(a[1]), int(a[2]), int(a[3])
    bx1, by1, bx2, by2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    bb = max(0, bx2 - bx1) * max(0, by2 - by1)
    den = aa + bb - inter
    return inter / den if den > 0 else 0.0


def _bbox_centroid_dist(a: List[int], b: List[int]) -> float:
    acx, acy = _bbox_center(a)
    bcx, bcy = _bbox_center(b)
    dx = float(acx - bcx)
    dy = float(acy - bcy)
    return (dx * dx + dy * dy) ** 0.5


def _best_plate_for_bbox(subject_bbox: List[int], plate_reads: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Pick best plate by IoU first, else nearest centroid."""
    if not plate_reads:
        return None
    min_iou = float(getattr(config, "PLATE_VEHICLE_ASSOC_IOU_MIN", 0.08))
    max_dist = float(getattr(config, "PLATE_VEHICLE_ASSOC_MAX_CENTROID_DIST", 140.0))
    best_iou = None
    best_iou_v = 0.0
    for pr in plate_reads:
        iou = _bbox_iou(subject_bbox, pr["bbox"])
        if iou > best_iou_v:
            best_iou_v = iou
            best_iou = pr
    if best_iou is not None and best_iou_v >= min_iou:
        return best_iou
    best_dist = None
    best_pr = None
    for pr in plate_reads:
        d = _bbox_centroid_dist(subject_bbox, pr["bbox"])
        if best_dist is None or d < best_dist:
            best_dist = d
            best_pr = pr
    if best_pr is not None and best_dist is not None and best_dist <= max_dist:
        return best_pr
    return None


class TrafficPipeline:
    def __init__(
        self,
        model_paths: Optional[Dict[str, str]] = None,
        truck_violation_active_start_hour: Optional[int] = None,
        truck_violation_active_end_hour: Optional[int] = None,
        enabled_rules: Optional[Set[str]] = None,
        roi_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.enabled_rules: Set[str] = set(enabled_rules or [])
        self.roi_config: Dict[str, Any] = dict(roi_config or {})
        self._source_fps: float = float(getattr(config, "VIDEO_TARGET_PROCESS_FPS", 30) or 30)
        self._decode_stride: int = 1

        if model_paths is not None:
            paths = dict(model_paths)
        else:
            paths = dict(config.MODEL_PATHS)
            plate_path = config.PLATE_MODEL_PATH
            if config.is_model_file_usable(plate_path) and config.PLATE_MODEL_KEY not in paths:
                paths[config.PLATE_MODEL_KEY] = plate_path

        zone_only = bool(self.enabled_rules & {"red_light", "no_parking"})
        if not paths and not zone_only:
            raise ValueError("At least one model path or zone rule must be provided.")

        self.detector: Optional[MultiModelDetector] = None
        if paths:
            self.detector = MultiModelDetector(paths)
            self.active_models: Set[str] = set(self.detector.models.keys())
            if not self.active_models:
                raise RuntimeError("No YOLO models loaded. Check that selected .pt files exist and are valid.")
        else:
            self.active_models = set()

        self.use_truck = "truck" in self.active_models and (
            not self.enabled_rules or "truck_restricted" in self.enabled_rules
        )
        self.use_triple = "triple" in self.active_models and (
            not self.enabled_rules or "triple" in self.enabled_rules
        )
        self.use_helmet = "helmet" in self.active_models and (
            not self.enabled_rules or "helmet" in self.enabled_rules
        )
        self.use_plate = config.PLATE_MODEL_KEY in self.active_models and (
            not self.enabled_rules or "plate_ocr" in self.enabled_rules
        )

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
        if self.use_triple and self.detector and "triple" in self.detector.models:
            t_mdl = self.detector.models["triple"]
            t_names = getattr(t_mdl, "names", None)
            if len(getattr(config, "TRIPLE_VIOLATION_CLASS_IDS", [])) == 0:
                triple_semantics = infer_triple_semantics_from_yolo_names(t_names)
                if triple_semantics is None and bool(getattr(config, "TRIPLE_AUTO_CLASS_FILTER", True)):
                    triple_allow_from_model = infer_triple_class_allowlist_from_yolo_names(t_names)

        helmet_viol_ids: Optional[Set[int]] = None
        helmet_present_ids: Optional[Set[int]] = None
        helmet_rider_ids: Optional[Set[int]] = None
        if self.use_helmet and self.detector and "helmet" in self.detector.models:
            h_names = getattr(self.detector.models["helmet"], "names", None)
            cfg_h = getattr(config, "HELMET_VIOLATION_CLASS_IDS", []) or []
            if len(cfg_h) > 0:
                helmet_viol_ids = {int(x) for x in cfg_h}
            else:
                inferred = infer_helmet_violation_class_ids(h_names)
                if inferred:
                    helmet_viol_ids = inferred
            # Classes used by the absence rule (rider without a confident helmet).
            helmet_present_ids = infer_helmet_present_class_ids(h_names)
            helmet_rider_ids = infer_helmet_rider_class_ids(h_names)
            if not helmet_viol_ids and not (
                bool(getattr(config, "HELMET_ABSENCE_RULE", True)) and helmet_rider_ids
            ):
                print(
                    "[WARN] Helmet model loaded but no violation classes inferred and absence "
                    "rule unavailable; set HELMET_VIOLATION_CLASS_IDS in config.py (e.g. [2])."
                )

        # Cached for crop-based helmet detection (which class id means violation / present).
        self._helmet_viol_ids: Set[int] = set(helmet_viol_ids or [])
        self._helmet_present_ids: Set[int] = set(helmet_present_ids or [])

        self._violation_snapshot_seen: Set[Tuple[Any, ...]] = set()
        # Incidents already counted, so a violation that persists across frames keeps its
        # red box but is tallied only once (per truck track / rider cell / zone track).
        self._incident_seen: Set[Tuple[Any, ...]] = set()

        if bool(getattr(config, "TRUCK_RESTRICTED_MATCH_VIOLATION_WINDOW", True)):
            restricted_s, restricted_e = self.truck_viol_start, self.truck_viol_end
        else:
            restricted_s = int(getattr(config, "TRUCK_RESTRICTED_START_HOUR", 0))
            restricted_e = int(getattr(config, "TRUCK_RESTRICTED_END_HOUR", 24))

        self.violation_manager = ViolationManager(
            truck_restricted_start=restricted_s,
            truck_restricted_end=restricted_e,
            triple_class_allowlist=triple_allow_from_model,
            triple_semantics=triple_semantics,
            helmet_viol_class_ids=helmet_viol_ids,
            helmet_present_class_ids=helmet_present_ids,
            helmet_rider_class_ids=helmet_rider_ids,
        )
        self._ocr_reader = None
        # Crop-based helmet detection (carrier detector loaded lazily on first frame).
        self.use_helmet_crop = self.use_helmet and bool(
            getattr(config, "HELMET_CROP_DETECTION", True)
        )
        self._carrier_model = None
        # Cache stores (track_id|None, class_id, conf, bbox) per frame.
        self._carrier_cache: Optional[List[Tuple[Optional[int], int, float, List[int]]]] = None
        self._carrier_cache_key: int = -1
        self.vehicle_overlay = bool(getattr(config, "VEHICLE_OVERLAY", True))
        # Per-rider tracking so each rider's no-helmet violation is counted once.
        self._helmet_tracker = CentroidTracker(
            max_disappeared=int(getattr(config, "HELMET_TRACK_MAX_DISAPPEARED", 30)),
            max_distance=int(getattr(config, "HELMET_TRACK_MAX_DISTANCE", 120)),
        )
        self._helmet_track_state: Dict[int, Dict[str, Any]] = {}
        self._helmet_new_violations: List[List[int]] = []
        self._plate_gate = (
            PlateOCRGate()
            if self.use_plate and bool(getattr(config, "PLATE_USE_TRACK_OCR_GATE", True))
            else None
        )
        self._plate_simple_frame = 0
        self._plate_yolo_frame_counter = 0
        self._cached_plate_dets: List[dict] = []
        # When dedicated plate YOLO is on, drop plate-like classes from truck/triple/etc. (same boxes from weaker head).
        self._aux_plate_class_ids_by_model: Dict[str, Set[int]] = {}
        if self.use_plate and self.detector:
            pk = config.PLATE_MODEL_KEY
            for mname, mdl in self.detector.models.items():
                if mname == pk:
                    continue
                aux = infer_plate_like_class_ids_from_yolo_names(getattr(mdl, "names", None))
                if aux:
                    self._aux_plate_class_ids_by_model[mname] = aux

        self._truck_class_allowlist: Optional[Set[int]] = None
        if self.use_truck and self.detector and "truck" in self.detector.models:
            cfg_ids = getattr(config, "TRUCK_CLASS_IDS", None)
            if cfg_ids is not None and len(cfg_ids) > 0:
                self._truck_class_allowlist = {int(x) for x in cfg_ids}
            elif bool(getattr(config, "TRUCK_AUTO_CLASS_FILTER", True)):
                inferred = infer_truck_class_allowlist_from_yolo_names(
                    getattr(self.detector.models["truck"], "names", None)
                )
                if inferred is not None:
                    self._truck_class_allowlist = set(inferred)

        self._red_light_engine = None
        self._no_parking_engine = None
        self.models_loaded: List[str] = sorted(self.active_models)
        if "red_light" in self.enabled_rules:
            from modules.red_light.red_light_engine import RedLightPipelineEngine

            self._red_light_engine = RedLightPipelineEngine(self.roi_config)
            self.models_loaded.append("yolov10s")
        if "no_parking" in self.enabled_rules:
            from modules.no_parking.engine import NoParkingEngine

            self._no_parking_engine = NoParkingEngine()
            self.models_loaded.append("yolov8n")
        self.engines_active: List[str] = []
        if self.active_models:
            self.engines_active.append("lane")
        if self._red_light_engine:
            self.engines_active.append("red_light")
        if self._no_parking_engine:
            self.engines_active.append("no_parking")

    def configure_video_timing(self, fps: float, decode_stride: int = 1) -> None:
        """Set source video fps/stride so zone engines use real elapsed seconds."""
        self._source_fps = max(float(fps), 1e-6)
        self._decode_stride = max(1, int(decode_stride))
        _log.info("Pipeline ready models=%s engines=%s", self.models_loaded, self.engines_active)

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
        """Load EasyOCR weights once. Inference is used only inside ``read_plate_from_crop`` on plate crops, not on ``frame``."""
        if self._ocr_reader is not None:
            return self._ocr_reader
        self._ocr_reader = get_plate_reader(config.EASYOCR_LANGS)
        return self._ocr_reader

    def _plate_reads_direct_no_gate(self, frame, plate_dets: List[dict], truck_dets: List[dict]) -> List[Dict[str, Any]]:
        """
        Plate YOLO → crop → EasyOCR with no temporal tracker (see ``PLATE_USE_TRACK_OCR_GATE``).
        Throttled by ``PLATE_OCR_ATTEMPT_EVERY_N_FRAMES``; off-stride frames show boxes only (pending).
        """
        self._plate_simple_frame += 1
        stride = max(1, int(getattr(config, "PLATE_OCR_ATTEMPT_EVERY_N_FRAMES", 1)))
        truck_boxes = [d["bbox"] for d in truck_dets]

        if (self._plate_simple_frame % stride) != 0:
            out: List[Dict[str, Any]] = []
            for i, d in enumerate(plate_dets):
                x1, y1, x2, y2 = [int(x) for x in d["bbox"]]
                raw = d.get("bbox_raw")
                out.append(
                    {
                        "track_id": i,
                        "bbox": [x1, y1, x2, y2],
                        "bbox_raw": [int(x) for x in raw] if raw is not None else [x1, y1, x2, y2],
                        "text": "",
                        "confidence": 0.0,
                        "yolo_conf": float(d.get("confidence", 0.0)),
                        "pending": True,
                        "ocr_error": False,
                        "sharpness": 0.0,
                        "stable_frames": 0,
                        "near_truck": False,
                    }
                )
            return out

        min_y = float(getattr(config, "PLATE_OCR_MIN_YOLO_CONF", 0.5))
        shot = ocr_plate_detections_one_shot(
            frame,
            plate_dets,
            self._get_ocr_reader(),
            min_yolo_conf=min_y,
            truck_boxes=truck_boxes,
        )
        plate_reads: List[Dict[str, Any]] = []
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
                }
            )
        return plate_reads

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

    def _plate_text_in_region(
        self, frame_bgr: Any, region_bbox: List[int], plate_dets: List[dict]
    ) -> str:
        """Read the number plate that sits inside a violation region (if any).

        Only the plate sub-crop is sent to OCR -- never the whole violation image --
        so we can show the violation and its plate number combined on one card.
        """
        if frame_bgr is None or not plate_dets:
            return ""
        rx1, ry1, rx2, ry2 = [int(v) for v in region_bbox]
        best, best_area = None, 0
        for d in plate_dets:
            b = [int(v) for v in d["bbox"]]
            ix1, iy1 = max(rx1, b[0]), max(ry1, b[1])
            ix2, iy2 = min(rx2, b[2]), min(ry2, b[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            parea = max(1, (b[2] - b[0]) * (b[3] - b[1]))
            # Plate must be mostly inside the violation region.
            if inter / parea >= 0.5 and parea > best_area:
                best, best_area = d, parea
        if best is None:
            return ""
        crop = self._crop_bbox_bgr(frame_bgr, [int(v) for v in best["bbox"]])
        if crop is None:
            return ""
        try:
            txt, conf = read_plate_from_crop(self._get_ocr_reader(), crop)
        except Exception:
            return ""
        min_conf = float(getattr(config, "PLATE_OCR_DISPLAY_MIN_CONF", 0.0))
        if txt and (looks_like_indian_plate(txt) or float(conf) >= min_conf):
            return txt
        return ""

    def _collect_violation_snapshots(
        self,
        frame_bgr: Any,
        viol_raw: List[str],
        detections_for_rules: List[dict],
        triple_bbox_queue: "deque[List[int]]",
        helmet_bbox_queue: "deque[List[int]]",
        truck_bbox_tid: Optional[Dict[Tuple[int, int, int, int], Optional[int]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        One crop per *new* violation incident (deduped by rule + subject).
        Uses the same raw messages as violation checks (before string dedup).
        """
        out: List[Dict[str, Any]] = []
        plate_dets = (
            [d for d in detections_for_rules if d["model"] == config.PLATE_MODEL_KEY]
            if self.use_plate
            else []
        )
        truck_dets_sorted = sorted(
            [d for d in detections_for_rules if d["model"] == "truck"],
            key=lambda d: float(d["bbox"][0]),
        )
        ti = 0
        cell_px = max(16, int(getattr(config, "TRIPLE_STREAK_CELL_PX", 72)))
        helmet_cell_px = max(16, int(getattr(config, "HELMET_STREAK_CELL_PX", 64)))

        for msg in viol_raw:
            key: Optional[Tuple[Any, ...]] = None
            bbox: Optional[List[int]] = None

            if msg == "Truck in restricted hours":
                if ti < len(truck_dets_sorted):
                    bbox = list(truck_dets_sorted[ti]["bbox"])
                    b = bbox
                    tid = (truck_bbox_tid or {}).get(tuple(int(v) for v in b))
                    if tid is not None:
                        # Stable per-truck key: one evidence crop per physical truck.
                        key = ("truck_hours", "tid", int(tid))
                    else:
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
            elif msg == HELMET_VIOLATION_LABEL:
                if helmet_bbox_queue:
                    bbox = list(helmet_bbox_queue.popleft())
                    cx, cy = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
                    key = ("helmet", cx // helmet_cell_px, cy // helmet_cell_px)

            if key is None or bbox is None:
                continue
            if key in self._violation_snapshot_seen:
                continue
            crop_bgr = self._crop_bbox_bgr(frame_bgr, bbox)
            if crop_bgr is None:
                continue
            self._violation_snapshot_seen.add(key)
            rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            plate_txt = self._plate_text_in_region(frame_bgr, bbox, plate_dets)
            out.append({"message": msg, "bbox": list(bbox), "thumb_rgb": rgb, "plate": plate_txt})
        return out

    @classmethod
    def draw_detection(cls, frame, det):
        x1, y1, x2, y2 = det["bbox"]
        model_key = det["model"]
        color = det.get("color") or MODEL_DRAW_COLORS.get(model_key, DEFAULT_DRAW_COLOR)
        if det.get("label"):
            label = f"{det['label']} {det['confidence']:.2f}"
        elif model_key == "truck":
            # All truck sub-classes (dump / mixed / rmc / truck) shown as one "truck".
            label = f"truck {det['confidence']:.2f}"
        else:
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

    def _get_carrier_model(self):
        """Lazily load the shared COCO detector (person/vehicle) used by the crop-based
        helmet and plate passes. Returns the model or None if it cannot be loaded."""
        if self._carrier_model is None:
            try:
                from ultralytics import YOLO

                self._carrier_model = YOLO(getattr(config, "HELMET_CARRIER_MODEL_PATH"))
            except Exception as e:  # pragma: no cover - load failure path
                _log.warning("Carrier model load failed (%s); crop passes disabled.", e)
                self._carrier_model = False
        return self._carrier_model or None

    def _carrier_full(self, frame) -> List[Tuple[Optional[int], int, float, List[int]]]:
        """COCO detections (track id, class id, conf, bbox) for the current frame, cached
        so the helmet/plate crop passes and the vehicle overlay share one inference.

        Uses Ultralytics' built-in ByteTrack (``model.track(persist=True)``) when the
        vehicle overlay is on so each box carries a stable id; falls back to plain
        detection otherwise (no id)."""
        key = self._unified_frame_idx
        if self._carrier_cache_key == key and self._carrier_cache is not None:
            return self._carrier_cache
        out: List[Tuple[Optional[int], int, float, List[int]]] = []
        model = self._get_carrier_model()
        if model is not None:
            imgsz = int(getattr(config, "HELMET_CARRIER_IMGSZ", 960))
            device = getattr(config, "YOLO_DEVICE", "cpu")
            try:
                if self.vehicle_overlay:
                    res = model.track(
                        frame, persist=True, tracker="bytetrack.yaml",
                        verbose=False, imgsz=imgsz, device=device,
                    )[0]
                else:
                    res = model(frame, verbose=False, imgsz=imgsz, device=device)[0]
                for b in res.boxes or []:
                    tid = int(b.id[0]) if getattr(b, "id", None) is not None else None
                    out.append(
                        (tid, int(b.cls[0]), float(b.conf[0]), [int(v) for v in b.xyxy[0].tolist()])
                    )
            except Exception:
                out = []
        self._carrier_cache = out
        self._carrier_cache_key = key
        return out

    def _carrier_detections(self, frame) -> List[Tuple[int, float, List[int]]]:
        """COCO detections as (class id, confidence, bbox) — id-stripped view for the
        helmet and plate crop passes."""
        return [(c, cf, b) for (_id, c, cf, b) in self._carrier_full(frame)]

    def _draw_vehicle_overlay(self, frame) -> None:
        """Label each vehicle with its class + stable ByteTrack id."""
        names = getattr(config, "COCO_VEHICLE_NAMES", {})
        min_conf = float(getattr(config, "VEHICLE_OVERLAY_MIN_CONF", 0.35))
        for tid, cid, conf, (x1, y1, x2, y2) in self._carrier_full(frame):
            if cid not in names or conf < min_conf:
                continue
            label = f"{names[cid]} #{tid}" if tid is not None else names[cid]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
            cv2.putText(
                frame, label, (x1, max(y1 - 6, 16)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2,
            )

    def _plate_vehicle_boxes(self, frame) -> List[List[int]]:
        """Vehicle boxes (car/motorcycle/bus/truck) to scope plate detection to."""
        ids = {int(x) for x in getattr(config, "PLATE_VEHICLE_CLASS_IDS", [2, 3, 5, 7])}
        mc = float(getattr(config, "PLATE_VEHICLE_MIN_CONF", 0.30))
        return [box for cid, cf, box in self._carrier_detections(frame) if cid in ids and cf >= mc]

    def _helmet_crop_detections(self, frame, *, immediate: bool = False) -> List[dict]:
        """Crop-based, per-rider helmet decision.

        Helmet checkpoints are trained on close-up riders and are very sensitive to
        crop framing, so a single full-frame pass misses no-helmet riders. Instead we:
          1. Find riders (COCO person on/over a two-wheeler; bikes alone as fallback).
          2. Run the helmet model over a few crop variants of each rider's head region.
          3. A rider is "helmeted" only if a *confident* "With Helmet" box appears.
             Otherwise — explicit "Without Helmet", or no confident helmet at all
             (the model mislabels/loses bare & capped heads) — it is a violation.

        Returns synthetic helmet-model detections (violation / present) in full-frame
        coordinates, anchored on each rider's head box.
        """
        if self.detector is None or "helmet" not in self.detector.models:
            return []
        if not self._helmet_viol_ids and not self._helmet_present_ids:
            return []
        carrier_dets = self._carrier_detections(frame)
        if not carrier_dets and self._get_carrier_model() is None:
            self.use_helmet_crop = False
            return []

        hmodel = self.detector.models["helmet"]
        H, W = frame.shape[:2]
        carrier_ids = {int(x) for x in getattr(config, "HELMET_CARRIER_CLASS_IDS", [1, 3])}
        c_conf = float(getattr(config, "HELMET_CARRIER_MIN_CONF", 0.30))
        crop_imgsz = int(getattr(config, "HELMET_CROP_IMGSZ", 640))
        crop_conf = float(getattr(config, "HELMET_CROP_CONF", 0.20))
        present_th = float(getattr(config, "HELMET_CROP_PRESENT_CONF", 0.45))
        viol_th = float(getattr(config, "HELMET_CROP_VIOL_CONF", 0.25))
        absence_on = bool(getattr(config, "HELMET_CROP_ABSENCE", True))
        absence_conf = float(getattr(config, "HELMET_CROP_ABSENCE_CONF", 0.60))
        max_riders = int(getattr(config, "HELMET_CROP_MAX_RIDERS", 10))

        merge_iou = float(getattr(config, "HELMET_RIDER_MERGE_IOU", 0.4))

        def nms(boxes: List[List[int]], thr: float) -> List[List[int]]:
            """Greedy NMS keeping the largest box first (one physical subject = one box)."""
            order = sorted(
                boxes,
                key=lambda bx: (bx[2] - bx[0]) * (bx[3] - bx[1]),
                reverse=True,
            )
            kept: List[List[int]] = []
            for bx in order:
                if all(_bbox_iou(bx, k) < thr for k in kept):
                    kept.append(bx)
            return kept

        persons: List[List[int]] = []
        bikes: List[List[int]] = []
        for cid, cf, box in carrier_dets:
            if cf < c_conf:
                continue
            if cid == 0:
                persons.append(list(box))
            elif cid in carrier_ids:
                bikes.append(list(box))

        # Merge overlapping detections so one bike / one person is not split into several.
        bikes = nms(bikes, 0.5)
        persons = nms(persons, 0.6)

        # Build rider boxes: persons sitting on / over a two-wheeler.
        riders: List[List[int]] = []
        matched_bikes: Set[int] = set()
        for p in persons:
            pcx = (p[0] + p[2]) // 2
            for bi, bk in enumerate(bikes):
                # rider centre within bike span and feet near/over the bike top
                if bk[0] - 20 <= pcx <= bk[2] + 20 and p[3] >= bk[1] - 20:
                    riders.append(p)
                    matched_bikes.add(bi)
                    break
        # Bikes with no matched person: synthesise a rider region above the seat.
        for bi, bk in enumerate(bikes):
            if bi in matched_bikes:
                continue
            bh = max(1, bk[3] - bk[1])
            riders.append([bk[0], max(0, bk[1] - int(bh * 1.1)), bk[2], bk[1] + int(bh * 0.2)])

        # Final merge so a person box and an overlapping synthesised bike-rider collapse to one.
        riders = nms(riders, merge_iou)[:max_riders]
        out: List[dict] = []
        self._helmet_new_violations = []

        def run_crop(cx1: int, cy1: int, cx2: int, cy2: int):
            cx1, cy1 = max(0, cx1), max(0, cy1)
            cx2, cy2 = min(W, cx2), min(H, cy2)
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0 or crop.shape[0] < 12 or crop.shape[1] < 12:
                return 0.0, 0.0
            try:
                r = hmodel(
                    crop, verbose=False, imgsz=crop_imgsz, conf=crop_conf,
                    device=getattr(config, "YOLO_DEVICE", "cpu"),
                )[0]
            except Exception:
                return 0.0, 0.0
            withc = woc = 0.0
            if r.boxes is not None:
                for b in r.boxes:
                    cls_id = int(b.cls[0])
                    cf = float(b.conf[0])
                    if cls_id in self._helmet_present_ids:
                        withc = max(withc, cf)
                    elif cls_id in self._helmet_viol_ids:
                        woc = max(woc, cf)
            return withc, woc

        # First pass: per-frame raw helmet evidence for each rider.
        decisions: List[Dict[str, Any]] = []
        for r in riders:
            x1, y1, x2, y2 = r
            w, h = max(1, x2 - x1), max(1, y2 - y1)
            # Head box = top portion of the rider box (used for drawing / association).
            head = [x1, y1, x2, y1 + int(h * 0.5)]
            best_with = best_woc = 0.0
            # Ensemble of crop variants around the rider's upper body / head.
            for head_frac, pad in ((0.5, 0.05), (0.65, 0.18), (0.85, 0.3)):
                cw, cwo = run_crop(
                    int(x1 - w * pad),
                    int(y1 - h * 0.12),
                    int(x2 + w * pad),
                    y1 + int(h * head_frac),
                )
                best_with = max(best_with, cw)
                best_woc = max(best_woc, cwo)
            decisions.append(
                {"rider": [int(v) for v in r], "head": head, "w": best_with, "wo": best_woc}
            )

        # Track riders so we can smooth the decision over time. The helmet model is very
        # noisy frame-to-frame, so we keep an EMA of evidence per rider and apply
        # hysteresis: once a rider is judged helmeted / not, it only flips when the
        # opposite evidence is clearly stronger. This stops the box from flickering back
        # to a false positive a few frames after a correct call.
        rider_rects = [tuple(d["rider"]) for d in decisions]
        tracked = self._helmet_tracker.update(rider_rects)
        rect_to_id: Dict[Tuple[int, int, int, int], int] = {}
        for tid, rect in tracked.items():
            rect_to_id.setdefault(tuple(int(v) for v in rect), tid)
        live_ids = set(tracked.keys())
        self._helmet_track_state = {
            k: v for k, v in self._helmet_track_state.items() if k in live_ids
        }

        alpha = float(getattr(config, "HELMET_STATUS_EMA_ALPHA", 0.45))
        flip = float(getattr(config, "HELMET_STATUS_FLIP_MARGIN", 0.18))
        confirm_frames = 1 if immediate else max(1, int(getattr(config, "HELMET_CONFIRM_FRAMES", 3)))

        for d in decisions:
            head = d["head"]
            tid = rect_to_id.get(tuple(d["rider"]))
            if tid is None:
                continue
            new_track = tid not in self._helmet_track_state
            st = self._helmet_track_state.setdefault(
                tid, {"counted": False, "w": 0.0, "wo": 0.0, "status": "unknown", "n": 0}
            )
            st["n"] += 1
            # Smooth evidence over time (seed EMA with the first observation so the
            # warm-up frame is not artificially weak).
            if new_track:
                st["w"], st["wo"] = d["w"], d["wo"]
            else:
                st["w"] = alpha * d["w"] + (1 - alpha) * st["w"]
                st["wo"] = alpha * d["wo"] + (1 - alpha) * st["wo"]
            sw, swo = st["w"], st["wo"]

            # Candidate status from smoothed evidence.
            if swo >= viol_th and swo >= sw:
                cand = "no_helmet"
            elif sw >= present_th and sw >= swo:
                cand = "helmet"
            elif absence_on:
                cand = "no_helmet"
            else:
                cand = "unknown"

            # Hysteresis: do not abandon a settled status without clearly stronger evidence.
            prev = st["status"]
            if prev == "no_helmet" and cand != "no_helmet":
                if not (sw >= present_th + flip and sw > swo + flip):
                    cand = "no_helmet"
            elif prev == "helmet" and cand != "helmet":
                if not (swo >= viol_th + flip and swo > sw + flip):
                    cand = "helmet"
            st["status"] = cand

            if cand == "helmet" and self._helmet_present_ids:
                out.append(
                    {
                        "model": "helmet",
                        "class": next(iter(self._helmet_present_ids)),
                        "confidence": max(sw, present_th),
                        "bbox": head,
                        "label": "HELMET",
                        "color": (0, 180, 0),
                    }
                )
            elif cand == "no_helmet" and self._helmet_viol_ids:
                conf = max(swo, absence_conf, float(getattr(config, "HELMET_MIN_CONFIDENCE", 0.35)))
                out.append(
                    {
                        "model": "helmet",
                        "class": next(iter(self._helmet_viol_ids)),
                        "confidence": conf,
                        "bbox": head,
                        "label": "NO HELMET",
                        "color": (0, 0, 255),
                    }
                )
                # Only count once the rider has persisted long enough to rule out a
                # one-frame false positive.
                if not st["counted"] and st["n"] >= confirm_frames:
                    st["counted"] = True
                    self._helmet_new_violations.append(list(head))
        return out

    def process_frame(
        self,
        frame,
        *,
        force_immediate_plate_ocr: bool = False,
        reference_time: Optional[datetime] = None,
        force_full_frame_plate: bool = False,
    ):
        """
        One frame through the full stack (sequential, not OCR-in-background):

        1. YOLO violation heads (truck, triple, helmet, …) on ``frame``.
        2. YOLO plate detector on ``frame`` (or truck ROI), producing plate boxes.
        3. Violation rules from detections (triple, helmet, truck restricted-time).
        4. EasyOCR **only** on ``_safe_crop(frame, plate_bbox)`` when the plate gate allows — never on the whole frame.
        """
        self._unified_frame_idx = getattr(self, "_unified_frame_idx", 0) + 1
        frame_idx = self._unified_frame_idx - 1
        time_sec = (frame_idx * self._decode_stride) / self._source_fps

        now = self._now_for_truck_rules(reference_time)
        plate_key = config.PLATE_MODEL_KEY
        plate_every = max(1, int(getattr(config, "PLATE_YOLO_EVERY_N_FRAMES", 1)))
        skip_plate_yolo = False
        if self.use_plate and plate_every > 1:
            skip_plate_yolo = self._plate_yolo_frame_counter % plate_every != 0
            self._plate_yolo_frame_counter += 1
        elif self.use_plate:
            self._plate_yolo_frame_counter += 1

        # Violation / vehicle YOLO passes (all enabled models except plate).
        if self.detector is not None:
            skip: Set[str] = set()
            if self.use_plate:
                skip.add(plate_key)
            # In crop mode the helmet model is run per-rider-crop below, not full-frame.
            if self.use_helmet_crop:
                skip.add("helmet")
            detections = self.detector.infer(frame, skip_models=skip or None)
            detections = self._filter_truck_model_detections(detections)
            if self.use_helmet_crop:
                detections.extend(
                    self._helmet_crop_detections(
                        frame, immediate=bool(force_immediate_plate_ocr or force_full_frame_plate)
                    )
                )
        else:
            detections = []

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
                pad = float(
                    getattr(
                        config,
                        "TRUCK_PLATE_ROI_PAD_FRAC",
                        getattr(config, "TRUCK_ROI_PLATE_PAD_FRAC", 0.18),
                    )
                )
                veh_crop = bool(getattr(config, "PLATE_VEHICLE_CROP", True))
                vehicle_boxes = self._plate_vehicle_boxes(frame) if (veh_crop and not use_roi) else []
                if use_roi:
                    plate_infer_mode = "truck_roi"
                    plate_dets = self.detector.infer_plate(
                        frame,
                        truck_boxes,
                        use_truck_roi=True,
                        truck_roi_pad_frac=pad,
                        include_full_frame_when_roi=bool(
                            getattr(config, "PLATE_INCLUDE_FULL_FRAME_WITH_TRUCK_ROI", True)
                        ),
                    )
                elif vehicle_boxes:
                    # Scope plate YOLO to each vehicle crop (plate stays near native size).
                    plate_infer_mode = "vehicle_roi"
                    plate_dets = self.detector.infer_plate(
                        frame,
                        vehicle_boxes,
                        use_truck_roi=True,
                        truck_roi_pad_frac=float(
                            getattr(config, "PLATE_VEHICLE_CROP_PAD_FRAC", 0.10)
                        ),
                        include_full_frame_when_roi=bool(
                            force_full_frame_plate
                            or getattr(config, "PLATE_VEHICLE_INCLUDE_FULL_FRAME", False)
                        ),
                    )
                else:
                    plate_infer_mode = "full_frame"
                    plate_dets = self.detector.infer_plate(frame, [], use_truck_roi=False)
                for d in plate_dets:
                    d["bbox_raw"] = [int(x) for x in d["bbox"]]
                self._cached_plate_dets = [dict(d) for d in plate_dets]
            plate_dets = _expand_plate_detections(frame, plate_dets)
            other_dets = [d for d in detections if d["model"] != plate_key]
        else:
            plate_dets = []
            other_dets = detections

        # Plate localization is universal via plate.pt only; strip plate-like classes from truck/triple
        # so we do not double-count or run rules on non-plate-head boxes.
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

        if self.vehicle_overlay:
            self._draw_vehicle_overlay(frame)

        plate_reads: List[Dict[str, Any]] = []
        if self.use_plate and (
            self._plate_gate is not None or not bool(getattr(config, "PLATE_USE_TRACK_OCR_GATE", True))
        ):
            try:
                if force_immediate_plate_ocr:
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
                elif self._plate_gate is not None:
                    plate_reads = self._plate_gate.update(frame, plate_dets, self._get_ocr_reader)
                else:
                    plate_reads = self._plate_reads_direct_no_gate(frame, plate_dets, truck_dets)
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

        # Associate plate reads with nearest non-plate detections (IoU preferred, else centroid distance).
        vehicle_like_dets = [d for d in other_dets if d["model"] != plate_key]
        for pr in plate_reads:
            best_det = None
            best_iou = 0.0
            min_iou = float(getattr(config, "PLATE_VEHICLE_ASSOC_IOU_MIN", 0.08))
            max_dist = float(getattr(config, "PLATE_VEHICLE_ASSOC_MAX_CENTROID_DIST", 140.0))
            for d in vehicle_like_dets:
                iou = _bbox_iou(pr["bbox"], d["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_det = d
            if best_det is not None and best_iou >= min_iou:
                pr["assoc_vehicle_model"] = str(best_det["model"])
                pr["assoc_vehicle_bbox"] = list(best_det["bbox"])
                pr["assoc_method"] = "iou"
                continue
            nearest = None
            nearest_d = None
            for d in vehicle_like_dets:
                dd = _bbox_centroid_dist(pr["bbox"], d["bbox"])
                if nearest_d is None or dd < nearest_d:
                    nearest_d = dd
                    nearest = d
            if nearest is not None and nearest_d is not None and nearest_d <= max_dist:
                pr["assoc_vehicle_model"] = str(nearest["model"])
                pr["assoc_vehicle_bbox"] = list(nearest["bbox"])
                pr["assoc_method"] = "centroid"

        viol_raw: List[str] = []
        truck_rules_active = False
        truck_tracking_only = False

        triple_bbox_queue: deque = deque()
        t_pairs: List[Tuple[str, List[int]]] = []
        if self.use_triple:
            t_pairs = self.violation_manager.check_triple_riding_pairs(detections_for_rules)
            viol_raw.extend(m for m, _ in t_pairs)
            for _, b in t_pairs:
                triple_bbox_queue.append(b)

        helmet_bbox_queue: deque = deque()
        h_pairs: List[Tuple[str, List[int]]] = []
        if self.use_helmet:
            if self.use_helmet_crop:
                # Crop path already decided + tracked riders; each fires once per rider.
                h_pairs = [
                    (HELMET_VIOLATION_LABEL, list(b)) for b in self._helmet_new_violations
                ]
            else:
                h_pairs = self.violation_manager.check_helmet_violation_pairs(detections_for_rules)
            viol_raw.extend(m for m, _ in h_pairs)
            for _, b in h_pairs:
                helmet_bbox_queue.append(b)

        # Helmet(no-helmet) <-> rider association (IoU preferred, else centroid distance).
        helmet_rider_links: List[Dict[str, Any]] = []
        if h_pairs:
            rider_candidates = [
                d
                for d in other_dets
                if d.get("model") == "helmet"
                and int(d.get("class", -1)) not in set(self.violation_manager.helmet_viol_class_ids)
            ]
            hiou = float(getattr(config, "HELMET_RIDER_ASSOC_IOU_MIN", 0.08))
            hdist = float(getattr(config, "HELMET_RIDER_ASSOC_MAX_CENTROID_DIST", 120.0))
            for msg, hb in h_pairs:
                link: Dict[str, Any] = {"message": msg, "helmet_bbox": list(hb)}
                best = None
                best_iou = 0.0
                for rd in rider_candidates:
                    iou = _bbox_iou(hb, rd["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best = rd
                if best is not None and best_iou >= hiou:
                    link["rider_bbox"] = list(best["bbox"])
                    link["method"] = "iou"
                else:
                    nearest = None
                    nearest_d = None
                    for rd in rider_candidates:
                        dd = _bbox_centroid_dist(hb, rd["bbox"])
                        if nearest_d is None or dd < nearest_d:
                            nearest_d = dd
                            nearest = rd
                    if nearest is not None and nearest_d is not None and nearest_d <= hdist:
                        link["rider_bbox"] = list(nearest["bbox"])
                        link["method"] = "centroid"
                helmet_rider_links.append(link)

        if self.use_truck:
            truck_rules_active = self.truck_violations_time_active(now)
            truck_tracking_only = not truck_rules_active

            if truck_rules_active:
                viol_raw.extend(self.violation_manager.check_truck_restriction(detections_for_rules, now))
                cv2.putText(
                    frame,
                    f"Restricted hours: ON (window {self.truck_viol_start:02d}:00–{self.truck_viol_end:02d}:00)",
                    (20, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (0, 255, 100),
                    2,
                )
            else:
                cv2.putText(
                    frame,
                    f"Restricted hours: OFF (window {self.truck_viol_start:02d}:00–{self.truck_viol_end:02d}:00) — track + plate only",
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

        # Map each truck detection to its stable tracker id so a violation is counted
        # once per physical truck — not once per frame, nor per pixel-cell it drifts through.
        truck_bbox_tid: Dict[Tuple[int, int, int, int], Optional[int]] = {}
        if self.use_truck and tracked_objects:
            tracked_centroids = [
                (tid, ((tb[0] + tb[2]) / 2.0, (tb[1] + tb[3]) / 2.0))
                for tid, tb in tracked_objects.items()
            ]
            for d in truck_dets:
                b = d["bbox"]
                cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
                best_tid, best_dist = None, None
                for tid, (tcx, tcy) in tracked_centroids:
                    dist = (cx - tcx) ** 2 + (cy - tcy) ** 2
                    if best_dist is None or dist < best_dist:
                        best_dist, best_tid = dist, tid
                truck_bbox_tid[tuple(int(v) for v in b)] = best_tid

        # Tally each violation a single time per incident (per truck track / rider cell /
        # zone track). The bounding box still draws every frame; only the count fires once.
        new_violation_count = 0

        def _mark_incident(key: Tuple[Any, ...]) -> None:
            nonlocal new_violation_count
            if key not in self._incident_seen:
                self._incident_seen.add(key)
                new_violation_count += 1

        _cell_px = max(16, int(getattr(config, "TRIPLE_STREAK_CELL_PX", 72)))
        _hcell_px = max(16, int(getattr(config, "HELMET_STREAK_CELL_PX", 64)))
        if self.use_truck and truck_rules_active:
            for d in truck_dets:
                tid = truck_bbox_tid.get(tuple(int(v) for v in d["bbox"]))
                _mark_incident(("truck", tid) if tid is not None else ("truck", tuple(int(v) for v in d["bbox"])))
        for _m, _b in t_pairs:
            _mark_incident(("triple", ((_b[0] + _b[2]) // 2) // _cell_px, ((_b[1] + _b[3]) // 2) // _cell_px))
        for _m, _b in h_pairs:
            _mark_incident(("helmet", ((_b[0] + _b[2]) // 2) // _hcell_px, ((_b[1] + _b[3]) // 2) // _hcell_px))

        # Attach plate text to each violation when a nearby / overlapping plate is available.
        violation_lines: List[str] = []
        truck_sorted = sorted(truck_dets, key=lambda d: float(d["bbox"][0]))
        ti = 0
        for msg, bb in t_pairs:
            best_plate = _best_plate_for_bbox(bb, plate_reads)
            ptxt = str(best_plate.get("text") or "") if best_plate else ""
            violation_lines.append(f"{msg} | {ptxt}" if ptxt else msg)
        for msg, bb in h_pairs:
            best_plate = _best_plate_for_bbox(bb, plate_reads)
            ptxt = str(best_plate.get("text") or "") if best_plate else ""
            violation_lines.append(f"{msg} | {ptxt}" if ptxt else msg)
        for msg in viol_raw:
            if msg == "Truck in restricted hours":
                bb = list(truck_sorted[ti]["bbox"]) if ti < len(truck_sorted) else None
                ti += 1
                if bb is None:
                    violation_lines.append(msg)
                    continue
                best_plate = _best_plate_for_bbox(bb, plate_reads)
                ptxt = str(best_plate.get("text") or "") if best_plate else ""
                violation_lines.append(f"{msg} | {ptxt}" if ptxt else msg)
        violations = list(dict.fromkeys(violation_lines if violation_lines else viol_raw))
        violation_snapshots = self._collect_violation_snapshots(
            frame,
            viol_raw,
            detections_for_rules,
            triple_bbox_queue,
            helmet_bbox_queue,
            truck_bbox_tid=truck_bbox_tid,
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
            "helmet_rider_links": helmet_rider_links if self.use_helmet else [],
            "models_loaded": list(self.models_loaded),
            "engines_active": list(self.engines_active),
        }

        engine_events: List[Dict[str, Any]] = []
        if self._red_light_engine is not None:
            frame, rl_events = self._red_light_engine.process_frame(frame, frame_idx, time_sec)
            engine_events.extend(rl_events)
        if self._no_parking_engine is not None:
            zone = self.roi_config.get("no_parking_zone")
            if zone:
                frame, np_events = self._no_parking_engine.process_frame(
                    frame, list(zone), frame_idx, time_sec
                )
                engine_events.extend(np_events)

        if engine_events:
            zone_lines = normalize_engine_events(engine_events)
            violations = list(dict.fromkeys(list(violations or []) + zone_lines))
            meta["engine_events"] = engine_events
            for ev in engine_events:
                etid = ev.get("track_id")
                _mark_incident(
                    (
                        "engine",
                        ev.get("violation_type"),
                        ev.get("zone"),
                        int(etid) if etid is not None else tuple(ev.get("bbox") or []),
                    )
                )

        # Number of *new* violations this frame (already-flagged incidents excluded), so the
        # running total counts each violation once instead of every frame it stays visible.
        meta["new_violation_count"] = new_violation_count

        return frame, violations, meta
