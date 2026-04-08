"""Violation checks for traffic events."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import config


def hour_in_half_open_window(hour: int, start_h: int, end_h: int) -> bool:
    """
    True if `hour` (0–23) lies in the half-open window [start_h, end_h).
    If start_h == end_h, treated as always active (full day).
    If start_h > end_h, window wraps midnight (e.g. 22–06). `end_h` may be 24 meaning “through 23:59”.
    """
    h = int(hour) % 24
    a, b = int(start_h), int(end_h)
    if a == b:
        return True
    if a < b:
        return a <= h < b
    bh = b % 24
    return h >= a or h < bh


def _iou(box_a, box_b) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union


def _point_in_bbox(pt: Tuple[int, int], bbox: List[int]) -> bool:
    x, y = pt
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


_TRIPLE_NAME_HINTS = (
    "triple",
    "3_rider",
    "3-rider",
    "three_rider",
    "three-rider",
    "3_seat",
    "3-seat",
    "three_seat",
    "3rider",
    "threerider",
)
_DOUBLE_NAME_HINTS = (
    "double",
    "2_rider",
    "2-rider",
    "two_rider",
    "two-rider",
    "2_seat",
    "2-seat",
    "two_seat",
    "duo",
    "2rider",
    "doublerider",
)


def _normalize_yolo_names(names) -> Dict[int, str]:
    """Ulalytics `model.names` as dict idx -> str, lowercased."""
    if names is None:
        return {}
    if isinstance(names, dict):
        return {int(k): str(v).lower() for k, v in names.items()}
    try:
        return {int(i): str(n).lower() for i, n in enumerate(names)}
    except (TypeError, ValueError):
        return {}


def _name_matches_any(name: str, hints: Tuple[str, ...]) -> bool:
    n = name.replace(" ", "_").replace("-", "_")
    return any(h.replace(" ", "_").replace("-", "_") in n for h in hints)


def _motorcycle_like_class_name(name: str) -> bool:
    n = name.lower().replace(" ", "_").replace("-", "_")
    return any(
        k in n
        for k in (
            "motorcycle",
            "motorbike",
            "scooter",
            "two_wheeler",
            "2_wheeler",
            "twowheeler",
        )
    )


def _carrier_vehicle_class_name(name: str) -> bool:
    """
    True for YOLO class names we treat as the *vehicle* when counting persons for triple-seat rules.
    Includes two-wheelers and common light multi-seat vehicles (rickshaw, etc.).
    """
    if _motorcycle_like_class_name(name):
        return True
    n = name.lower().replace(" ", "_").replace("-", "_")
    if n in ("auto", "autos"):
        return True
    if "rickshaw" in n or "tuk" in n or "tempo" in n:
        return True
    if "three_wheel" in n or "3_wheel" in n or "threewheel" in n:
        return True
    if "e_rickshaw" in n or "erickshaw" in n or "e_rick" in n:
        return True
    return False


# Shown in logs / UI when person count on one carrier reaches the threshold.
TRIPLE_SEAT_VIOLATION_LABEL = "Triple seat violation"
# Helmet checkpoint: default message when a no-helmet class fires (see `check_helmet_violation_pairs`).
HELMET_VIOLATION_LABEL = "No helmet violation"

_WITHOUT_HELMET_NAME_HINTS = (
    "no_helmet",
    "nohelmet",
    "no helmet",
    "without_helmet",
    "without helmet",
    "withouthelmet",
    "not_wearing",
    "not wearing",
    "bare_head",
    "bare head",
    "unhelmeted",
    "no_helm",
)


def _person_like_class_name(name: str) -> bool:
    n = name.lower().replace(" ", "_")
    if n in ("person", "people", "pedestrian", "rider", "human", "riders"):
        return True
    return n.endswith("_person") or n.endswith("_rider")


def _plate_like_class_name(name: str) -> bool:
    """True if this YOLO class name denotes a plate (usually an extra head on truck/triple, not ``plate.pt``)."""
    n = name.lower().replace(" ", "_").replace("-", "_")
    if n in ("plate", "plates", "numberplate", "lp"):
        return True
    if "numberplate" in n or "number_plate" in n:
        return True
    if ("license" in n or "licence" in n) and "plate" in n:
        return True
    if "registration" in n and "plate" in n:
        return True
    if n.endswith("_lp") or n.startswith("lp_"):
        return True
    return False


def _without_helmet_class_name(name: str) -> bool:
    """True if YOLO class name denotes a rider **without** a helmet (violation target)."""
    n = name.lower().replace(" ", "_").replace("-", "_")
    if _name_matches_any(n, _WITHOUT_HELMET_NAME_HINTS):
        return True
    return "no_helmet" in n or "without_helmet" in n or n.startswith("nohelm")


def infer_helmet_violation_class_ids(names) -> set[int]:
    """
    Class indices on the helmet checkpoint that should emit `HELMET_VIOLATION_LABEL`.

    Matches common labels such as ``no_helmet``, ``without helmet``, etc.
    """
    idx = _normalize_yolo_names(names)
    return {i for i, n in idx.items() if _without_helmet_class_name(n)}


def infer_plate_like_class_ids_from_yolo_names(names) -> set[int]:
    """Class indices whose names look like plates — stripped from truck/triple outputs when plate.pt is on.

    Plates are always detected via the dedicated plate model; these auxiliary plate classes are ignored
    so rules and overlays do not duplicate or conflict with ``plate.pt`` boxes.
    """
    idx = _normalize_yolo_names(names)
    return {i for i, n in idx.items() if _plate_like_class_name(n)}


def infer_triple_semantics_from_yolo_names(names) -> Optional[Dict[str, object]]:
    """
    If the triple checkpoint has separate **vehicle** + **person** classes (and optionally plate),
    return semantics to count how many persons sit on each vehicle; triple seat = count >= config threshold.
    """
    idx_to_name = _normalize_yolo_names(names)
    if len(idx_to_name) < 2:
        return None
    veh_ids = [i for i, n in idx_to_name.items() if _carrier_vehicle_class_name(n)]
    pr_ids = [i for i, n in idx_to_name.items() if _person_like_class_name(n)]
    if not veh_ids or not pr_ids:
        return None
    return {
        "mode": "vehicle_person",
        "vehicle_class_ids": sorted(set(veh_ids)),
        "person_class_ids": sorted(set(pr_ids)),
        # Backward compat for any code reading motorcycle_class_ids
        "motorcycle_class_ids": sorted(set(veh_ids)),
    }


def _heavy_truck_like_class_name(name: str) -> bool:
    """YOLO class names treated as the heavy / commercial vehicle for truck rules."""
    n = name.lower().replace(" ", "_").replace("-", "_")
    if n in ("truck", "trucks", "lorry", "lorries", "bus", "buses", "heavy_vehicle", "heavy_truck", "hgv"):
        return True
    if any(
        k in n
        for k in (
            "truck",
            "lorry",
            "tipper",
            "tanker",
            "trailer",
            "dumper",
            "pickup",
            "goods_vehicle",
            "commercial_vehicle",
            "transporter",
            "articulated",
            "semi_trailer",
            "semitrailer",
        )
    ):
        return True
    return False


def _exclude_from_truck_head_class_name(name: str) -> bool:
    """Classes from a multi-head 'vehicle' model that are not trucks (cars, autos, two-wheelers, etc.)."""
    n = name.lower().replace(" ", "_").replace("-", "_")
    if n in (
        "car",
        "cars",
        "sedan",
        "suv",
        "hatchback",
        "coupe",
        "wagon",
        "auto",
        "autos",
        "jeep",
        "van",
        "minivan",
    ):
        return True
    if "rickshaw" in n or "tuk" in n or "three_wheel" in n or "3_wheel" in n or "threewheel" in n:
        return True
    if "autorickshaw" in n or "auto_rickshaw" in n:
        return True
    if _motorcycle_like_class_name(name):
        return True
    if _person_like_class_name(name) or _plate_like_class_name(name):
        return True
    if n in ("bicycle", "cycle", "cyclist"):
        return True
    return False


def infer_truck_class_allowlist_from_yolo_names(names) -> Optional[List[int]]:
    """
    For a multi-class truck checkpoint, return class indices that should count as trucks.

    1) If any name matches heavy-vehicle hints, keep only those indices.
    2) Else drop indices whose names look like cars, autos, two-wheelers, etc.
    3) Single-class or unknown labels → None (no filtering — backward compatible).
    """
    idx = _normalize_yolo_names(names)
    if len(idx) <= 1:
        return None
    heavy_ids = [i for i, n in idx.items() if _heavy_truck_like_class_name(n)]
    if heavy_ids:
        return sorted(set(heavy_ids))
    neg_ids = {i for i, n in idx.items() if _exclude_from_truck_head_class_name(n)}
    if not neg_ids or neg_ids >= set(idx.keys()):
        return None
    keep = [i for i in idx if i not in neg_ids]
    return sorted(set(keep)) if keep else None


def infer_triple_class_allowlist_from_yolo_names(names) -> Optional[List[int]]:
    """
    If the triple model has multiple classes and names distinguish double vs triple, return only
    class indices that should trigger a violation. Returns None when unknown (single-class or
    ambiguous labels) — caller keeps all classes in that case.
    """
    idx_to_name = _normalize_yolo_names(names)
    if len(idx_to_name) <= 1:
        return None

    triple_ids = [i for i, n in idx_to_name.items() if _name_matches_any(n, _TRIPLE_NAME_HINTS)]
    double_ids = [i for i, n in idx_to_name.items() if _name_matches_any(n, _DOUBLE_NAME_HINTS)]

    if triple_ids:
        return sorted(set(triple_ids))
    # Only "double" (etc.) named — assume the other class is triple / violation target.
    if double_ids:
        allowed = [i for i in idx_to_name if i not in double_ids]
        if len(allowed) == 1:
            return allowed
    return None


def _nms_by_iou(dets: List[dict], iou_thresh: float) -> List[dict]:
    """Keep higher-confidence boxes; suppress others that overlap above iou_thresh."""
    if not dets:
        return []
    ordered = sorted(dets, key=lambda d: -float(d.get("confidence", 0.0)))
    kept: List[dict] = []
    for d in ordered:
        if any(_iou(d["bbox"], k["bbox"]) >= iou_thresh for k in kept):
            continue
        kept.append(d)
    return kept


class ViolationManager:
    """
    Rule-based checks: triple seat, helmet (no-helmet classes), truck restricted-time.
    """

    def __init__(
        self,
        truck_restricted_start: int,
        truck_restricted_end: int,
        triple_class_allowlist: Optional[List[int]] = None,
        triple_semantics: Optional[Dict[str, Any]] = None,
        helmet_viol_class_ids: Optional[Set[int]] = None,
    ) -> None:
        self.truck_restricted_start = truck_restricted_start
        self.truck_restricted_end = truck_restricted_end
        # From triple YOLO `.names` when TRIPLE_AUTO_CLASS_FILTER and no explicit config list
        self.triple_class_allowlist: Optional[List[int]] = triple_class_allowlist
        # motorcycle+person style triple head (see infer_triple_semantics_from_yolo_names)
        self.triple_semantics: Optional[Dict[str, Any]] = triple_semantics
        # Class IDs on helmet model that mean "no helmet" (empty set / None → no helmet rule)
        self.helmet_viol_class_ids: Set[int] = set(helmet_viol_class_ids or [])

        # (cell_x, cell_y) -> consecutive frames with a triple candidate in that cell
        self._triple_cell_streak: Dict[Tuple[int, int], int] = {}
        # Same for helmet violations (separate grid so triple and helmet do not share streak state)
        self._helmet_cell_streak: Dict[Tuple[int, int], int] = {}

    def _expand_bbox(self, bbox: List[int], frac: float) -> List[int]:
        x1, y1, x2, y2 = bbox
        w, h = max(1, x2 - x1), max(1, y2 - y1)
        dx, dy = int(w * frac), int(h * frac)
        return [x1 - dx, y1 - dy, x2 + dx, y2 + dy]

    def _person_on_vehicle(
        self, vehicle_bbox: List[int], person_bbox: List[int], expand_frac: float, iou_min: float
    ) -> bool:
        """True if this person detection is considered riding / seated on this vehicle box."""
        ex = self._expand_bbox(vehicle_bbox, expand_frac)
        pcx, pcy = (person_bbox[0] + person_bbox[2]) // 2, (person_bbox[1] + person_bbox[3]) // 2
        if _point_in_bbox((pcx, pcy), ex):
            return True
        return _iou(vehicle_bbox, person_bbox) >= iou_min

    def _check_triple_seat_vehicle_person(
        self,
        detections: List[dict],
        truck_boxes: List[List[int]],
        min_conf: float,
        min_streak: int,
        cell_px: int,
    ) -> List[Tuple[str, List[int]]]:
        """
        Triple-seat rule: count **person** boxes assigned to each **carrier vehicle** box.
        If any vehicle has >= TRIPLE_MIN_PERSONS_ON_MOTORCYCLE persons (default 3), emit one violation per vehicle (per cell / streak).
        """
        sem = self.triple_semantics or {}
        veh_classes = set(
            int(x) for x in (sem.get("vehicle_class_ids") or sem.get("motorcycle_class_ids") or [])
        )
        pr_classes = set(int(x) for x in sem.get("person_class_ids", []))
        need = max(1, int(getattr(config, "TRIPLE_MIN_PERSONS_ON_MOTORCYCLE", 3)))
        expand_frac = float(getattr(config, "TRIPLE_MC_EXPAND_FRAC", 0.2))
        iou_min = float(getattr(config, "TRIPLE_MC_PERSON_IOU_MIN", 0.04))
        merge_veh = float(getattr(config, "TRIPLE_MERGE_IOU", 0.5))
        merge_pr = float(getattr(config, "TRIPLE_PERSON_NMS_IOU", 0.42))

        veh_dets: List[dict] = []
        pr_dets: List[dict] = []
        for det in detections:
            if det["model"] != "triple":
                continue
            if float(det.get("confidence", 0.0)) < min_conf:
                continue
            cid = int(det.get("class", -1))
            if cid in veh_classes:
                veh_dets.append(det)
            elif cid in pr_classes:
                pr_dets.append(det)

        veh_dets = _nms_by_iou(veh_dets, merge_veh)
        pr_dets = _nms_by_iou(pr_dets, merge_pr)

        pairs: List[Tuple[str, List[int]]] = []
        violating_veh: List[dict] = []

        # Each person is assigned to at most one vehicle: the vehicle that best overlaps them.
        veh_counts: Dict[int, int] = {}
        for pr in pr_dets:
            best_i, best_v = -1, -1.0
            for i, veh in enumerate(veh_dets):
                vb = veh["bbox"]
                if any(_iou(vb, tb) > 0.25 for tb in truck_boxes):
                    continue
                if not self._person_on_vehicle(vb, pr["bbox"], expand_frac, iou_min):
                    continue
                v = _iou(vb, pr["bbox"])
                if v > best_v:
                    best_v, best_i = v, i
            if best_i >= 0:
                veh_counts[best_i] = veh_counts.get(best_i, 0) + 1

        for i, veh in enumerate(veh_dets):
            vb = veh["bbox"]
            if any(_iou(vb, tb) > 0.25 for tb in truck_boxes):
                continue
            if veh_counts.get(i, 0) >= need:
                violating_veh.append(veh)

        def _cell_key(bbox: List[int]) -> Tuple[int, int]:
            x1, y1, x2, y2 = bbox
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            return (cx // cell_px, cy // cell_px)

        cells_this_frame = {_cell_key(v["bbox"]) for v in violating_veh}
        next_streak: Dict[Tuple[int, int], int] = {}
        for ck in cells_this_frame:
            next_streak[ck] = self._triple_cell_streak.get(ck, 0) + 1
        self._triple_cell_streak = next_streak

        reported: set = set()
        for v in violating_veh:
            ck = _cell_key(v["bbox"])
            if ck in reported:
                continue
            if self._triple_cell_streak.get(ck, 0) >= min_streak:
                reported.add(ck)
                pairs.append((TRIPLE_SEAT_VIOLATION_LABEL, list(v["bbox"])))

        return pairs

    def check_triple_riding_pairs(self, detections: List[dict]) -> List[Tuple[str, List[int]]]:
        """
        Same as check_triple_riding, but each emitted violation includes the subject bbox (motorcycle / merged triple box).
        """
        truck_boxes = [det["bbox"] for det in detections if det["model"] == "truck"]

        min_conf = float(getattr(config, "TRIPLE_MIN_CONFIDENCE", 0.0))
        merge_iou = float(getattr(config, "TRIPLE_MERGE_IOU", 0.5))
        min_streak = max(1, int(getattr(config, "TRIPLE_MIN_CONSECUTIVE_FRAMES", 1)))
        cell_px = max(16, int(getattr(config, "TRIPLE_STREAK_CELL_PX", 72)))

        cfg_allow: Sequence[int] = getattr(config, "TRIPLE_VIOLATION_CLASS_IDS", [])
        mode = (self.triple_semantics or {}).get("mode")
        use_vehicle_person = (
            self.triple_semantics is not None
            and mode in ("vehicle_person", "motorcycle_person")
            and len(cfg_allow) == 0
        )
        if use_vehicle_person:
            return self._check_triple_seat_vehicle_person(
                detections, truck_boxes, min_conf, min_streak, cell_px
            )

        if len(cfg_allow) > 0:
            allow_classes: Optional[List[int]] = list(cfg_allow)
        elif self.triple_class_allowlist is not None and bool(
            getattr(config, "TRIPLE_AUTO_CLASS_FILTER", True)
        ):
            allow_classes = list(self.triple_class_allowlist)
        else:
            allow_classes = None
        use_class_filter = allow_classes is not None and len(allow_classes) > 0

        candidates: List[dict] = []
        for det in detections:
            if det["model"] != "triple":
                continue
            if float(det.get("confidence", 0.0)) < min_conf:
                continue
            if use_class_filter and int(det.get("class", -1)) not in allow_classes:
                continue
            overlaps_truck = any(_iou(det["bbox"], truck_box) > 0.25 for truck_box in truck_boxes)
            if overlaps_truck:
                continue
            candidates.append(det)

        merged = _nms_by_iou(candidates, merge_iou)

        def _cell_key(bbox: List[int]) -> Tuple[int, int]:
            x1, y1, x2, y2 = bbox
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            return (cx // cell_px, cy // cell_px)

        cells_this_frame = {_cell_key(d["bbox"]) for d in merged}
        next_streak: Dict[Tuple[int, int], int] = {}
        for ck in cells_this_frame:
            next_streak[ck] = self._triple_cell_streak.get(ck, 0) + 1
        self._triple_cell_streak = next_streak

        pairs: List[Tuple[str, List[int]]] = []
        reported: set = set()
        for det in merged:
            ck = _cell_key(det["bbox"])
            if ck in reported:
                continue
            if self._triple_cell_streak.get(ck, 0) >= min_streak:
                reported.add(ck)
                pairs.append((TRIPLE_SEAT_VIOLATION_LABEL, list(det["bbox"])))

        return pairs

    def check_triple_riding(self, detections: List[dict]) -> List[str]:
        return [m for m, _ in self.check_triple_riding_pairs(detections)]

    def check_helmet_violation_pairs(self, detections: List[dict]) -> List[Tuple[str, List[int]]]:
        """
        Flag **no-helmet** class detections from the helmet YOLO head.

        Other classes (``helmet``, ``motorcycle``, ``rider``, …) are drawn for context but only
        configured violation classes emit `HELMET_VIOLATION_LABEL`.
        """
        viol_ids = self.helmet_viol_class_ids
        if not viol_ids:
            return []

        min_conf = float(getattr(config, "HELMET_MIN_CONFIDENCE", 0.35))
        merge_iou = float(getattr(config, "HELMET_MERGE_IOU", 0.45))
        min_streak = max(1, int(getattr(config, "HELMET_MIN_CONSECUTIVE_FRAMES", 1)))
        cell_px = max(16, int(getattr(config, "HELMET_STREAK_CELL_PX", 64)))

        candidates: List[dict] = []
        for det in detections:
            if det.get("model") != "helmet":
                continue
            if float(det.get("confidence", 0.0)) < min_conf:
                continue
            if int(det.get("class", -1)) not in viol_ids:
                continue
            candidates.append(det)

        merged = _nms_by_iou(candidates, merge_iou)

        def _cell_key(bbox: List[int]) -> Tuple[int, int]:
            x1, y1, x2, y2 = bbox
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            return (cx // cell_px, cy // cell_px)

        cells_this_frame = {_cell_key(d["bbox"]) for d in merged}
        next_streak: Dict[Tuple[int, int], int] = {}
        for ck in cells_this_frame:
            next_streak[ck] = self._helmet_cell_streak.get(ck, 0) + 1
        self._helmet_cell_streak = next_streak

        pairs: List[Tuple[str, List[int]]] = []
        reported: set = set()
        for det in merged:
            ck = _cell_key(det["bbox"])
            if ck in reported:
                continue
            if self._helmet_cell_streak.get(ck, 0) >= min_streak:
                reported.add(ck)
                pairs.append((HELMET_VIOLATION_LABEL, list(det["bbox"])))

        return pairs

    def check_truck_restriction(self, detections: List[dict], now: datetime) -> List[str]:
        """
        Flag trucks when current time is inside the configured restricted window.
        Window uses the same half-open / overnight semantics as the pipeline violation schedule.
        """
        violations = []
        restricted_now = hour_in_half_open_window(now.hour, self.truck_restricted_start, self.truck_restricted_end)

        if not restricted_now:
            return violations

        for det in detections:
            if det["model"] == "truck":
                violations.append("Truck in restricted hours")
        return violations
