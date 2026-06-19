"""Red-light geometry helpers (from TrafficLight pmain1, no global YOLO)."""

from __future__ import annotations

import os
from datetime import datetime
from typing import List, Optional, Tuple

import config

_sig_ema: Optional[List[float]] = None

TRACK_DIRECTION = os.environ.get("TRACK_DIRECTION", getattr(config, "RED_LIGHT_TRACK_DIRECTION", "none"))
MIN_MOVE_PIXELS = int(os.environ.get("MIN_MOVE_PIXELS", getattr(config, "RED_LIGHT_MIN_MOVE_PIXELS", 1)))
VEHICLE_CONF = float(os.environ.get("VEHICLE_YOLO_CONF", getattr(config, "RED_LIGHT_VEHICLE_CONF", 0.22)))
MIN_ROI_BOX_OVERLAP = float(os.environ.get("MIN_ROI_BOX_OVERLAP", getattr(config, "RED_LIGHT_MIN_ROI_BOX_OVERLAP", 0.17)))
RED_LIGHT_GRACE_FRAMES = int(os.environ.get("RED_LIGHT_GRACE_FRAMES", getattr(config, "RED_LIGHT_GRACE_FRAMES", 6)))
VIOLATION_CLASSES = frozenset({"car", "truck", "bus", "motorcycle", "person"})
TL_CONF = 0.22


def reset_signal_ema() -> None:
    global _sig_ema
    _sig_ema = None


def get_sig_ema():
    return _sig_ema


def set_sig_ema(val) -> None:
    global _sig_ema
    _sig_ema = val


def _iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(1, (ax2 - ax1) * (ay2 - ay1))
    ba = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(aa + ba - inter)


def _clamp_xyxy(box, fw, fh):
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), fw - 1))
    y1 = max(0, min(int(y1), fh - 1))
    x2 = max(x1 + 1, min(int(x2), fw))
    y2 = max(y1 + 1, min(int(y2), fh))
    return [x1, y1, x2, y2]


def _pick_traffic_light_bbox(px, class_list, anchor, fw, fh):
    ax1, ay1, ax2, ay2 = anchor
    best_box = None
    best_score = -1.0
    for _, row in px.iterrows():
        if float(row[4]) < TL_CONF:
            continue
        di = int(row[5])
        if di < 0 or di >= len(class_list) or class_list[di] != "traffic light":
            continue
        x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
        if x2 <= x1 or y2 <= y1:
            continue
        iou = _iou_xyxy((ax1, ay1, ax2, ay2), (x1, y1, x2, y2))
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        inside = ax1 <= cx <= ax2 and ay1 <= cy <= ay2
        score = iou * 2.0 + (0.35 if inside else 0.0) + float(row[4]) * 0.15
        if score > best_score:
            best_score = score
            best_box = (x1, y1, x2, y2)
    if best_box is None:
        return None
    x1, y1, x2, y2 = best_box
    w, h = x2 - x1, y2 - y1
    pad = int(0.14 * max(w, h) + 5)
    return [
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(fw, x2 + pad),
        min(fh, y2 + pad),
    ]


def _smooth_signal_roi(new_box, anchor, fw, fh):
    global _sig_ema
    a = list(anchor)
    if new_box is None:
        if _sig_ema is None:
            return _clamp_xyxy(a, fw, fh)
        beta = 0.12
        for i in range(4):
            _sig_ema[i] = beta * a[i] + (1.0 - beta) * _sig_ema[i]
        return _clamp_xyxy([int(round(x)) for x in _sig_ema], fw, fh)
    nb = [float(x) for x in new_box]
    if _sig_ema is None:
        _sig_ema = nb[:]
    else:
        alpha = 0.55
        for i in range(4):
            _sig_ema[i] = alpha * nb[i] + (1.0 - alpha) * _sig_ema[i]
    return _clamp_xyxy([int(round(x)) for x in _sig_ema], fw, fh)


def _roi_containing(px, py, rois):
    for roi in rois:
        rx1, ry1, rx2, ry2 = roi
        if rx1 <= px <= rx2 and ry1 <= py <= ry2:
            return roi
    return None


def _exited_forward(cx, cy, roi, direction, prev_pt=None):
    if roi is None:
        return False
    rx1, ry1, rx2, ry2 = roi
    if direction == "down":
        if cy > ry2:
            return True
        if prev_pt is not None and cy < ry1 and prev_pt[1] > ry1:
            return True
        return False
    if direction == "up":
        if cy < ry1:
            return True
        if prev_pt is not None and cy > ry2 and prev_pt[1] < ry2:
            return True
        return False
    if direction == "right":
        if cx > rx2:
            return True
        if prev_pt is not None and cx < rx1 and prev_pt[0] > rx1:
            return True
        return False
    if direction == "left":
        if cx < rx1:
            return True
        if prev_pt is not None and cx > rx2 and prev_pt[0] < rx2:
            return True
        return False
    return True


def _crossed_stop_line(prev_pt, curr_pt, roi, direction):
    if prev_pt is None or curr_pt is None or roi is None:
        return False
    px, py = prev_pt
    cx, cy = curr_pt
    rx1, ry1, rx2, ry2 = roi
    if direction == "down":
        return px >= rx1 and px <= rx2 and ((py < ry1 <= cy) or (py > ry1 >= cy))
    if direction == "up":
        return px >= rx1 and px <= rx2 and ((py > ry2 >= cy) or (py < ry2 <= cy))
    if direction == "right":
        return py >= ry1 and py <= ry2 and ((px < rx1 <= cx) or (px > rx1 >= cx))
    if direction == "left":
        return py >= ry1 and py <= ry2 and ((px > rx2 >= cx) or (px < rx2 <= cx))
    return False


def _best_matching_roi(cx, cy, x1, y1, x2, y2, rois):
    for roi in rois:
        rx1, ry1, rx2, ry2 = roi
        if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
            return roi
    box_area = max(1, (x2 - x1) * (y2 - y1))
    best_roi = None
    best_frac = 0.0
    for roi in rois:
        rx1, ry1, rx2, ry2 = roi
        ix1 = max(x1, rx1)
        iy1 = max(y1, ry1)
        ix2 = min(x2, rx2)
        iy2 = min(y2, ry2)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        frac = ((ix2 - ix1) * (iy2 - iy1)) / float(box_area)
        if frac > best_frac:
            best_frac = frac
            best_roi = roi
    if best_frac >= MIN_ROI_BOX_OVERLAP:
        return best_roi
    return None


def save_violation_evidence(frame, obj_class, track_id, box_xyxy, output_dir, violated_ids) -> bool:
    import cv2

    x3, y3, x4, y4 = box_xyxy
    if track_id in violated_ids:
        return False
    violated_ids.add(track_id)
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%H-%M-%S-%f")[:-3]
    image_filename = f"{obj_class}_ID{track_id}_{timestamp}.jpg"
    output_path = os.path.join(output_dir, image_filename)
    cv2.imwrite(output_path, frame)
    return True
