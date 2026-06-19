import cv2
from ultralytics import YOLO
import pandas as pd
import cvzone
import numpy as np
import os
import sys
import json
from datetime import datetime

import config
from modules.red_light.test1 import process_frame
from modules.red_light.tracker import Tracker
from modules.red_light.roi_selector import ROISelector
from modules.red_light.plate_reader import PlateReader, default_plate_weights, safe_filename_plate
from modules.red_light.dashboard import RuntimeDashboard

HEADLESS_DEBUG = os.environ.get("HEADLESS_DEBUG") == "1"
SHOW_DASHBOARD = os.environ.get("SHOW_DASHBOARD", "1") != "0"
HEADLESS_MAX_FRAMES = int(os.environ.get("HEADLESS_MAX_FRAMES", "250"))

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = YOLO(config.RED_LIGHT_MODEL_PATH)
    return _model


class _ModelProxy:
    def __call__(self, *args, **kwargs):
        return _get_model()(*args, **kwargs)


model = _ModelProxy()
video_path = "ty.mp4"
# Direction filter can block all events if set wrong; start with no filter.
# Options: "down", "up", "right", "left", "none"
TRACK_DIRECTION = os.environ.get("TRACK_DIRECTION", "none")
MIN_MOVE_PIXELS = int(os.environ.get("MIN_MOVE_PIXELS", "1"))
# Lower = more vehicle recalls (fewer missed detections)
VEHICLE_CONF = float(os.environ.get("VEHICLE_YOLO_CONF", "0.22"))
# Count as "in zone" if bbox overlaps ROI by this much OR centroid is inside
MIN_ROI_BOX_OVERLAP = float(os.environ.get("MIN_ROI_BOX_OVERLAP", "0.17"))
# If light flickers, still treat as red for a few frames after last RED
RED_LIGHT_GRACE_FRAMES = int(os.environ.get("RED_LIGHT_GRACE_FRAMES", "6"))
# Violation is recorded only after leaving the zone on the forward side (not while still inside).
VIOLATION_CLASSES = frozenset({"car", "truck", "bus", "motorcycle", "person"})

# Follow the traffic light in-frame (camera shake / motion) using YOLO + smoothed box
_sig_ema = None
TL_CONF = 0.22


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


def _pick_traffic_light_bbox(px, class_list, anchor, fw, fh):
    """Best traffic-light detection aligned with the user-drawn anchor ROI."""
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


def _clamp_xyxy(box, fw, fh):
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), fw - 1))
    y1 = max(0, min(int(y1), fh - 1))
    x2 = max(x1 + 1, min(int(x2), fw))
    y2 = max(y1 + 1, min(int(y2), fh))
    return [x1, y1, x2, y2]


def _smooth_signal_roi(new_box, anchor, fw, fh):
    """EMA on corners; when YOLO misses a frame, ease back toward the saved ROI."""
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
    """
    True if the object left the ROI on a "crossing" side (not a lateral nudge inside the band).
    For "down", count both bottom exit (cy > ry2) and top exit (cy < ry1) when prev_pt was
    inside vertically — wide ROIs / camera tilt often use the top edge as the far side.
    """
    if roi is None:
        return False
    rx1, ry1, rx2, ry2 = roi
    if direction == "down":
        if cy > ry2:
            return True
        if (
            prev_pt is not None
            and cy < ry1
            and prev_pt[1] > ry1
        ):
            return True
        return False
    if direction == "up":
        if cy < ry1:
            return True
        if (
            prev_pt is not None
            and cy > ry2
            and prev_pt[1] < ry2
        ):
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
    """Detect crossing of the near stop-line edge of ROI between frames."""
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


def _vehicle_box_in_roi_fraction(x1, y1, x2, y2, rois):
    """Fraction of vehicle bbox area overlapping violation ROI(s)."""
    box_area = max(1, (x2 - x1) * (y2 - y1))
    best = 0.0
    for roi in rois:
        rx1, ry1, rx2, ry2 = roi
        ix1 = max(x1, rx1)
        iy1 = max(y1, ry1)
        ix2 = min(x2, rx2)
        iy2 = min(y2, ry2)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        inter = (ix2 - ix1) * (iy2 - iy1)
        best = max(best, inter / float(box_area))
    return best

def _best_matching_roi(cx, cy, x1, y1, x2, y2, rois):
    """Pick ROI for this object (centroid-first, else max overlap)."""
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



def _save_violation(frame, obj_class, track_id, box_xyxy, output_dir, violated_ids, plate_reader):
    """Save one evidence frame and try plate OCR immediately at violation moment."""
    x3, y3, x4, y4 = box_xyxy
    if track_id in violated_ids:
        return False
    violated_ids.add(track_id)

    timestamp = datetime.now().strftime("%H-%M-%S-%f")[:-3]
    plate_tag = ""
    if plate_reader is not None:
        try:
            ptxt, pbox, _raw = plate_reader.read_from_vehicle_roi_robust(frame, (x3, y3, x4, y4))
            if ptxt and pbox:
                plate_tag = f"plate_{safe_filename_plate(ptxt)}_"
                px1, py1, px2, py2 = pbox
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 255), 2)
                cv2.putText(frame, ptxt, (px1, max(16, py1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        except Exception:
            pass

    image_filename = f"{obj_class}_ID{track_id}_{plate_tag}{timestamp}.jpg"
    output_path = os.path.join(output_dir, image_filename)
    cv2.imwrite(output_path, frame)
    return True




def run_cli_main():
    global TRACK_DIRECTION
    # Violation zones (vehicles inside + RED = violation), then one box around the traffic signal.
    if HEADLESS_DEBUG:
        with open("roi_config.json", "r", encoding="utf-8") as _rf:
            _cfg = json.load(_rf)
        violation_rois = list(_cfg.get("violation_rois") or _cfg.get("rois") or [])
        _sr = _cfg.get("signal_roi")
        signal_roi = list(_sr) if _sr and len(_sr) == 4 else None
        if not violation_rois or not signal_roi:
            print("HEADLESS_DEBUG: need violation_rois and signal_roi in roi_config.json")
            sys.exit(1)
    else:
        selector_v = ROISelector(video_path, "roi_config.json", mode="violation")
        violation_rois = selector_v.run()
        if not violation_rois:
            print("No violation ROI selected. Exiting...")
            raise SystemExit(0)

        selector_s = ROISelector(video_path, "roi_config.json", mode="signal")
        signal_result = selector_s.run()
        if not signal_result or len(signal_result) != 1:
            print("Traffic-signal ROI is required (draw one box, then S). Exiting...")
            raise SystemExit(0)
        signal_roi = signal_result[0]

    cap = cv2.VideoCapture(video_path)
    total_vid_frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)

    if HEADLESS_DEBUG:
        output_video = None
    else:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        output_video = cv2.VideoWriter("output_video.avi", fourcc, 20.0, (1020, 600))

    my_file = open("coco.txt", "r")
    data = my_file.read()
    class_list = data.split("\n")

    tracker = Tracker()

    _plate_w = default_plate_weights()
    plate_reader = None
    if _plate_w:
        try:
            _ocr_langs = [
                x.strip()
                for x in os.environ.get("PLATE_OCR_LANGS", "en").split(",")
                if x.strip()
            ] or ["en"]
            plate_reader = PlateReader(_plate_w, ocr_langs=_ocr_langs)
            print(f"Plate detection + EasyOCR enabled ({_plate_w}) langs={_ocr_langs}")
        except Exception as _e:
            print("Plate module disabled:", _e)

    # Create directory for today's date
    today_date = datetime.now().strftime('%Y-%m-%d')
    output_dir = os.path.join('saved_images', today_date)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    violated_ids = set()
    violation_total = 0
    violation_by_class = {}
    dash = RuntimeDashboard(600, 1020) if SHOW_DASHBOARD else None
    prev_centers = {}
    was_in_violation_roi = {}
    pending_red_crossing = {}
    last_violation_roi = {}
    _headless_frame_i = 0
    frame_idx = 0
    _last_red_frame = -10**9
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (1020, 600))
        fh, fw = frame.shape[0], frame.shape[1]
        frame_idx += 1

        results = model(
            frame,
            conf=VEHICLE_CONF,
            iou=0.5,
            verbose=False,
        )
        a = results[0].boxes.data.cpu()
        px = pd.DataFrame(a).astype("float")

        tl_box = _pick_traffic_light_bbox(px, class_list, signal_roi, fw, fh)
        current_signal_roi = _smooth_signal_roi(tl_box, signal_roi, fw, fh)
        processed_frame, detected_label = process_frame(
            frame,
            signal_roi=current_signal_roi,
            signal_label_on_frame=(dash is None),
        )
        if detected_label == "RED":
            _last_red_frame = frame_idx

        def _red_ok():
            if detected_label == "RED":
                return True
            if detected_label == "GREEN":
                return False
            return (frame_idx - _last_red_frame) <= RED_LIGHT_GRACE_FRAMES

        print(detected_label)

        detections = []
        det_meta = []
        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])

            d = int(row[5])
            c = class_list[d]
            # tracker expects [x, y, w, h]
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            detections.append([x1, y1, w, h])
            det_meta.append({"bbox": [x1, y1, x2, y2], "class": c})

        bbox_idx = tracker.update(detections)

        for bbox in bbox_idx:
            x3, y3, w, h, id = bbox
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
                    if det["class"] not in VIOLATION_CLASSES:
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
            prev_center = prev_centers.get(id)
            move_ok = True
            if prev_center is not None:
                dx = cx - prev_center[0]
                dy = cy - prev_center[1]
                if TRACK_DIRECTION == "down":
                    move_ok = dy > MIN_MOVE_PIXELS
                elif TRACK_DIRECTION == "up":
                    move_ok = dy < -MIN_MOVE_PIXELS
                elif TRACK_DIRECTION == "right":
                    move_ok = dx > MIN_MOVE_PIXELS
                elif TRACK_DIRECTION == "left":
                    move_ok = dx < -MIN_MOVE_PIXELS

            matched_roi = _best_matching_roi(
                cx, cy, x3, y3, x4, y4, violation_rois
            ) if violation_rois else None
            in_any_roi = matched_roi is not None

            is_target = obj_class in VIOLATION_CLASSES
            inside_zone = in_any_roi and is_target
            prev_in = was_in_violation_roi.get(id, False)

            if in_any_roi and not is_target:
                cvzone.putTextRect(frame, f"{obj_class} {id}", (x3, y3), 1, 1)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
            elif is_target:
                if inside_zone:
                    last_violation_roi[id] = matched_roi
                    crossed_now = bool(
                        prev_center is not None
                        and matched_roi is not None
                        and _crossed_stop_line(prev_center, (cx, cy), matched_roi, TRACK_DIRECTION)
                    )
                    if _red_ok() and crossed_now:
                        cvzone.putTextRect(frame, f"{obj_class} {id} VIOLATION", (x3, y3), 1, 1)
                        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
                        if _save_violation(
                            frame,
                            obj_class,
                            id,
                            (x3, y3, x4, y4),
                            output_dir,
                            violated_ids,
                            plate_reader,
                        ):
                            violation_total += 1
                            violation_by_class[obj_class] = (
                                violation_by_class.get(obj_class, 0) + 1
                            )
                        pending_red_crossing[id] = False
                    elif _red_ok():
                        if move_ok:
                            pending_red_crossing[id] = True
                            cvzone.putTextRect(
                                frame,
                                f"{obj_class} {id} APPROACH",
                                (x3, y3),
                                1,
                                1,
                            )
                            cv2.rectangle(
                                frame, (x3, y3), (x4, y4), (0, 165, 255), 2
                            )
                        else:
                            cvzone.putTextRect(
                                frame, f"{obj_class} {id} IGNORE", (x3, y3), 1, 1
                            )
                            cv2.rectangle(
                                frame, (x3, y3), (x4, y4), (255, 0, 0), 2
                            )
                    else:
                        pending_red_crossing[id] = False
                        cvzone.putTextRect(frame, f"{obj_class} {id} OK", (x3, y3), 1, 1)
                        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                else:
                    did_violation = False
                    if prev_center is not None and _red_ok():
                        ref_roi = last_violation_roi.get(id) or _roi_containing(
                            prev_center[0], prev_center[1], violation_rois
                        )
                        armed = bool(pending_red_crossing.get(id))
                        crossed_line = bool(
                            ref_roi
                            and _crossed_stop_line(
                                prev_center, (cx, cy), ref_roi, TRACK_DIRECTION
                            )
                        )
                        if (
                            ref_roi
                            and _exited_forward(
                                cx, cy, ref_roi, TRACK_DIRECTION, prev_center
                            )
                            and (armed or crossed_line or prev_in)
                        ):
                            cvzone.putTextRect(
                                frame,
                                f"{obj_class} {id} VIOLATION",
                                (x3, y3),
                                1,
                                1,
                            )
                            cv2.rectangle(
                                frame, (x3, y3), (x4, y4), (0, 0, 255), 2
                            )
                            if _save_violation(
                                frame,
                                obj_class,
                                id,
                                (x3, y3, x4, y4),
                                output_dir,
                                violated_ids,
                                plate_reader,
                            ):
                                violation_total += 1
                                violation_by_class[obj_class] = (
                                    violation_by_class.get(obj_class, 0) + 1
                                )
                            did_violation = True
                    if not did_violation:
                        cvzone.putTextRect(
                            frame, f"{obj_class} {id}", (x3, y3), 1, 1
                        )
                        cv2.rectangle(
                            frame, (x3, y3), (x4, y4), (0, 255, 0), 2
                        )
                    pending_red_crossing[id] = False
                    if id in last_violation_roi:
                        del last_violation_roi[id]
                was_in_violation_roi[id] = inside_zone
            else:
                cvzone.putTextRect(frame, f"{obj_class} {id}", (x3, y3), 1, 1)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)

            prev_centers[id] = (cx, cy)
                
        for idx, roi in enumerate(violation_rois):
            rx1, ry1, rx2, ry2 = roi
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)
            if dash is None:
                cv2.putText(
                    frame,
                    f"V{idx + 1}",
                    (rx1, max(20, ry1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )
        # Signal ROI outline + dots: process_frame. RED/GREEN text only if dashboard is off.

        if dash is None:
            cv2.putText(
                frame,
                f"Direction: {TRACK_DIRECTION.upper()}  [U/D/L/R, N none, Q quit]  conf={VEHICLE_CONF:.2f}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
            )

        if dash is not None:
            dash.tick_fps()
            dash.draw(
                frame,
                light_label=str(detected_label),
                frame_idx=frame_idx,
                total_frames=total_vid_frames,
                violation_total=violation_total,
                by_class=violation_by_class,
                n_tracks=len(bbox_idx),
                track_direction=TRACK_DIRECTION,
                vehicle_conf=VEHICLE_CONF,
                n_violation_rois=len(violation_rois),
                video_name=os.path.basename(video_path),
                plate_enabled=plate_reader is not None,
                red_grace_frames=RED_LIGHT_GRACE_FRAMES,
            )

        if output_video is not None:
            output_video.write(frame)

        if HEADLESS_DEBUG:
            _headless_frame_i += 1
            if _headless_frame_i >= HEADLESS_MAX_FRAMES:
                break
        else:
            cv2.imshow("RGB", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("u"), ord("U")):
                TRACK_DIRECTION = "up"
            elif key in (ord("d"), ord("D")):
                TRACK_DIRECTION = "down"
            elif key in (ord("l"), ord("L")):
                TRACK_DIRECTION = "left"
            elif key in (ord("r"), ord("R")):
                TRACK_DIRECTION = "right"
            elif key in (ord("n"), ord("N")):
                TRACK_DIRECTION = "none"

            if key == ord("q"):
                break

    cap.release()
    if output_video is not None:
        output_video.release()
    if not HEADLESS_DEBUG:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_cli_main()
