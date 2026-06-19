"""CLI entry for OpenCV ROI selection subprocess."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from modules.red_light.roi_selector import ROISelector


def main() -> int:
    parser = argparse.ArgumentParser(description="VioLane ROI selector")
    parser.add_argument("--video", required=True)
    parser.add_argument("--mode", choices=["violation", "signal", "no_parking"], required=True)
    parser.add_argument("--out", required=True, help="Session ROI JSON path")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    session: dict = {}
    if out_path.is_file():
        try:
            session = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            session = {}

    if args.mode == "no_parking":
        cap = cv2.VideoCapture(args.video)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            print("Could not read video frame", file=sys.stderr)
            return 1
        zone = [0, 0, 0, 0]
        drawing = False
        start_pt = (0, 0)

        def _mouse(event, x, y, flags, param):
            nonlocal drawing, start_pt, zone
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                start_pt = (x, y)
                zone = [x, y, x, y]
            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                zone[2], zone[3] = x, y
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                zone[2], zone[3] = x, y

        win = "No parking zone — drag rectangle, press S to save, Q to quit"
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, _mouse)
        while True:
            disp = frame.copy()
            x1, y1, x2, y2 = zone
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.imshow(win, disp)
            key = cv2.waitKey(20) & 0xFF
            if key == ord("q"):
                cv2.destroyAllWindows()
                return 2
            if key == ord("s"):
                if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                    session["no_parking_zone"] = [x1, y1, x2, y2]
                    out_path.write_text(json.dumps(session, indent=2), encoding="utf-8")
                    cv2.destroyAllWindows()
                    return 0
        return 1

    selector = ROISelector(args.video, str(out_path), mode=args.mode)
    result = selector.run()
    if not result:
        return 2
    if args.mode == "signal":
        if len(result) != 1:
            return 1
        session["signal_roi"] = result[0]
    else:
        session["violation_rois"] = result
        session["rois"] = result
    out_path.write_text(json.dumps(session, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
