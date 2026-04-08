"""
Traffic Violation Detection — OpenCV runner

Inspired by `New-Truck-Detection/main.py` (restricted-time truck + plate ideas), but wired to
this repo’s `TrafficPipeline`, `config`, and class filters — same logic as
`streamlit run dashboard.py`, without the web UI. Scoped plate search uses the truck ROI
bottom strip when `TRUCK_PLATE_ROI_BOTTOM_HALF_ONLY` is enabled in `config.py`.

Run:
  pip install -r requirements.txt
  python main.py
  python main.py --source 0
  python main.py --source /path/to/video.mp4 --max-width 1280

Other entry points:
  streamlit run dashboard.py
  uvicorn web_app:app --host 127.0.0.1 --port 8765
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Union

import cv2

import config
from utils.pipeline import TrafficPipeline
from utils.ui_common import paths_from_model_ids
from utils.video_decode import is_static_image_path, load_image_bgr


def _parse_source(s: str) -> Union[int, str]:
    """Webcam index or file path (string that looks like int → int)."""
    try:
        return int(s)
    except ValueError:
        return s


def _maybe_resize(frame, max_w: int):
    """Downscale wide frames (reference script caps ~1280px width for speed)."""
    if max_w <= 0:
        return frame
    h, w = frame.shape[:2]
    if w <= max_w:
        return frame
    nh = max(1, int(h * (max_w / w)))
    return cv2.resize(frame, (max_w, nh), interpolation=cv2.INTER_AREA)


def _now_for_hud() -> datetime:
    """Match pipeline truck rules when `TRUCK_RULES_TIMEZONE` is set (e.g. Asia/Kolkata)."""
    tz_name = getattr(config, "TRUCK_RULES_TIMEZONE", None)
    if tz_name:
        try:
            from zoneinfo import ZoneInfo

            return datetime.now(ZoneInfo(str(tz_name)))
        except Exception:
            pass
    return datetime.now()


def _draw_text_outline(
    img,
    text: str,
    org,
    *,
    scale: float = 0.55,
    fg=(240, 245, 250),
    bg=(0, 0, 0),
) -> None:
    x, y = org
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)):
        cv2.putText(
            img,
            text,
            (x + dx, y + dy),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            bg,
            2,
            cv2.LINE_AA,
        )
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, fg, 1, cv2.LINE_AA)


def _draw_footer_hud(
    frame,
    *,
    fps: float,
    meta: Dict[str, Any],
    violations: List[str],
) -> None:
    """Extra status bar (pipeline already draws truck rules / violations at the top)."""
    h, _w = frame.shape[:2]
    t = _now_for_hud()
    tz = getattr(config, "TRUCK_RULES_TIMEZONE", None)
    clock = t.strftime("%Y-%m-%d %H:%M:%S")
    if tz:
        clock = f"{clock} ({tz})"

    start_h = getattr(config, "TRUCK_VIOLATIONS_ACTIVE_START_HOUR", 0)
    end_h = getattr(config, "TRUCK_VIOLATIONS_ACTIVE_END_HOUR", 24)
    window_txt = f"Truck rules window (config): {start_h:02d}:00 – {end_h:02d}:00"

    lines = [
        f"FPS {fps:5.1f}  |  {clock}",
        window_txt,
    ]
    if meta.get("truck_rules_active"):
        lines.append("Restricted window: ACTIVE (truck restricted-hours rule applies)")
    elif meta.get("truck_tracking_only"):
        lines.append("Restricted window: OFF — tracking + plate only")
    lines.append(f"Violations (this frame): {len(violations)}")

    y = h - 18
    for line in reversed(lines):
        if y < 60:
            break
        _draw_text_outline(frame, line[:140], (10, y), scale=0.52)
        y -= 22


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenCV loop: same TrafficPipeline as the dashboard, on webcam or file.",
    )
    parser.add_argument(
        "--source",
        default=str(config.VIDEO_SOURCE),
        help="Webcam index (0, 1, …) or path to mp4/avi (default: config.VIDEO_SOURCE)",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=1280,
        help="If frame width exceeds this, resize down (0 = disable). Default 1280.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N frames (0 = until video ends or user quits).",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        metavar="ID",
        help=(
            "YOLO heads to load (catalog ids: truck, triple, helmet, plate). "
            "If omitted, loads all paths in config.MODEL_PATHS plus plate when available."
        ),
    )
    args = parser.parse_args()

    src = _parse_source(args.source)
    if args.models:
        paths = paths_from_model_ids([str(x).strip() for x in args.models if str(x).strip()])
        if not paths:
            raise SystemExit("No usable .pt files for --models (check models/ and catalog ids).")
        pipeline = TrafficPipeline(model_paths=paths)
    else:
        pipeline = TrafficPipeline()
    window = getattr(config, "WINDOW_NAME", "Traffic Violation Detection")

    prev_t = time.perf_counter()
    fps_smooth = 0.0
    n = 0

    # Still images: VideoCapture is unreliable; load with Pillow/OpenCV imread path.
    if isinstance(src, str) and is_static_image_path(Path(src)):
        frame = load_image_bgr(Path(src))
        if frame is None or frame.size == 0:
            raise RuntimeError(f"Could not read image {src!r}")
        frame = _maybe_resize(frame, args.max_width)
        frame, violations, meta = pipeline.process_frame(
            frame,
            force_immediate_plate_ocr=True,
            force_full_frame_plate=True,
        )
        now_t = time.perf_counter()
        dt = max(now_t - prev_t, 1e-6)
        fps_smooth = 1.0 / dt
        _draw_footer_hud(frame, fps=fps_smooth, meta=meta, violations=violations)
        cv2.imshow(window, frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source {src!r}. Check path or camera index.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = _maybe_resize(frame, args.max_width)
        frame, violations, meta = pipeline.process_frame(frame)

        now_t = time.perf_counter()
        dt = max(now_t - prev_t, 1e-6)
        prev_t = now_t
        inst_fps = 1.0 / dt
        fps_smooth = inst_fps if fps_smooth <= 0 else (0.85 * fps_smooth + 0.15 * inst_fps)

        _draw_footer_hud(frame, fps=fps_smooth, meta=meta, violations=violations)

        cv2.imshow(window, frame)
        n += 1
        if args.max_frames and n >= args.max_frames:
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
