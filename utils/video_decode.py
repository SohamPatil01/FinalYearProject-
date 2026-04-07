"""
Shared full-video decode: YOLO + zones + plate pipeline.

Optionally writes an annotated MP4 (FastAPI download). Streamlit can disable that and use snapshots only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import cv2
import numpy as np

import config
from utils.pipeline import TrafficPipeline
from utils.ui_common import append_plate_capture_from_frame


def load_image_bgr(path: Path) -> Optional[np.ndarray]:
    """
    Load BGR image for OpenCV / YOLO. Applies EXIF orientation when Pillow is available
    (phone photos often need this; cv2.imread ignores orientation).
    """
    p = str(path)
    try:
        from PIL import Image, ImageOps

        pil = Image.open(p)
        pil = ImageOps.exif_transpose(pil)
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        rgb = np.asarray(pil)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        return cv2.imread(p)


def iter_decode_video(
    in_path: Path,
    out_path: Optional[Path],
    pipeline: TrafficPipeline,
    *,
    write_annotated_mp4: bool = True,
) -> Generator[Dict[str, Any], None, None]:
    """
    Yields {"kind": "frame", ...} per processed frame, then {"kind": "done", ...}.
    When write_annotated_mp4 is True, writes annotated BGR frames to out_path (mp4v).
    """
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    src_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    dec_skip = max(1, int(getattr(config, "VIDEO_DECODE_EVERY_N_FRAME", 1)))
    out_fps = max(1.0, fps / dec_skip)
    writer: Optional[cv2.VideoWriter] = None
    if write_annotated_mp4:
        if out_path is None:
            raise ValueError("out_path is required when write_annotated_mp4 is True")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, out_fps, (w, h))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Could not create output video writer")

    captures: List[Dict[str, Any]] = []
    seen: set = set()
    max_gal = int(getattr(config, "DASHBOARD_PLATE_GALLERY_MAX_ITEMS", 80))
    th_w = int(getattr(config, "DASHBOARD_SIDEBAR_PLATE_THUMB", 132))

    est_decoded = max(1, (src_total + dec_skip - 1) // dec_skip) if src_total > 0 else 1

    read_idx = 0
    frame_idx = 0
    cum_viol = 0
    last_bgr: Optional[np.ndarray] = None
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            read_idx += 1
            if (read_idx - 1) % dec_skip != 0:
                continue
            frame_idx += 1
            orig = frame.copy() if pipeline.use_plate else None
            processed, violations, meta = pipeline.process_frame(frame)
            cum_viol += len(violations)
            n_before = len(captures)
            if pipeline.use_plate and orig is not None:
                append_plate_capture_from_frame(
                    orig,
                    meta.get("plates") or [],
                    frame_idx=frame_idx,
                    seen=seen,
                    captures=captures,
                    max_items=max_gal,
                    thumb_w=th_w,
                )
            new_captures = captures[n_before:]
            if writer is not None:
                writer.write(processed)
            last_bgr = processed
            yield {
                "kind": "frame",
                "frame_idx": frame_idx,
                "frame_total_est": est_decoded,
                "fps": float(fps),
                "dec_skip": dec_skip,
                "processed": processed,
                "violations": list(violations),
                "meta": meta,
                "cum_viol": cum_viol,
                "new_captures": [dict(c) for c in new_captures],
                "unique_plate_tracks": len({int(c["tid"]) for c in captures}),
                "plates_locked_count": len(captures),
            }
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if frame_idx > 0 and src_total <= 0:
        est_decoded = frame_idx

    yield {
        "kind": "done",
        "frame_idx": frame_idx,
        "cum_viol": cum_viol,
        "captures": captures,
        "last_bgr": last_bgr,
        "fps": float(fps),
        "dec_skip": dec_skip,
        "est_decoded": est_decoded,
    }


def iter_decode_image(
    in_path: Path,
    out_path: Optional[Path],
    pipeline: TrafficPipeline,
    *,
    write_annotated_mp4: bool = True,
) -> Generator[Dict[str, Any], None, None]:
    """
    Single image: same event shape as ``iter_decode_video`` (one frame + done).
    Uses ``force_immediate_plate_ocr`` so plate reads appear without temporal gating.
    """
    frame = load_image_bgr(in_path)
    if frame is None or frame.size == 0:
        raise RuntimeError("Could not read image (format not supported or file corrupt)")
    h, w = int(frame.shape[0]), int(frame.shape[1])
    fps = 1.0
    dec_skip = 1
    est_decoded = 1
    frame_idx = 1

    writer: Optional[cv2.VideoWriter] = None
    if write_annotated_mp4:
        if out_path is None:
            raise ValueError("out_path is required when write_annotated_mp4 is True")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, 1.0, (w, h))
        if not writer.isOpened():
            raise RuntimeError("Could not create output video writer")

    captures: List[Dict[str, Any]] = []
    seen: set = set()
    max_gal = int(getattr(config, "DASHBOARD_PLATE_GALLERY_MAX_ITEMS", 80))
    th_w = int(getattr(config, "DASHBOARD_SIDEBAR_PLATE_THUMB", 132))

    orig = frame.copy() if pipeline.use_plate else None
    processed, violations, meta = pipeline.process_frame(
        frame,
        force_immediate_plate_ocr=True,
        force_full_frame_plate=True,
    )

    cum_viol = len(violations)
    n_before = len(captures)
    if pipeline.use_plate and orig is not None:
        append_plate_capture_from_frame(
            orig,
            meta.get("plates") or [],
            frame_idx=frame_idx,
            seen=seen,
            captures=captures,
            max_items=max_gal,
            thumb_w=th_w,
        )
    new_captures = captures[n_before:]

    if writer is not None:
        writer.write(processed)
        writer.release()
        writer = None

    last_bgr = processed
    yield {
        "kind": "frame",
        "frame_idx": frame_idx,
        "frame_total_est": est_decoded,
        "fps": float(fps),
        "dec_skip": dec_skip,
        "processed": processed,
        "violations": list(violations),
        "meta": meta,
        "cum_viol": cum_viol,
        "new_captures": [dict(c) for c in new_captures],
        "unique_plate_tracks": len({int(c["tid"]) for c in captures}),
        "plates_locked_count": len(captures),
    }
    yield {
        "kind": "done",
        "frame_idx": frame_idx,
        "cum_viol": cum_viol,
        "captures": captures,
        "last_bgr": last_bgr,
        "fps": float(fps),
        "dec_skip": dec_skip,
        "est_decoded": est_decoded,
    }
