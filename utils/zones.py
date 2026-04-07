"""Zone definitions and helper functions."""

from __future__ import annotations

from typing import Dict

import cv2
import numpy as np


def get_zones(frame_shape) -> Dict[str, np.ndarray]:
    """
    Define polygon zones based on frame size.
    This keeps zones adaptable to different resolutions.
    """
    height, width = frame_shape[:2]

    # No parking zone (example rectangle in lower-middle area)
    no_parking_zone = np.array(
        [
            (int(width * 0.30), int(height * 0.55)),
            (int(width * 0.70), int(height * 0.55)),
            (int(width * 0.75), int(height * 0.95)),
            (int(width * 0.25), int(height * 0.95)),
        ],
        dtype=np.int32,
    )

    # Signal stop-line zone near the top-middle road section
    signal_line_zone = np.array(
        [
            (int(width * 0.20), int(height * 0.35)),
            (int(width * 0.80), int(height * 0.35)),
            (int(width * 0.80), int(height * 0.42)),
            (int(width * 0.20), int(height * 0.42)),
        ],
        dtype=np.int32,
    )

    # Small zone where traffic light appears (top-right by default)
    signal_light_zone = np.array(
        [
            (int(width * 0.88), int(height * 0.05)),
            (int(width * 0.98), int(height * 0.05)),
            (int(width * 0.98), int(height * 0.20)),
            (int(width * 0.88), int(height * 0.20)),
        ],
        dtype=np.int32,
    )

    return {
        "no_parking": no_parking_zone,
        "signal_line": signal_line_zone,
        "signal_light": signal_light_zone,
    }


def point_inside_polygon(point, polygon: np.ndarray) -> bool:
    """Check whether a point (x, y) is inside a polygon."""
    return cv2.pointPolygonTest(polygon, point, False) >= 0
