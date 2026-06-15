"""Single on-frame HUD panel for pmain1 (stats, light, source, controls)."""

from __future__ import annotations

import time
from typing import Dict, Optional

import cv2


class RuntimeDashboard:
    def __init__(self, frame_height: int, frame_width: int, panel_width: int = 300):
        self.h = frame_height
        self.w = frame_width
        self.panel_w = max(220, min(panel_width, int(frame_width * 0.42)))
        self._ema_fps: Optional[float] = None
        self._last_t: Optional[float] = None

    def tick_fps(self) -> float:
        now = time.perf_counter()
        if self._last_t is not None:
            dt = now - self._last_t
            if dt > 1e-6:
                inst = 1.0 / dt
                if self._ema_fps is None:
                    self._ema_fps = inst
                else:
                    self._ema_fps = 0.12 * inst + 0.88 * self._ema_fps
        self._last_t = now
        return float(self._ema_fps or 0.0)

    def draw(
        self,
        frame,
        *,
        light_label: str,
        frame_idx: int,
        total_frames: float,
        violation_total: int,
        by_class: Dict[str, int],
        n_tracks: int,
        track_direction: str,
        vehicle_conf: float,
        n_violation_rois: int,
        video_name: str,
        plate_enabled: bool,
        red_grace_frames: int,
    ) -> None:
        fh, fw = frame.shape[0], frame.shape[1]
        if fw != self.w or fh != self.h:
            self.w, self.h = fw, fh
        pw = min(self.panel_w, fw - 8)
        x0 = fw - pw

        cv2.rectangle(frame, (x0, 0), (fw - 1, fh - 1), (20, 22, 28), -1)
        cv2.line(frame, (x0, 0), (x0, fh - 1), (65, 70, 82), 1)

        muted = (130, 135, 145)
        value = (205, 210, 218)
        y = 26
        lh = 20
        lh_tight = 17

        def row(label: str, text: str, color=value, label_w: int = 118):
            cv2.putText(
                frame,
                label,
                (x0 + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                muted,
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                text,
                (x0 + label_w, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            frame,
            "Traffic monitor",
            (x0 + 10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (235, 237, 242),
            1,
            cv2.LINE_AA,
        )
        y += lh + 4

        vname = (video_name or "—")[-36:]
        row("Source", vname, value)
        y += lh
        row("Direction", track_direction.upper(), (190, 200, 230))
        y += lh
        row("Vehicle conf", f"{vehicle_conf:.2f}", value)
        y += lh
        row("Red grace (fr)", str(red_grace_frames), value)
        y += lh
        row("Violation ROIs", str(n_violation_rois), value)
        y += lh
        row("Plate OCR", "on" if plate_enabled else "off", (140, 220, 160) if plate_enabled else (160, 160, 170))
        y += lh + 6

        ll = (light_label or "?").upper()
        if ll == "RED":
            lc = (60, 60, 255)
        elif ll == "GREEN":
            lc = (70, 220, 110)
        else:
            lc = (175, 185, 200)
        row("Light", ll, lc)
        y += lh
        row("Frame", str(frame_idx), value)
        y += lh
        row("Progress", self._progress_str(frame_idx, total_frames), value)
        y += lh
        fps = self._ema_fps or 0.0
        row("FPS", f"{fps:.1f}", value)
        y += lh
        row("Tracks", str(n_tracks), value)
        y += lh
        row("Violations", str(violation_total), (95, 175, 255))
        y += lh + 4

        footer_y = fh - 18
        y_max = footer_y - 28

        cv2.putText(
            frame,
            "Violations by class",
            (x0 + 10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            muted,
            1,
            cv2.LINE_AA,
        )
        y += lh_tight + 4
        if not by_class:
            cv2.putText(
                frame,
                "—",
                (x0 + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (110, 115, 125),
                1,
                cv2.LINE_AA,
            )
        else:
            for name, cnt in sorted(by_class.items(), key=lambda x: -x[1]):
                if y > y_max:
                    break
                cv2.putText(
                    frame,
                    f"{name[:14]}  {cnt}",
                    (x0 + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (188, 192, 200),
                    1,
                    cv2.LINE_AA,
                )
                y += lh_tight

        cv2.line(frame, (x0 + 8, footer_y - 14), (fw - 8, footer_y - 14), (55, 58, 68), 1)
        cv2.putText(
            frame,
            "U D L R  direction   N none   Q quit",
            (x0 + 10, footer_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (150, 155, 165),
            1,
            cv2.LINE_AA,
        )

    @staticmethod
    def _progress_str(frame_idx: int, total_frames: float) -> str:
        if total_frames and total_frames > 1:
            pct = min(100.0, 100.0 * frame_idx / total_frames)
            return f"{frame_idx}/{int(total_frames)} ({pct:.0f}%)"
        return str(frame_idx)
