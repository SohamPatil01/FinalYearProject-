"""Launch OpenCV ROI selector as subprocess."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import config


def session_roi_path(session_id: str) -> Path:
    config.ROI_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    return config.ROI_SESSIONS_DIR / f"{session_id}.json"


def load_session_roi(session_id: str) -> Dict[str, Any]:
    p = session_roi_path(session_id)
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_session_roi(session_id: str, data: Dict[str, Any]) -> Path:
    p = session_roi_path(session_id)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return p


def launch_roi_tool(video_path: str, session_id: str, mode: str) -> Tuple[bool, str]:
    out = session_roi_path(session_id)
    if not out.is_file():
        out.write_text("{}", encoding="utf-8")
    cmd = [
        sys.executable,
        "-m",
        "modules.red_light.roi_cli",
        "--video",
        video_path,
        "--mode",
        mode,
        "--out",
        str(out),
    ]
    proc = subprocess.run(cmd, cwd=str(config.BASE_DIR))
    if proc.returncode == 0:
        return True, f"ROI saved ({mode})"
    if proc.returncode == 2:
        return False, "ROI setup cancelled"
    return False, f"ROI setup failed (exit {proc.returncode})"


def roi_status(session_id: str) -> Dict[str, Any]:
    data = load_session_roi(session_id)
    return {
        "session_id": session_id,
        "signal_roi": bool(data.get("signal_roi")),
        "violation_rois": bool(data.get("violation_rois") or data.get("rois")),
        "no_parking_zone": bool(data.get("no_parking_zone")),
        "raw": data,
    }
