"""Pipeline construction + helpers (rule expansion, full-pass runner, SSE pack).

Moved verbatim from ``web_app.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from utils.pipeline import TrafficPipeline
from utils.rule_validation import parse_csv_ids
from utils.ui_common import paths_from_model_ids
from utils.video_decode import iter_decode_media


def expand_rules(model_ids: List[str], rules_raw: str) -> Set[str]:
    rules: Set[str] = set(parse_csv_ids(rules_raw))
    mids = set(model_ids)
    if "helmet" in mids:
        rules.add("helmet")
    if "triple" in mids:
        rules.add("triple")
    if "truck" in mids:
        rules.add("truck_restricted")
    if "plate" in mids:
        rules.add("plate_ocr")
    return rules


def build_pipeline(
    model_ids: List[str],
    rules_raw: str,
    roi_config: Dict[str, Any],
    truck_start: int,
    truck_end: int,
) -> TrafficPipeline:
    enabled = expand_rules(model_ids, rules_raw)
    paths = paths_from_model_ids(model_ids) if model_ids else {}
    return TrafficPipeline(
        model_paths=paths if paths else {},
        truck_violation_active_start_hour=int(truck_start),
        truck_violation_active_end_hour=int(truck_end),
        enabled_rules=enabled,
        roi_config=roi_config,
    )


def run_full_pass(
    in_path: Path,
    out_path: Path,
    pipeline: TrafficPipeline,
) -> Tuple[int, int, List[Dict[str, Any]], Optional[np.ndarray], float, int, int]:
    done: Optional[Dict[str, Any]] = None
    last_frame: Optional[Dict[str, Any]] = None
    for ev in iter_decode_media(in_path, out_path, pipeline):
        if ev["kind"] == "frame":
            last_frame = ev
        else:
            done = ev
    if not done:
        raise RuntimeError("Processing produced no frames")
    n_frames = int(done["frame_idx"])
    est = int(done["est_decoded"])
    return (
        n_frames,
        int(done["cum_viol"]),
        done["captures"],
        done["last_bgr"],
        float(done["fps"]),
        int(done["dec_skip"]),
        est,
    )


def sse_pack(obj: Dict[str, Any]) -> bytes:
    return f"data: {json.dumps(obj, default=str)}\n\n".encode("utf-8")
