"""Validate user model + rule selections before pipeline init."""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import config


def parse_csv_ids(raw: str) -> List[str]:
    return [x.strip() for x in (raw or "").split(",") if x.strip()]


def models_for_rules(rules: Set[str]) -> Dict[str, str]:
    """Map rule id -> required catalog model id (lane rules only)."""
    out: Dict[str, str] = {}
    for entry in config.RULE_CATALOG:
        rid = entry["id"]
        if rid not in rules:
            continue
        req = entry.get("requires_model")
        if req:
            out[rid] = str(req)
    return out


def validate_selection(
    model_ids: List[str],
    rule_ids: List[str],
    roi_config: Optional[dict] = None,
) -> Tuple[List[str], List[str]]:
    """
    Returns (errors, warnings).
    errors block run; warnings are heavy-combo hints.
    """
    errors: List[str] = []
    warnings: List[str] = []
    models = set(model_ids or [])
    rules = set(rule_ids or [])

    if not models and not rules:
        errors.append("Select at least one detector or zone rule.")
        return errors, warnings

    for entry in config.RULE_CATALOG:
        rid = entry["id"]
        if rid not in rules:
            continue
        req = entry.get("requires_model")
        if req and req not in models:
            errors.append(f"Rule '{entry.get('title', rid)}' requires the '{req}' model.")

    roi = roi_config or {}
    if "red_light" in rules:
        if not roi.get("signal_roi"):
            errors.append("Red light: configure signal ROI before running.")
        if not (roi.get("violation_rois") or roi.get("rois")):
            errors.append("Red light: configure violation zone ROI(s) before running.")
    if "no_parking" in rules:
        if not roi.get("no_parking_zone"):
            errors.append("No parking: configure the no-parking zone before running.")

    yolo_load_count = len(models)
    if "red_light" in rules:
        yolo_load_count += 1
    if "no_parking" in rules:
        yolo_load_count += 1
    if yolo_load_count >= 3:
        warnings.append(
            f"This run loads {yolo_load_count} YOLO weights — first inference may be slow. "
            "Uncheck detectors you do not need."
        )
    if "red_light" in rules and "no_parking" in rules:
        warnings.append("Red light + no parking both enabled — two extra detectors will load.")

    return errors, warnings


def count_yolo_loads(model_ids: List[str], rule_ids: List[str]) -> int:
    n = len(set(model_ids or []))
    rules = set(rule_ids or [])
    if "red_light" in rules:
        n += 1
    if "no_parking" in rules:
        n += 1
    return n
