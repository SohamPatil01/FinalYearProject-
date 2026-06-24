"""Catalog route: available models + zone rules."""

from __future__ import annotations

from fastapi import APIRouter

import config

router = APIRouter()


@router.get("/api/catalog")
def api_catalog():
    entries = []
    paths = config.catalog_model_paths()
    for e in config.MODEL_CATALOG:
        p = paths.get(e["id"], "")
        ok = bool(p and config.is_model_file_usable(p))
        entries.append(
            {
                "id": e["id"],
                "title": e["title"],
                "summary": e.get("summary", ""),
                "ready": ok,
            }
        )
    rules = [
        {
            "id": r["id"],
            "title": r["title"],
            "summary": r.get("summary", ""),
            "needs_roi": r.get("needs_roi") or [],
            "requires_model": r.get("requires_model"),
        }
        for r in config.RULE_CATALOG
    ]
    return {
        "models": entries,
        "rules": rules,
        "heavy_combo_threshold": 3,
        "heavy_combo_hints": [
            "3+ YOLO weights selected — consider unchecking unused detectors.",
            "Red light + no parking together load yolov10s and yolov8n in addition to lane models.",
        ],
    }
