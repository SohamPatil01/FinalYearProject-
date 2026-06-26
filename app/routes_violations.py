"""Read API for stored violations (dashboard table + filters)."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter

from app.db import list_violations

router = APIRouter()


@router.get("/api/violations")
def api_violations(type: Optional[str] = None, limit: int = 200):
    return {"violations": list_violations(type, min(max(limit, 1), 1000))}
