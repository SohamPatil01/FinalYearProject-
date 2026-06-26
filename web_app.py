"""VioLane web UI: FastAPI + HTML/JS (uploaded video only).

Live preview: POST /api/run-stream — Server-Sent Events with per-frame annotated JPEGs,
violations, and plate track state while the clip is processed.

The application is assembled from the ``app`` package:
  - app.routes_pages    GET /
  - app.routes_catalog  /api/catalog
  - app.routes_media    /api/stage-upload, /api/roi/*, /api/download
  - app.routes_run      /api/run, /api/run-stream

Run: uvicorn web_app:app --reload --host 127.0.0.1 --port 8765
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.db import init_db
from app.routes_catalog import router as catalog_router
from app.routes_media import router as media_router
from app.routes_pages import router as pages_router
from app.routes_run import router as run_router
from app.routes_violations import router as violations_router
from app.runtime import STATIC_DIR

app = FastAPI(title="VioLane")
init_db()

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.include_router(pages_router)
app.include_router(catalog_router)
app.include_router(media_router)
app.include_router(run_router)
app.include_router(violations_router)
