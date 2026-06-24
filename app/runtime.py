"""Shared runtime state for the VioLane web app.

Kept in one module so every router talks to the same in-memory stores and the
same Jinja templates instance (unchanged from the original ``web_app.py``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from fastapi.templating import Jinja2Templates

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
STATIC_DIR = BASE_DIR / "static"

# In-memory stores (same lifetime/semantics as before).
JOBS: Dict[str, Dict[str, Any]] = {}
STAGED: Dict[str, Dict[str, Any]] = {}
