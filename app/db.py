"""SQLite store for detected violations (stdlib sqlite3, one table).

ponytail: one process-wide lock serialises writes. Fine while a single job runs
at a time; upgrade path is WAL + a connection pool if concurrent jobs land here.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import config

DB_PATH = config.BASE_DIR / "violations.db"
_lock = threading.Lock()

_COLS = (
    "ts",
    "camera",
    "violation_type",
    "vehicle_class",
    "track_id",
    "plate_text",
    "plate_conf",
    "fine_amount",
    "frame_index",
    "t_sec",
    "evidence",
)


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(DB_PATH))
    c.row_factory = sqlite3.Row
    return c


def init_db() -> None:
    with _lock, _conn() as c:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                camera TEXT,
                violation_type TEXT,
                vehicle_class TEXT,
                track_id TEXT,
                plate_text TEXT,
                plate_conf REAL,
                fine_amount INTEGER,
                frame_index INTEGER,
                t_sec REAL,
                evidence TEXT
            )
            """
        )


def insert_violation(**f: Any) -> None:
    f.setdefault("ts", datetime.now().isoformat(timespec="seconds"))
    vals = [f.get(k) for k in _COLS]
    placeholders = ",".join("?" * len(_COLS))
    with _lock, _conn() as c:
        c.execute(
            f"INSERT INTO violations ({','.join(_COLS)}) VALUES ({placeholders})",
            vals,
        )


def list_violations(violation_type: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
    q = "SELECT * FROM violations"
    args: List[Any] = []
    if violation_type:
        q += " WHERE violation_type LIKE ?"
        args.append(f"%{violation_type}%")
    q += " ORDER BY id DESC LIMIT ?"
    args.append(int(limit))
    with _lock, _conn() as c:
        return [dict(r) for r in c.execute(q, args).fetchall()]


if __name__ == "__main__":  # self-check: python -m app.db
    import tempfile

    DB_PATH = Path(tempfile.mkdtemp()) / "t.db"  # type: ignore[assignment,name-defined]
    init_db()
    insert_violation(camera="CAM-01", violation_type="Red light jump", fine_amount=1000)
    insert_violation(camera="CAM-02", violation_type="No helmet", fine_amount=500)
    rows = list_violations()
    assert len(rows) == 2 and rows[0]["id"] > rows[1]["id"], rows
    assert len(list_violations("helmet")) == 1, "type filter broken"
    assert list_violations(limit=1) and len(list_violations(limit=1)) == 1
    print("app.db self-check OK")
