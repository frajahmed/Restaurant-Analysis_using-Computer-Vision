"""SQLite storage for finished visits and per-minute snapshots."""
from __future__ import annotations

import os
import sqlite3
import threading
import time
from pathlib import Path

from rcv.config import ROOT
from rcv.tables import Visit

#: Database file. Defaults to the project root; can be changed with the ``RCV_DB``
#: environment variable (the Docker image keeps it in the ``data/`` volume).
DB_PATH = Path(os.environ.get("RCV_DB", ROOT / "restaurant.db"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS visits (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    table_id  TEXT    NOT NULL,
    start_ts  REAL    NOT NULL,   -- Unix time (seconds)
    end_ts    REAL    NOT NULL,
    duration  REAL    NOT NULL,   -- seconds
    guests    INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshots (
    ts              REAL PRIMARY KEY,
    guests          INTEGER NOT NULL,
    occupied_tables INTEGER NOT NULL,
    total_tables    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_visits_start ON visits(start_ts);
"""


class Database:
    """Small thread-safe wrapper around one SQLite connection.

    The analysis thread writes and the web server reads at the same time, so
    every access is guarded by a lock.
    """

    def __init__(self, path: Path | str = DB_PATH):
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)

    # ------------------------------------------------------------------ write
    def add_visit(self, visit: Visit) -> None:
        """Store a finished visit (``start``/``end`` must be Unix timestamps)."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO visits (table_id, start_ts, end_ts, duration, guests) VALUES (?, ?, ?, ?, ?)",
                (visit.table_id, visit.start, visit.end, visit.duration, int(visit.guests)))
            self._conn.commit()

    def add_snapshot(self, ts: float, guests: int, occupied: int, total: int) -> None:
        """Store the current occupancy (used for the hourly chart)."""
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO snapshots VALUES (?, ?, ?, ?)",
                               (ts, int(guests), int(occupied), int(total)))
            self._conn.commit()

    def clear(self) -> None:
        """Delete all stored data."""
        with self._lock:
            self._conn.execute("DELETE FROM visits")
            self._conn.execute("DELETE FROM snapshots")
            self._conn.commit()

    # ------------------------------------------------------------------ read
    def recent_visits(self, limit: int = 50) -> list[dict]:
        """The most recently finished visits, newest first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM visits ORDER BY end_ts DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def today_summary(self) -> dict:
        """Number of visits, guests and average duration since midnight."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n, COALESCE(SUM(guests), 0) AS g, COALESCE(AVG(duration), 0) AS d "
                "FROM visits WHERE start_ts >= ?", (self._midnight(),)).fetchone()
        return {"visits": row["n"], "guests": row["g"], "avg_duration": round(row["d"], 1)}

    def hourly(self, day_start: float | None = None) -> list[dict]:
        """For each hour of a day: average/max guests (from snapshots) and visits started."""
        day_start = day_start or self._midnight()
        day_end = day_start + 86400
        hour_expr = "CAST(strftime('%H', {col}, 'unixepoch', 'localtime') AS INTEGER)"
        with self._lock:
            snaps = self._conn.execute(
                f"SELECT {hour_expr.format(col='ts')} AS h, AVG(guests) AS g, MAX(guests) AS m "
                "FROM snapshots WHERE ts >= ? AND ts < ? GROUP BY h", (day_start, day_end)).fetchall()
            visits = self._conn.execute(
                f"SELECT {hour_expr.format(col='start_ts')} AS h, COUNT(*) AS n "
                "FROM visits WHERE start_ts >= ? AND start_ts < ? GROUP BY h", (day_start, day_end)).fetchall()

        by_hour = {r["h"]: r for r in snaps}
        visit_count = {r["h"]: r["n"] for r in visits}
        return [{
            "hour": h,
            "avg_guests": round(by_hour[h]["g"], 1) if h in by_hour else 0,
            "max_guests": by_hour[h]["m"] if h in by_hour else 0,
            "visits": visit_count.get(h, 0),
        } for h in range(24)]

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _midnight() -> float:
        """Unix timestamp of today's local midnight."""
        lt = time.localtime()
        return time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1))
