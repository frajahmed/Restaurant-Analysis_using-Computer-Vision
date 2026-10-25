"""3.3 SQLite: Besuche und minütliche Momentaufnahmen speichern und auslesen."""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "restaurant.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS visits (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    table_id  TEXT    NOT NULL,
    start_ts  REAL    NOT NULL,
    end_ts    REAL    NOT NULL,
    duration  REAL    NOT NULL,
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
    def __init__(self, path: Path | str = DB_PATH):
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.conn.executescript(SCHEMA)

    # ------------------------------------------------------------- schreiben
    def add_visit(self, table_id, start_ts, end_ts, guests):
        with self.lock:
            self.conn.execute(
                "INSERT INTO visits(table_id,start_ts,end_ts,duration,guests) VALUES (?,?,?,?,?)",
                (table_id, start_ts, end_ts, end_ts - start_ts, int(guests)))
            self.conn.commit()

    def add_snapshot(self, ts, guests, occupied, total):
        with self.lock:
            self.conn.execute("INSERT OR REPLACE INTO snapshots VALUES (?,?,?,?)",
                              (ts, int(guests), int(occupied), int(total)))
            self.conn.commit()

    def clear(self):
        with self.lock:
            self.conn.execute("DELETE FROM visits")
            self.conn.execute("DELETE FROM snapshots")
            self.conn.commit()

    # ------------------------------------------------------------- lesen
    def recent_visits(self, limit=50) -> list[dict]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT * FROM visits ORDER BY end_ts DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def today_summary(self) -> dict:
        start = _midnight()
        with self.lock:
            r = self.conn.execute(
                "SELECT COUNT(*) n, COALESCE(SUM(guests),0) g, COALESCE(AVG(duration),0) d "
                "FROM visits WHERE start_ts >= ?", (start,)).fetchone()
        return {"visits": r["n"], "guests": r["g"], "avg_duration": round(r["d"], 1)}

    def hourly(self, day_start: float | None = None) -> list[dict]:
        """Pro Stunde: Ø Gäste (aus Snapshots) und Anzahl begonnener Besuche."""
        day_start = day_start or _midnight()
        day_end = day_start + 86400
        with self.lock:
            snaps = self.conn.execute(
                "SELECT CAST(strftime('%H', ts, 'unixepoch', 'localtime') AS INTEGER) h, AVG(guests) g, "
                "MAX(guests) m FROM snapshots WHERE ts >= ? AND ts < ? GROUP BY h", (day_start, day_end)).fetchall()
            vis = self.conn.execute(
                "SELECT CAST(strftime('%H', start_ts, 'unixepoch', 'localtime') AS INTEGER) h, COUNT(*) n "
                "FROM visits WHERE start_ts >= ? AND start_ts < ? GROUP BY h", (day_start, day_end)).fetchall()
        s = {r["h"]: r for r in snaps}
        v = {r["h"]: r["n"] for r in vis}
        return [{"hour": h, "avg_guests": round(s[h]["g"], 1) if h in s else 0,
                 "max_guests": s[h]["m"] if h in s else 0, "visits": v.get(h, 0)} for h in range(24)]


def _midnight() -> float:
    lt = time.localtime()
    return time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, 0, 0, -1))
