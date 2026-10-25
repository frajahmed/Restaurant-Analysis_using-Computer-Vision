"""Tisch-Logik: Zuordnung, Belegt-Status mit Wartezeit, stabile Gästezahl, Besuchs-Timer."""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

from .detection import L_HIP, R_HIP, Person
from .utils import anchor_point, point_in_polygon

FREE, PENDING, OCCUPIED, LEAVING = "free", "pending", "occupied", "leaving"


def person_point(p: Person, min_conf=0.35):
    """Hüftmitte aus der Pose, sonst Referenzpunkt der Box."""
    if p.keypoints is not None:
        hips = [p.keypoints[i][:2] for i in (L_HIP, R_HIP) if p.keypoints[i][2] >= min_conf]
        if hips:
            m = np.mean(hips, axis=0)
            return float(m[0]), float(m[1])
    return anchor_point(p.box)


def assign_tables(people: list[Person], tables_cfg: list[dict], only_sitting=False) -> dict[str, list[Person]]:
    out = {t["id"]: [] for t in tables_cfg}
    for p in people:
        if only_sitting and p.posture == "standing":
            continue
        pt = person_point(p)
        for t in tables_cfg:
            if point_in_polygon(pt, t["polygon"]):
                p.table = t["id"]
                out[t["id"]].append(p)
                break
    return out


class CountSmoother:
    """2.5/2.6 – Gästezahl beruhigen.
    `value`  = häufigster Wert im Zeitfenster (Modus, bei Gleichstand der größere)
    `locked` = ändert sich erst, wenn ein neuer Modus `window` Sekunden lang hält."""

    def __init__(self, window=4.0):
        self.window = window
        self.hist: deque = deque()
        self.locked = 0
        self._cand = None
        self._cand_since = 0.0

    def reset(self):
        self.hist.clear()
        self.locked = 0
        self._cand = None

    def add(self, t: float, count: int) -> int:
        self.hist.append((t, count))
        while self.hist and t - self.hist[0][0] > self.window:
            self.hist.popleft()
        c = Counter(v for _, v in self.hist)
        best = max(c.values())
        mode = max(v for v, n in c.items() if n == best)
        self.value = mode

        if mode == self.locked:
            self._cand = None
        elif self._cand != mode:
            self._cand, self._cand_since = mode, t
        elif t - self._cand_since >= self.window:
            self.locked, self._cand = mode, None
        return self.locked

    def force(self, value: int):
        self.locked = value
        self._cand = None


@dataclass
class Table:
    id: str
    seats: int
    polygon: list
    occupy_after: float = 3.0
    free_after: float = 6.0
    lock_window: float = 4.0

    state: str = FREE
    raw: int = 0
    guests: int = 0
    max_guests: int = 0
    occupied_since: float | None = None
    _first_seen: float | None = None
    _last_seen: float | None = None
    smoother: CountSmoother = field(init=False)

    def __post_init__(self):
        self.smoother = CountSmoother(self.lock_window)

    @property
    def occupied(self) -> bool:
        return self.state in (OCCUPIED, LEAVING)

    def duration(self, now: float) -> float:
        return (now - self.occupied_since) if self.occupied_since is not None else 0.0

    def update(self, t: float, raw_count: int) -> dict | None:
        """Gibt ein Besuchs-Dict zurück, wenn ein Besuch gerade beendet wurde."""
        self.raw = raw_count
        present = raw_count > 0
        finished = None

        if present:
            self._last_seen = t
            if self._first_seen is None:
                self._first_seen = t

        if self.state == FREE:
            if present:
                self.state = PENDING
        elif self.state == PENDING:
            if not present:
                self.state, self._first_seen = FREE, None
                self.smoother.reset()
            elif t - self._first_seen >= self.occupy_after:
                self.state = OCCUPIED
                self.occupied_since = self._first_seen
                self.smoother.force(self.smoother.value if self.smoother.hist else raw_count)
        elif self.state in (OCCUPIED, LEAVING):
            if present:
                self.state = OCCUPIED
            else:
                self.state = LEAVING
                if t - self._last_seen >= self.free_after:
                    finished = {
                        "table": self.id,
                        "start": self.occupied_since,
                        "end": self._last_seen,
                        "duration": self._last_seen - self.occupied_since,
                        "guests": self.max_guests,
                    }
                    self.state = FREE
                    self.occupied_since = self._first_seen = None
                    self.max_guests = 0
                    self.smoother.reset()

        # Gästezahl nur während Anwesenheit glätten (Lücken nicht als 0 zählen)
        if present or self.state == PENDING:
            self.guests = self.smoother.add(t, raw_count)
        if self.state == FREE:
            self.guests = 0
        if self.occupied:
            self.guests = max(1, self.guests)
            self.max_guests = max(self.max_guests, self.guests)
        return finished

    def snapshot(self, now: float) -> dict:
        return {
            "id": self.id, "seats": self.seats, "state": self.state,
            "occupied": self.occupied, "guests": self.guests if self.occupied else 0,
            "raw": self.raw, "duration": round(self.duration(now), 1) if self.occupied else 0,
            "polygon": self.polygon,
        }


def build_tables(cfg: dict) -> list[Table]:
    return [Table(t["id"], int(t.get("seats", 4)), t["polygon"],
                  cfg.get("occupy_after_sec", 3.0), cfg.get("free_after_sec", 6.0),
                  cfg.get("lock_window_sec", 4.0)) for t in cfg["tables"]]
