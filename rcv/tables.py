"""Table logic: assigning people, occupancy state, stable guest count and visit timer."""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .config import Config, TableConfig
from .detection import Person, Posture
from .geometry import Polygon


class TableState(str, Enum):
    """Life cycle of a table::

        FREE ──person──▶ PENDING ──occupy_after──▶ OCCUPIED ──empty──▶ LEAVING ──free_after──▶ FREE
                            │                         ▲                   │
                            └──── empty again ──▶ FREE └──── back again ──┘
    """

    FREE = "free"          # nobody there
    PENDING = "pending"    # someone there, but not long enough yet (could be passing by)
    OCCUPIED = "occupied"  # guests are seated
    LEAVING = "leaving"    # empty for a moment – maybe just hidden, maybe gone

    @property
    def is_occupied(self) -> bool:
        return self in (TableState.OCCUPIED, TableState.LEAVING)


@dataclass(frozen=True)
class Visit:
    """A finished visit at one table. Times are in the timestamp unit of the video source."""

    table_id: str
    start: float
    end: float
    guests: int

    @property
    def duration(self) -> float:
        return self.end - self.start


class CountSmoother:
    """Keeps a noisy per-frame count steady (steps 2.5 / 2.6).

    * ``mode`` – the most frequent value within the last ``window`` seconds
      (ties go to the larger value, because people are hidden more often than
      they are invented).
    * ``locked`` – the displayed value. It only changes once a new mode has
      stayed the same for a whole window.
    """

    def __init__(self, window: float = 4.0):
        self.window = window
        self.locked = 0
        self.mode = 0
        self._history: deque[tuple[float, int]] = deque()
        self._candidate: int | None = None
        self._candidate_since = 0.0

    @property
    def has_data(self) -> bool:
        return bool(self._history)

    def reset(self) -> None:
        """Forget everything (used when a table becomes free)."""
        self._history.clear()
        self.locked = self.mode = 0
        self._candidate = None

    def force(self, value: int) -> None:
        """Set the locked value directly, e.g. when a table becomes occupied."""
        self.locked = value
        self._candidate = None

    def add(self, timestamp: float, count: int) -> int:
        """Add one measurement and return the (possibly unchanged) locked value."""
        # 1. Keep only measurements within the time window.
        self._history.append((timestamp, count))
        while self._history and timestamp - self._history[0][0] > self.window:
            self._history.popleft()

        # 2. Most frequent value in the window; ties → larger value.
        counts = Counter(value for _, value in self._history)
        best = max(counts.values())
        self.mode = max(value for value, n in counts.items() if n == best)

        # 3. Only switch the locked value once the new mode has held for a whole window.
        if self.mode == self.locked:
            self._candidate = None
        elif self._candidate != self.mode:
            self._candidate, self._candidate_since = self.mode, timestamp
        elif timestamp - self._candidate_since >= self.window:
            self.locked, self._candidate = self.mode, None
        return self.locked


class Table:
    """One table: its area, its state and the current visit."""

    def __init__(self, config: TableConfig, occupy_after: float = 3.0,
                 free_after: float = 6.0, lock_window: float = 4.0):
        self.id = config.id
        self.seats = config.seats
        self.area = Polygon(config.polygon)
        self.occupy_after = occupy_after
        self.free_after = free_after

        self.state = TableState.FREE
        self.raw_count = 0                     # seated people in the current frame
        self.guests = 0                        # smoothed guest count
        self.occupied_since: float | None = None

        self._smoother = CountSmoother(lock_window)
        self._first_seen: float | None = None  # first frame of the current presence
        self._last_seen: float | None = None   # last frame anyone was at the table
        self._max_guests = 0                   # largest group size during the visit

    # ------------------------------------------------------------------ queries
    @property
    def is_occupied(self) -> bool:
        return self.state.is_occupied

    def contains(self, person: Person) -> bool:
        """``True`` if the person's reference point lies within this table's area."""
        return self.area.contains(person.reference_point)

    def duration(self, now: float) -> float:
        """Seconds since the current visit started (0 if the table is not occupied)."""
        return now - self.occupied_since if self.is_occupied and self.occupied_since is not None else 0.0

    # ------------------------------------------------------------------ update
    def update(self, now: float, count: int) -> Visit | None:
        """Feed the number of seated people in this frame.

        Returns:
            A :class:`Visit` when a visit has just ended, otherwise ``None``.
        """
        self.raw_count = count
        present = count > 0
        if present:
            self._last_seen = now
            if self._first_seen is None:
                self._first_seen = now

        finished = self._advance_state(now, present)
        self._update_guests(now, present)
        return finished

    def _advance_state(self, now: float, present: bool) -> Visit | None:
        """State machine – see :class:`TableState`."""
        if self.state is TableState.FREE:
            if present:
                self.state = TableState.PENDING

        elif self.state is TableState.PENDING:
            if not present:
                # Only passing by → back to free.
                self._reset()
            elif now - self._first_seen >= self.occupy_after:
                self.state = TableState.OCCUPIED
                self.occupied_since = self._first_seen
                # Start the visit with the smoothed value collected while pending.
                self._smoother.force(self._smoother.mode if self._smoother.has_data else self.raw_count)

        elif self.state.is_occupied:
            if present:
                self.state = TableState.OCCUPIED
            else:
                self.state = TableState.LEAVING
                if now - self._last_seen >= self.free_after:
                    visit = Visit(self.id, self.occupied_since, self._last_seen, self._max_guests)
                    self._reset()
                    return visit
        return None

    def _update_guests(self, now: float, present: bool) -> None:
        """Smooth the guest count. Gaps (nobody visible) are not counted as 0 guests."""
        if present or self.state is TableState.PENDING:
            self.guests = self._smoother.add(now, self.raw_count)
        if self.state is TableState.FREE:
            self.guests = 0
        elif self.is_occupied:
            self.guests = max(1, self.guests)   # an occupied table has at least one guest
            self._max_guests = max(self._max_guests, self.guests)

    def _reset(self) -> None:
        self.state = TableState.FREE
        self.guests = 0
        self.occupied_since = self._first_seen = None
        self._max_guests = 0
        self._smoother.reset()

    # ------------------------------------------------------------------ output
    def to_dict(self, now: float) -> dict[str, Any]:
        """JSON-friendly snapshot for the dashboard."""
        return {
            "id": self.id,
            "seats": self.seats,
            "state": self.state.value,
            "occupied": self.is_occupied,
            "guests": self.guests if self.is_occupied else 0,
            "raw": self.raw_count,
            "duration": round(self.duration(now), 1),
            "polygon": self.area.points,
        }


class TableManager:
    """All tables of the restaurant: assigns people and updates every table."""

    def __init__(self, tables: list[Table]):
        self.tables = tables

    @classmethod
    def from_config(cls, config: Config) -> "TableManager":
        return cls([Table(t, config.occupy_after_sec, config.free_after_sec, config.lock_window_sec)
                    for t in config.tables])

    # ------------------------------------------------------------------ assignment
    def assign(self, people: list[Person], seated_only: bool = False) -> dict[str, list[Person]]:
        """Assign each person to the first table whose area contains them.

        Args:
            seated_only: ignore standing people (waiters) – step 2.4.

        Returns:
            ``{table_id: [people at that table]}``; sets ``person.table_id``.
        """
        result: dict[str, list[Person]] = {t.id: [] for t in self.tables}
        for person in people:
            person.table_id = None
            if seated_only and person.posture is Posture.STANDING:
                continue
            for table in self.tables:
                if table.contains(person):
                    person.table_id = table.id
                    result[table.id].append(person)
                    break
        return result

    def update(self, now: float, people: list[Person], seated_only: bool = True) -> list[Visit]:
        """Assign people, update all tables and return visits that just ended."""
        per_table = self.assign(people, seated_only)
        visits = [table.update(now, len(per_table[table.id])) for table in self.tables]
        return [v for v in visits if v is not None]

    # ------------------------------------------------------------------ totals
    @property
    def guests(self) -> int:
        """Guests at all occupied tables (smoothed)."""
        return sum(t.guests for t in self.tables if t.is_occupied)

    @property
    def raw_guests(self) -> int:
        return sum(t.raw_count for t in self.tables)

    @property
    def occupied_count(self) -> int:
        return sum(t.is_occupied for t in self.tables)

    def __len__(self) -> int:
        return len(self.tables)

    def __iter__(self):
        return iter(self.tables)
