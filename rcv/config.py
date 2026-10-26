"""Project configuration.

All settings live in ``config.json`` in the project root. This module loads
that file into typed objects (:class:`Config`, :class:`TableConfig`) so the
rest of the code works with attributes instead of dictionary keys and typos
are caught immediately.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, ClassVar

#: Project root folder (one level above ``rcv/``).
ROOT = Path(__file__).resolve().parent.parent
#: Default location of the configuration file.
CONFIG_PATH = ROOT / "config.json"


class ConfigError(ValueError):
    """Raised when a configuration value is invalid."""


@dataclass
class TableConfig:
    """A table as it appears in the camera image.

    Attributes:
        id: Short name, e.g. ``"T1"``.
        seats: Number of seats at this table.
        polygon: Corner points ``[[x, y], ...]`` of the area (table incl. chairs).
    """

    id: str
    seats: int
    polygon: list[list[int]]

    def __post_init__(self) -> None:
        self.seats = int(self.seats)
        if len(self.polygon) < 3:
            raise ConfigError(f"Tisch {self.id}: mindestens 3 Punkte nötig")
        if self.seats < 1:
            raise ConfigError(f"Tisch {self.id}: mindestens 1 Sitzplatz nötig")
        self.polygon = [[int(x), int(y)] for x, y in self.polygon]


@dataclass
class Config:
    """All project settings. Defaults apply when a key is missing in ``config.json``."""

    # --- Source & models --------------------------------------------------
    source: str = "videos/restaurant.mp4"   # file, webcam index ("0") or rtsp://...
    model: str = "yolo11n.pt"               # detection model (phase 1)
    pose_model: str = "yolo11n-pose.pt"     # pose model (from step 2.4 on)
    process_width: int = 1280               # frames are resized to this width

    # --- Detection --------------------------------------------------------
    conf: float = 0.4                       # minimum detection confidence
    dedupe_iou: float = 0.6                 # boxes overlapping more than this are duplicates
    dedupe_contain: float = 0.8             # a box this much inside another is a duplicate
    sit_ratio: float = 0.55                 # thigh/torso ratio below which a person sits

    # --- Table logic ------------------------------------------------------
    occupy_after_sec: float = 3.0           # presence needed before a table is "occupied"
    free_after_sec: float = 6.0             # absence needed before a table is "free" again
    lock_window_sec: float = 4.0            # time window used to smooth the guest count
    seats_total: int = 0                    # 0 = sum of all table seats

    tables: list[TableConfig] = field(default_factory=list)

    # Fields that may be edited from the settings page, with their types.
    NUMERIC_FIELDS: ClassVar[dict[str, type]] = {
        "process_width": int, "conf": float, "dedupe_iou": float, "dedupe_contain": float,
        "sit_ratio": float, "occupy_after_sec": float, "free_after_sec": float,
        "lock_window_sec": float, "seats_total": int,
    }
    TEXT_FIELDS: ClassVar[tuple[str, ...]] = ("source", "model", "pose_model")

    # ------------------------------------------------------------------ load / save
    @classmethod
    def load(cls, path: Path | str = CONFIG_PATH) -> "Config":
        """Load the configuration from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """Build a configuration from a dict. Unknown keys are ignored."""
        known = {f.name for f in fields(cls)}
        values = {k: v for k, v in data.items() if k in known and k != "tables"}
        tables = [TableConfig(**t) for t in data.get("tables", [])]
        return cls(**values, tables=tables)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict."""
        return asdict(self)

    def save(self, path: Path | str = CONFIG_PATH) -> None:
        """Write the configuration to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------ update
    def update(self, data: dict[str, Any]) -> None:
        """Apply changed values (e.g. from the settings page) with validation.

        Raises:
            ConfigError: if a value cannot be converted to the expected type.
        """
        for key, cast in self.NUMERIC_FIELDS.items():
            if key in data:
                try:
                    setattr(self, key, cast(data[key]))
                except (TypeError, ValueError) as exc:
                    raise ConfigError(f"Ungültiger Wert für {key}: {data[key]!r}") from exc
        for key in self.TEXT_FIELDS:
            if key in data and str(data[key]).strip():
                setattr(self, key, str(data[key]).strip())
        if "tables" in data:
            self.tables = [
                TableConfig(id=str(t.get("id") or f"T{i + 1}"),
                            seats=int(t.get("seats", 4)),
                            polygon=t.get("polygon", []))
                for i, t in enumerate(data["tables"])
            ]

    # ------------------------------------------------------------------ derived values
    @property
    def total_seats(self) -> int:
        """Total seats: ``seats_total`` or, if 0, the sum over all tables."""
        return self.seats_total or sum(t.seats for t in self.tables)
