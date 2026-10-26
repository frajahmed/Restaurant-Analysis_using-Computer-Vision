"""The full analysis pipeline for one frame, used by step 2.7 and the dashboard."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import Config
from .detection import Detector, DuplicateFilter, Person, PersonDetector, Posture, PostureClassifier
from .tables import TableManager, Visit


@dataclass
class FrameResult:
    """Everything the analyser found in one frame."""

    timestamp: float
    people: list[Person]                       # after duplicate removal
    raw_people: list[Person] = field(default_factory=list)  # straight from the detector
    finished_visits: list[Visit] = field(default_factory=list)

    @property
    def standing_count(self) -> int:
        return sum(p.posture is Posture.STANDING for p in self.people)


class RestaurantAnalyzer:
    """Combines detection → duplicate filter → posture → tables.

    Usage::

        analyzer = RestaurantAnalyzer(Config.load())
        for frame in source:
            result = analyzer.process(frame, source.timestamp)

    A custom ``detector`` can be injected (e.g. a fake one in tests).
    """

    def __init__(self, config: Config, detector: Detector | None = None, use_pose: bool = True):
        self.config = config
        self.use_pose = use_pose
        self.detector: Detector = detector or PersonDetector(
            config.pose_model if use_pose else config.model, config.conf)
        self.duplicates = DuplicateFilter(config.dedupe_iou, config.dedupe_contain)
        self.posture = PostureClassifier(config.sit_ratio)
        self.tables = TableManager.from_config(config)

    def process(self, frame: np.ndarray, timestamp: float) -> FrameResult:
        """Analyse one frame and update all tables."""
        raw = self.detector.detect(frame)
        people = self.duplicates.apply(raw)
        if self.use_pose:
            self.posture.apply(people)
        visits = self.tables.update(timestamp, people, seated_only=self.use_pose)
        return FrameResult(timestamp, people, raw, visits)

    def summary(self, result: FrameResult) -> dict[str, Any]:
        """Key figures of the current state, ready to be sent as JSON."""
        seats = self.config.total_seats
        guests = self.tables.guests
        return {
            "people_total": len(result.people),
            "standing": result.standing_count,
            "guests": guests,
            "seats_total": seats,
            "capacity": round(guests / max(1, seats), 3),
            "tables_occupied": self.tables.occupied_count,
            "tables_total": len(self.tables),
            "tables": [t.to_dict(result.timestamp) for t in self.tables],
        }
