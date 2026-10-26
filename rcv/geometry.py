"""Geometry primitives: axis-aligned rectangle (:class:`Box`) and polygon (:class:`Polygon`)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np

Point = tuple[float, float]


@dataclass(frozen=True)
class Box:
    """Axis-aligned rectangle in pixel coordinates (top-left to bottom-right)."""

    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def from_seq(cls, values: Sequence[float]) -> "Box":
        """Create a box from ``[x1, y1, x2, y2]``."""
        x1, y1, x2, y2 = (float(v) for v in values[:4])
        return cls(x1, y1, x2, y2)

    # ------------------------------------------------------------------ size
    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        """Height / width – standing people are much taller than they are wide."""
        return self.height / max(1.0, self.width)

    def as_int(self) -> tuple[int, int, int, int]:
        """Integer coordinates for OpenCV drawing functions."""
        return int(self.x1), int(self.y1), int(self.x2), int(self.y2)

    def anchor(self, ratio: float = 0.7) -> Point:
        """Reference point: horizontally centred, ``ratio`` of the height down (≈ seat level)."""
        return (self.x1 + self.x2) / 2, self.y1 + ratio * self.height

    # ------------------------------------------------------------------ comparing two boxes
    def intersection(self, other: "Box") -> float:
        """Area shared with ``other``."""
        w = min(self.x2, other.x2) - max(self.x1, other.x1)
        h = min(self.y2, other.y2) - max(self.y1, other.y1)
        return max(0.0, w) * max(0.0, h)

    def iou(self, other: "Box") -> float:
        """Intersection over Union: 0 = no overlap, 1 = identical."""
        inter = self.intersection(other)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def inside_ratio(self, other: "Box") -> float:
        """Fraction of this box that lies inside ``other`` (0 … 1)."""
        return self.intersection(other) / self.area if self.area > 0 else 0.0


class Polygon:
    """Closed polygon, e.g. the area of a table."""

    def __init__(self, points: Iterable[Sequence[float]]):
        self.points: list[list[int]] = [[int(x), int(y)] for x, y in points]
        if len(self.points) < 3:
            raise ValueError("Ein Polygon braucht mindestens 3 Punkte")
        # OpenCV expects an int32 array of shape (N, 2).
        self._array = np.array(self.points, dtype=np.int32)

    @property
    def array(self) -> np.ndarray:
        """Points as a NumPy array (for OpenCV)."""
        return self._array

    def contains(self, point: Point) -> bool:
        """``True`` if the point lies inside the polygon or on its edge."""
        return cv2.pointPolygonTest(self._array, (float(point[0]), float(point[1])), False) >= 0

    @property
    def centroid(self) -> Point:
        """Mean of all corner points."""
        cx, cy = self._array.mean(axis=0)
        return float(cx), float(cy)

    @property
    def top_point(self) -> tuple[int, int]:
        """Top-most corner – a good position for a label."""
        x, y = min(self.points, key=lambda p: p[1])
        return x, y
