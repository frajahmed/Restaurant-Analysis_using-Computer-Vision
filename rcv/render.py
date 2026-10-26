"""Drawing results onto frames and showing them in a window."""
from __future__ import annotations

from typing import Iterable, Sequence

import cv2
import numpy as np

from .detection import SKELETON, Person, Posture
from .geometry import Box, Polygon
from .tables import Table, TableState


class Color:
    """Colour palette in OpenCV's BGR order."""

    GREEN = (80, 200, 120)
    RED = (60, 60, 230)
    YELLOW = (0, 200, 255)
    ORANGE = (0, 140, 255)
    GREY = (160, 160, 160)
    WHITE = (255, 255, 255)
    DARK = (25, 25, 25)
    MAGENTA = (255, 0, 255)
    CYAN = (255, 200, 0)


def format_duration(seconds: float) -> str:
    """Format seconds as ``m:ss`` or ``h:mm:ss``."""
    s = int(seconds)
    if s >= 3600:
        return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"
    return f"{s // 60}:{s % 60:02d}"


class Renderer:
    """All drawing helpers in one place. Every method draws onto ``frame`` in place."""

    FONT = cv2.FONT_HERSHEY_SIMPLEX

    STATE_COLORS = {
        TableState.FREE: Color.GREEN,
        TableState.PENDING: Color.YELLOW,
        TableState.OCCUPIED: Color.RED,
        TableState.LEAVING: Color.ORANGE,
    }
    STATE_LABELS = {
        TableState.FREE: "frei",
        TableState.PENDING: "wartet",
        TableState.OCCUPIED: "belegt",
        TableState.LEAVING: "geht?",
    }

    # ------------------------------------------------------------------ basic shapes
    def label(self, frame: np.ndarray, text: str, origin: Sequence[float],
              color: tuple[int, int, int] = Color.GREEN, scale: float = 0.55) -> None:
        """Text on a filled background so it stays readable on any image."""
        (w, h), _ = cv2.getTextSize(text, self.FONT, scale, 1)
        x, y = int(origin[0]), int(max(origin[1], h + 4))
        cv2.rectangle(frame, (x, y - h - 4), (x + w + 6, y + 4), color, -1)
        cv2.putText(frame, text, (x + 3, y), self.FONT, scale, Color.DARK, 1, cv2.LINE_AA)

    def box(self, frame: np.ndarray, box: Box, color: tuple[int, int, int] = Color.GREEN,
            text: str | None = None) -> None:
        """Bounding box with an optional label above it."""
        x1, y1, x2, y2 = box.as_int()
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if text:
            self.label(frame, text, (x1, y1 - 6), color)

    def polygon(self, frame: np.ndarray, polygon: Polygon, color: tuple[int, int, int],
                alpha: float = 0.22) -> None:
        """Semi-transparent filled polygon with a solid outline."""
        overlay = frame.copy()
        cv2.fillPoly(overlay, [polygon.array], color)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.polylines(frame, [polygon.array], True, color, 2)

    def panel(self, frame: np.ndarray, lines: Iterable[str], origin: tuple[int, int] = (12, 12)) -> int:
        """Semi-transparent info panel with one or more lines of text.

        Returns:
            The y coordinate of the panel's bottom edge (to place things below it).
        """
        lines = list(lines)
        pad, line_height = 10, 26
        width = max(cv2.getTextSize(line, self.FONT, 0.65, 2)[0][0] for line in lines) + 2 * pad
        height = line_height * len(lines) + pad
        x, y = origin
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + width, y + height), Color.DARK, -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (x + pad, y + pad + 16 + i * line_height),
                        self.FONT, 0.65, Color.WHITE, 2, cv2.LINE_AA)
        return y + height

    def progress_bar(self, frame: np.ndarray, value: float, origin: tuple[int, int] = (12, 80),
                     size: tuple[int, int] = (260, 14)) -> None:
        """Horizontal bar for a value between 0 and 1: green → yellow → red."""
        x, y = origin
        w, h = size
        color = Color.GREEN if value < 0.6 else Color.YELLOW if value < 0.9 else Color.RED
        cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 60, 60), -1)
        cv2.rectangle(frame, (x, y), (x + int(w * min(max(value, 0.0), 1.0)), y + h), color, -1)

    # ------------------------------------------------------------------ people
    def person(self, frame: np.ndarray, person: Person, show_point: bool = True) -> None:
        """Box coloured by posture/table, plus the reference point used for table assignment."""
        if person.posture is Posture.STANDING:
            self.box(frame, person.box, Color.GREY, "steht")
        else:
            self.box(frame, person.box, Color.YELLOW if person.table_id else Color.GREEN, person.table_id)
        if show_point:
            x, y = person.reference_point
            cv2.circle(frame, (int(x), int(y)), 4, Color.MAGENTA, -1)

    def people(self, frame: np.ndarray, people: Iterable[Person]) -> None:
        for p in people:
            self.person(frame, p)

    def skeleton(self, frame: np.ndarray, person: Person) -> None:
        """Pose skeleton (only if the person has keypoints)."""
        for a, b in SKELETON:
            pa, pb = person.keypoint(a), person.keypoint(b)
            if pa and pb:
                cv2.line(frame, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), Color.CYAN, 2)

    # ------------------------------------------------------------------ tables
    def table(self, frame: np.ndarray, table: Table, now: float,
              show_guests: bool = True, show_timer: bool = True) -> None:
        """Table area coloured by state, with a label like ``T1 | belegt | 3/4 P | 12:05``."""
        color = self.STATE_COLORS[table.state]
        self.polygon(frame, table.area, color)
        parts = [table.id, self.STATE_LABELS[table.state]]
        if show_guests and table.is_occupied:
            parts.append(f"{table.guests}/{table.seats} P")
        if show_timer and table.is_occupied:
            parts.append(format_duration(table.duration(now)))
        x, y = table.area.top_point
        self.label(frame, " | ".join(parts), (x, y - 8), color)

    def tables(self, frame: np.ndarray, tables: Iterable[Table], now: float, **kwargs) -> None:
        for t in tables:
            self.table(frame, t, now, **kwargs)


class Window:
    """Thin wrapper around an OpenCV window. ``show()`` returns ``False`` once q/ESC is pressed."""

    QUIT_KEYS = (ord("q"), 27)

    def __init__(self, title: str):
        self.title = title

    def show(self, frame: np.ndarray) -> bool:
        cv2.imshow(self.title, frame)
        return (cv2.waitKey(1) & 0xFF) not in self.QUIT_KEYS

    def close(self) -> None:
        cv2.destroyWindow(self.title)

    def __enter__(self) -> "Window":
        return self

    def __exit__(self, *_exc) -> None:
        cv2.destroyAllWindows()
