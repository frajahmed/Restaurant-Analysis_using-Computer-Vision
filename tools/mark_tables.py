"""Draw the tables onto a still frame of the video and save them in ``config.json``.

Controls:
    left click     add a point (outline the table including its chairs)
    ENTER / n      close the polygon → then press 1-9 for the number of seats
    u              undo the last point (or the last table)
    c              clear all tables
    s              save
    q / ESC        quit

Usage:  python main.py tables  [--keep]   (--keep starts with the saved tables)
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rcv.config import Config, TableConfig  # noqa: E402
from rcv.geometry import Polygon  # noqa: E402
from rcv.render import Color, Renderer  # noqa: E402
from rcv.video import VideoSource  # noqa: E402

KEY_ENTER, KEY_ESC = 13, 27


class TableMarker:
    """Small OpenCV tool to draw table polygons with the mouse."""

    WINDOW = "Tische markieren"

    def __init__(self, config: Config, keep_existing: bool = False):
        self.config = config
        self.frame = self._grab_frame()
        self.tables: list[TableConfig] = list(config.tables) if keep_existing else []
        self.points: list[list[int]] = []        # polygon currently being drawn
        self.waiting_for_seats = False
        self.renderer = Renderer()

    # ------------------------------------------------------------------ setup
    def _grab_frame(self) -> np.ndarray:
        """Return a frame about one second into the video (the first one is often black)."""
        frame = None
        with VideoSource(self.config.source, self.config.process_width, loop=False) as source:
            for _ in range(int(source.fps)):
                next_frame = source.read()
                if next_frame is None:
                    break
                frame = next_frame
        if frame is None:
            raise RuntimeError("Konnte kein Bild aus der Videoquelle lesen.")
        return frame

    # ------------------------------------------------------------------ events
    def _on_mouse(self, event, x, y, *_args) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and not self.waiting_for_seats:
            self.points.append([x, y])

    def _handle_key(self, key: int) -> bool:
        """React to a key press. Returns ``False`` when the tool should close."""
        if self.waiting_for_seats and ord("1") <= key <= ord("9"):
            self.tables.append(TableConfig(f"T{len(self.tables) + 1}", key - ord("0"), self.points))
            self.points = []
            self.waiting_for_seats = False
        elif key in (KEY_ENTER, ord("n")) and len(self.points) >= 3:
            self.waiting_for_seats = True
        elif key == ord("u"):
            if self.points:
                self.points.pop()
            elif self.tables:
                self.tables.pop()
        elif key == ord("c"):
            self.tables.clear()
            self.points.clear()
        elif key == ord("s"):
            self.save()
        elif key in (ord("q"), KEY_ESC):
            return False
        return True

    # ------------------------------------------------------------------ drawing
    def _draw(self) -> np.ndarray:
        img = self.frame.copy()
        for table in self.tables:
            polygon = Polygon(table.polygon)
            self.renderer.polygon(img, polygon, Color.GREEN)
            self.renderer.label(img, f"{table.id} ({table.seats} Plaetze)", polygon.top_point, Color.GREEN)
        for point in self.points:
            cv2.circle(img, tuple(point), 4, Color.YELLOW, -1)
        if len(self.points) > 1:
            cv2.polylines(img, [np.array(self.points, dtype=np.int32)], False, Color.YELLOW, 2)

        hint = ("Sitzplaetze? Ziffer 1-9" if self.waiting_for_seats
                else "Klick=Punkt  ENTER=fertig  u=undo  s=speichern  q=ende")
        self.renderer.label(img, hint, (10, 25), Color.YELLOW, 0.6)
        return img

    # ------------------------------------------------------------------ public API
    def save(self) -> None:
        """Write the tables (and the resulting seat total) to ``config.json``."""
        self.config.tables = list(self.tables)
        self.config.seats_total = 0                 # 0 = derive from the tables
        self.config.save()
        print(f"Gespeichert: {len(self.tables)} Tische, {self.config.total_seats} Plätze → config.json")

    def run(self) -> None:
        cv2.namedWindow(self.WINDOW)
        cv2.setMouseCallback(self.WINDOW, self._on_mouse)
        try:
            while True:
                cv2.imshow(self.WINDOW, self._draw())
                if not self._handle_key(cv2.waitKey(20) & 0xFF):
                    break
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    TableMarker(Config.load(), keep_existing="--keep" in sys.argv).run()
