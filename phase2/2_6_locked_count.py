"""Step 2.6 – Locked count.

Instead of the raw per-frame number we use the most frequent value of the last
``lock_window_sec`` seconds. The displayed ("locked") number only changes once a
new value has been stable that long (see ``rcv.tables.CountSmoother``).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rcv.analyzer import RestaurantAnalyzer  # noqa: E402
from rcv.step import VideoStep  # noqa: E402
from rcv.render import Renderer  # noqa: E402


class LockedCount(VideoStep):
    title = "2.6 Locked Count"

    def __init__(self):
        super().__init__()
        self.analyzer = RestaurantAnalyzer(self.config)

    def process(self, frame, now):
        result = self.analyzer.process(frame, now)
        tables = self.analyzer.tables

        for table in tables:
            color = Renderer.STATE_COLORS[table.state]
            self.renderer.polygon(frame, table.area, color)
            self.renderer.label(frame, f"{table.id}: {table.guests} Gaeste (roh {table.raw_count})",
                                table.area.top_point, color)

        self.renderer.people(frame, result.people)
        self.renderer.panel(frame, [f"Gaeste (stabil): {tables.guests}",
                                    f"Gaeste (roh): {tables.raw_guests}"])
        return frame


if __name__ == "__main__":
    LockedCount().run()
