"""Step 2.3 – Occupancy timer.

A table only counts as occupied after someone has been there for
``occupy_after_sec`` (walking past does not count), and only becomes free after
it has been empty for ``free_after_sec`` (being hidden briefly does not count).

    free ──person──▶ pending ──3 s──▶ occupied ──empty──▶ leaving ──6 s──▶ free
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rcv.step import VideoStep  # noqa: E402
from rcv.detection import DuplicateFilter, PersonDetector  # noqa: E402
from rcv.tables import TableManager  # noqa: E402


class OccupancyTimer(VideoStep):
    title = "2.3 Occupancy Timer"

    def __init__(self):
        super().__init__()
        self.detector = PersonDetector(self.config.model, self.config.conf)
        self.duplicates = DuplicateFilter(self.config.dedupe_iou, self.config.dedupe_contain)
        self.tables = TableManager.from_config(self.config)

    def process(self, frame, now):
        people = self.duplicates.apply(self.detector.detect(frame))
        # No posture yet → count everybody (standing people too).
        self.tables.update(now, people, seated_only=False)

        self.renderer.tables(frame, self.tables, now, show_guests=False)
        self.renderer.people(frame, people)
        self.renderer.panel(frame, [f"Tische belegt: {self.tables.occupied_count}/{len(self.tables)}"])
        return frame


if __name__ == "__main__":
    OccupancyTimer().run()
