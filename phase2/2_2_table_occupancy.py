"""Step 2.2 – Table occupancy.

Tables are polygons from ``config.json`` (draw them with ``python main.py tables``).
A person belongs to a table when their reference point lies inside its polygon.
In this step a table is "occupied" as soon as anybody is inside – no delay yet.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rcv.step import VideoStep  # noqa: E402
from rcv.detection import DuplicateFilter, PersonDetector  # noqa: E402
from rcv.render import Color  # noqa: E402
from rcv.tables import TableManager  # noqa: E402


class TableOccupancy(VideoStep):
    title = "2.2 Table Occupancy"

    def __init__(self):
        super().__init__()
        self.detector = PersonDetector(self.config.model, self.config.conf)
        self.duplicates = DuplicateFilter(self.config.dedupe_iou, self.config.dedupe_contain)
        self.tables = TableManager.from_config(self.config)

    def process(self, frame, now):
        people = self.duplicates.apply(self.detector.detect(frame))
        per_table = self.tables.assign(people)

        busy = 0
        for table in self.tables:
            occupied = len(per_table[table.id]) > 0
            busy += occupied
            color = Color.RED if occupied else Color.GREEN
            self.renderer.polygon(frame, table.area, color)
            self.renderer.label(frame, f"{table.id} {'belegt' if occupied else 'frei'}",
                                table.area.top_point, color)

        self.renderer.people(frame, people)
        self.renderer.panel(frame, [f"Tische belegt: {busy}/{len(self.tables)}"])
        return frame


if __name__ == "__main__":
    TableOccupancy().run()
