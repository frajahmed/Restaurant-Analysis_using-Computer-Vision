"""Step 1.4 – Capacity readout: people compared with the number of seats, plus a coloured bar."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rcv.step import VideoStep  # noqa: E402
from rcv.detection import PersonDetector  # noqa: E402


class CapacityReadout(VideoStep):
    """Head count vs. seats: green < 60 %, yellow < 90 %, red above."""

    title = "1.4 Capacity Readout"

    def __init__(self):
        super().__init__()
        self.detector = PersonDetector(self.config.model, self.config.conf)
        self.seats = self.config.total_seats

    def process(self, frame, now):
        people = self.detector.detect(frame)
        for person in people:
            self.renderer.box(frame, person.box)

        load = len(people) / max(1, self.seats)
        bottom = self.renderer.panel(frame, [f"Gaeste: {len(people)} / {self.seats}",
                                             f"Auslastung: {load:.0%}"])
        self.renderer.progress_bar(frame, load, (12, bottom + 8))
        return frame


if __name__ == "__main__":
    CapacityReadout().run()
