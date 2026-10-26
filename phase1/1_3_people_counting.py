"""Step 1.3 – People counting: number of boxes = number of people in the frame."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rcv.step import VideoStep  # noqa: E402
from rcv.detection import PersonDetector  # noqa: E402


class PeopleCounting(VideoStep):
    """Shows the current head count in an info panel."""

    title = "1.3 People Counting"

    def __init__(self):
        super().__init__()
        self.detector = PersonDetector(self.config.model, self.config.conf)

    def process(self, frame, now):
        people = self.detector.detect(frame)
        for person in people:
            self.renderer.box(frame, person.box)
        self.renderer.panel(frame, [f"Personen: {len(people)}"])
        return frame


if __name__ == "__main__":
    PeopleCounting().run()
