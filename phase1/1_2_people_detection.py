"""Step 1.2 – People detection: keep only class 0 ("person") above a minimum confidence."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rcv.step import VideoStep  # noqa: E402
from rcv.detection import PersonDetector  # noqa: E402


class PeopleDetection(VideoStep):
    """Draws a box with the confidence score around every detected person."""

    title = "1.2 People Detection"

    def __init__(self):
        super().__init__()
        # PersonDetector passes classes=[0] and conf=... to YOLO for us.
        self.detector = PersonDetector(self.config.model, self.config.conf)

    def process(self, frame, now):
        for person in self.detector.detect(frame):
            self.renderer.box(frame, person.box, text=f"Person {person.confidence:.2f}")
        return frame


if __name__ == "__main__":
    PeopleDetection().run()
