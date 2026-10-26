"""Step 2.1 – Remove duplicates.

YOLO sometimes returns two boxes for one person (e.g. full body + upper body).
Grey = discarded duplicate, green = kept.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rcv.step import VideoStep  # noqa: E402
from rcv.detection import DuplicateFilter, PersonDetector  # noqa: E402
from rcv.render import Color  # noqa: E402


class RemoveDuplicates(VideoStep):
    title = "2.1 Remove Duplicates"

    def __init__(self):
        super().__init__()
        self.detector = PersonDetector(self.config.model, self.config.conf)
        self.duplicates = DuplicateFilter(self.config.dedupe_iou, self.config.dedupe_contain)

    def process(self, frame, now):
        raw = self.detector.detect(frame)
        kept = self.duplicates.apply(raw)

        for person in raw:
            if person not in kept:      # identity comparison (Person uses eq=False)
                self.renderer.box(frame, person.box, Color.GREY, "doppelt")
        for person in kept:
            self.renderer.box(frame, person.box, Color.GREEN)

        self.renderer.panel(frame, [f"Roh: {len(raw)}", f"Ohne Duplikate: {len(kept)}"])
        return frame


if __name__ == "__main__":
    RemoveDuplicates().run()
