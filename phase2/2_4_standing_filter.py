"""Step 2.4 – Standing filter.

Waiters standing at a table must not make it "occupied". We switch to the pose
model (people + 17 body keypoints) and check whether the thigh is roughly
horizontal (sitting) or vertical (standing). Standing people are drawn grey and
ignored by the table logic.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rcv.step import VideoStep  # noqa: E402
from rcv.detection import DuplicateFilter, PersonDetector, PostureClassifier  # noqa: E402
from rcv.tables import TableManager  # noqa: E402


class StandingFilter(VideoStep):
    title = "2.4 Standing Filter"

    def __init__(self):
        super().__init__()
        self.detector = PersonDetector(self.config.pose_model, self.config.conf)   # pose model!
        self.duplicates = DuplicateFilter(self.config.dedupe_iou, self.config.dedupe_contain)
        self.posture = PostureClassifier(self.config.sit_ratio)
        self.tables = TableManager.from_config(self.config)

    def process(self, frame, now):
        people = self.duplicates.apply(self.detector.detect(frame))
        self.posture.apply(people)
        self.tables.update(now, people, seated_only=True)

        for person in people:
            self.renderer.skeleton(frame, person)
        self.renderer.tables(frame, self.tables, now, show_guests=False)
        self.renderer.people(frame, people)

        standing = sum(p.is_standing for p in people)
        self.renderer.panel(frame, [f"Sitzend: {len(people) - standing}", f"Stehend: {standing}"])
        return frame


if __name__ == "__main__":
    StandingFilter().run()
