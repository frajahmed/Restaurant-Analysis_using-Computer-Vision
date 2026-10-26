"""Person detection with YOLO11, duplicate removal and sitting/standing classification."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Protocol

import numpy as np

from .geometry import Box, Point


class Keypoint(IntEnum):
    """Indices of the 17 COCO body keypoints returned by YOLO pose models."""

    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


#: Bones drawn between keypoints when visualising a skeleton.
SKELETON: tuple[tuple[Keypoint, Keypoint], ...] = (
    (Keypoint.LEFT_SHOULDER, Keypoint.RIGHT_SHOULDER),
    (Keypoint.LEFT_SHOULDER, Keypoint.LEFT_ELBOW), (Keypoint.LEFT_ELBOW, Keypoint.LEFT_WRIST),
    (Keypoint.RIGHT_SHOULDER, Keypoint.RIGHT_ELBOW), (Keypoint.RIGHT_ELBOW, Keypoint.RIGHT_WRIST),
    (Keypoint.LEFT_SHOULDER, Keypoint.LEFT_HIP), (Keypoint.RIGHT_SHOULDER, Keypoint.RIGHT_HIP),
    (Keypoint.LEFT_HIP, Keypoint.RIGHT_HIP),
    (Keypoint.LEFT_HIP, Keypoint.LEFT_KNEE), (Keypoint.LEFT_KNEE, Keypoint.LEFT_ANKLE),
    (Keypoint.RIGHT_HIP, Keypoint.RIGHT_KNEE), (Keypoint.RIGHT_KNEE, Keypoint.RIGHT_ANKLE),
)


class Posture(str, Enum):
    """Body posture of a detected person."""

    SITTING = "sitting"
    STANDING = "standing"
    UNKNOWN = "unknown"


@dataclass(eq=False)  # compare by identity: keypoint arrays are not comparable with ==
class Person:
    """A single detected person.

    Attributes:
        box: Bounding box in pixels.
        confidence: Detection confidence (0 … 1).
        keypoints: ``(17, 3)`` array of ``x, y, confidence`` (pose model only).
        posture: Set by :class:`PostureClassifier`.
        table_id: Set when the person is assigned to a table.
    """

    box: Box
    confidence: float
    keypoints: np.ndarray | None = None
    posture: Posture = Posture.UNKNOWN
    table_id: str | None = field(default=None)

    KEYPOINT_MIN_CONF = 0.35  # keypoints below this confidence are treated as invisible

    def keypoint(self, index: Keypoint) -> Point | None:
        """Position of a keypoint, or ``None`` if missing or not confident enough."""
        if self.keypoints is None:
            return None
        x, y, conf = self.keypoints[index]
        return (float(x), float(y)) if conf >= self.KEYPOINT_MIN_CONF else None

    def midpoint(self, left: Keypoint, right: Keypoint) -> Point | None:
        """Mean of the visible keypoints of a left/right pair (e.g. both hips)."""
        points = [p for p in (self.keypoint(left), self.keypoint(right)) if p is not None]
        if not points:
            return None
        return float(np.mean([p[0] for p in points])), float(np.mean([p[1] for p in points]))

    @property
    def is_standing(self) -> bool:
        return self.posture is Posture.STANDING

    @property
    def reference_point(self) -> Point:
        """Point used to decide which table a person belongs to.

        The hip centre is the best choice because it is where the person
        sits. Without pose keypoints, a point 70 % down the box is used.
        """
        hip = self.midpoint(Keypoint.LEFT_HIP, Keypoint.RIGHT_HIP)
        return hip if hip is not None else self.box.anchor(0.7)


class Detector(Protocol):
    """Anything that turns a frame into a list of people (makes testing without YOLO easy)."""

    def detect(self, frame: np.ndarray) -> list[Person]: ...


class PersonDetector:
    """Wraps an Ultralytics YOLO model and returns only people.

    Works with both detection models (``yolo11n.pt``) and pose models
    (``yolo11n-pose.pt``); with a pose model each person also gets keypoints.
    Model weights are downloaded automatically on first use.
    """

    PERSON_CLASS = 0  # COCO class id of "person"

    def __init__(self, model_name: str, confidence: float = 0.4):
        # Imported here so the rest of the package (and the tests) work without PyTorch.
        from ultralytics import YOLO

        self.model_name = model_name
        self.confidence = confidence
        self.model = YOLO(model_name)

    def detect(self, frame: np.ndarray) -> list[Person]:
        """Run the model on one frame and return all detected people."""
        result = self.model(frame, classes=[self.PERSON_CLASS], conf=self.confidence, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        keypoints = None
        if getattr(result, "keypoints", None) is not None and result.keypoints.data is not None:
            keypoints = result.keypoints.data.cpu().numpy()

        return [
            Person(box=Box.from_seq(box), confidence=float(score),
                   keypoints=keypoints[i] if keypoints is not None and i < len(keypoints) else None)
            for i, (box, score) in enumerate(zip(boxes, scores))
        ]


class DuplicateFilter:
    """Removes double boxes for the same person (step 2.1).

    YOLO sometimes returns two boxes for one person, e.g. the full body and
    just the upper body. Two boxes are duplicates when they overlap strongly
    (IoU) or when one lies almost completely inside the other. The box with
    the higher confidence is kept.
    """

    def __init__(self, iou_threshold: float = 0.6, contain_threshold: float = 0.8):
        self.iou_threshold = iou_threshold
        self.contain_threshold = contain_threshold

    def is_duplicate(self, a: Box, b: Box) -> bool:
        """``True`` if the two boxes most likely show the same person."""
        return (a.iou(b) > self.iou_threshold
                or a.inside_ratio(b) > self.contain_threshold
                or b.inside_ratio(a) > self.contain_threshold)

    def apply(self, people: list[Person]) -> list[Person]:
        """Return the people without duplicates, sorted by confidence (highest first)."""
        kept: list[Person] = []
        for person in sorted(people, key=lambda p: p.confidence, reverse=True):
            if not any(self.is_duplicate(person.box, k.box) for k in kept):
                kept.append(person)
        return kept


class PostureClassifier:
    """Decides whether a person sits or stands, using pose keypoints (step 2.4).

    Idea: when sitting, the thigh is roughly horizontal, so the vertical
    distance hip → knee is small compared to the torso length
    (shoulder → hip). When standing, the thigh is vertical and that distance
    is about as long as the torso.

    If the knees are hidden (typical under a table), the box shape decides:
    seated people are more compact, standing people are tall and narrow.
    """

    STANDING_THIGH_RATIO = 0.85   # thigh/torso above this → clearly standing
    STANDING_LEG_RATIO = 1.4      # ankle far below hip → standing
    STANDING_BOX_RATIO = 2.6      # tall narrow box without visible knees → standing
    STANDING_BOX_RATIO_NO_HIPS = 2.4

    def __init__(self, sit_ratio: float = 0.55):
        self.sit_ratio = sit_ratio

    def classify(self, person: Person) -> Posture:
        """Return the posture of one person (does not modify the person)."""
        if person.keypoints is None:
            return Posture.UNKNOWN

        shoulder = person.midpoint(Keypoint.LEFT_SHOULDER, Keypoint.RIGHT_SHOULDER)
        hip = person.midpoint(Keypoint.LEFT_HIP, Keypoint.RIGHT_HIP)
        knee = person.midpoint(Keypoint.LEFT_KNEE, Keypoint.RIGHT_KNEE)
        ankle = person.midpoint(Keypoint.LEFT_ANKLE, Keypoint.RIGHT_ANKLE)

        # Without shoulders and hips there is no reference length – use the box shape.
        if shoulder is None or hip is None:
            if person.box.aspect_ratio > self.STANDING_BOX_RATIO_NO_HIPS:
                return Posture.STANDING
            return Posture.UNKNOWN

        torso = max(1.0, hip[1] - shoulder[1])

        if knee is not None:
            thigh_ratio = (knee[1] - hip[1]) / torso
            if thigh_ratio < self.sit_ratio:
                return Posture.SITTING
            if ankle is not None and (ankle[1] - hip[1]) / torso > self.STANDING_LEG_RATIO:
                return Posture.STANDING
            return Posture.STANDING if thigh_ratio > self.STANDING_THIGH_RATIO else Posture.SITTING

        # Knees not visible → decide by box shape.
        return Posture.STANDING if person.box.aspect_ratio > self.STANDING_BOX_RATIO else Posture.SITTING

    def apply(self, people: list[Person]) -> None:
        """Set the ``posture`` attribute of every person in place."""
        for person in people:
            person.posture = self.classify(person)
