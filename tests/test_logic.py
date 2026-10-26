"""Unit tests for the core logic – no YOLO / PyTorch required.

Run with ``python main.py test``, ``python tests/test_logic.py`` or ``pytest``.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rcv import (  # noqa: E402
    Box, Config, ConfigError, CountSmoother, DuplicateFilter, Person, Polygon, Posture,
    PostureClassifier, RestaurantAnalyzer, Table, TableConfig, TableManager, TableState,
    format_duration,
)


# ---------------------------------------------------------------------- helpers
def make_keypoints(points: dict[int, tuple[float, float]]) -> np.ndarray:
    """Build a (17, 3) keypoint array; listed points are visible, all others hidden."""
    kps = np.zeros((17, 3))
    for index, (x, y) in points.items():
        kps[index] = (x, y, 0.9)
    return kps


def square_table(table_id="T1", seats=4, size=300) -> TableConfig:
    return TableConfig(table_id, seats, [[0, 0], [size, 0], [size, size], [0, size]])


class FakeDetector:
    """Stands in for PersonDetector and returns a fixed list of people."""

    def __init__(self, people):
        self.people = people

    def detect(self, _frame):
        return list(self.people)


# ---------------------------------------------------------------------- geometry
def test_box_metrics():
    a, b = Box(0, 0, 10, 10), Box(5, 0, 15, 10)
    assert a.area == 100 and a.intersection(b) == 50
    assert abs(a.iou(b) - 50 / 150) < 1e-9
    assert Box(2, 2, 4, 4).inside_ratio(a) == 1.0


def test_polygon_contains():
    poly = Polygon([[0, 0], [10, 0], [10, 10], [0, 10]])
    assert poly.contains((5, 5)) and not poly.contains((20, 5))
    assert poly.centroid == (5.0, 5.0)


# ---------------------------------------------------------------------- detection
def test_duplicate_filter():
    a = Person(Box(100, 100, 200, 400), 0.9)
    b = Person(Box(105, 102, 198, 395), 0.6)   # almost the same box
    c = Person(Box(110, 100, 190, 220), 0.5)   # upper body inside a
    d = Person(Box(500, 100, 600, 400), 0.8)   # a different person
    assert DuplicateFilter().apply([a, b, c, d]) == [a, d]


def test_posture_classifier():
    clf = PostureClassifier()
    sitting = Person(Box(0, 0, 100, 160), 0.9, make_keypoints(
        {5: (30, 20), 6: (70, 20), 11: (35, 100), 12: (65, 100), 13: (20, 115), 14: (80, 115)}))
    standing = Person(Box(0, 0, 60, 300), 0.9, make_keypoints(
        {5: (20, 40), 6: (40, 40), 11: (22, 140), 12: (38, 140),
         13: (22, 210), 14: (38, 210), 15: (22, 290), 16: (38, 290)}))
    no_pose = Person(Box(0, 0, 60, 300), 0.9)
    assert clf.classify(sitting) is Posture.SITTING
    assert clf.classify(standing) is Posture.STANDING
    assert clf.classify(no_pose) is Posture.UNKNOWN


# ---------------------------------------------------------------------- tables
def test_table_manager_assign_ignores_standing():
    cfg = Config(tables=[square_table()])
    manager = TableManager.from_config(cfg)
    inside = Person(Box(100, 50, 200, 250), 0.9)
    outside = Person(Box(500, 50, 600, 250), 0.9)
    waiter = Person(Box(100, 50, 200, 250), 0.9, posture=Posture.STANDING)
    result = manager.assign([inside, outside, waiter], seated_only=True)
    assert result["T1"] == [inside]
    assert inside.table_id == "T1" and outside.table_id is None and waiter.table_id is None


def test_smoother_ignores_flicker():
    smoother = CountSmoother(window=2.0)
    smoother.force(3)
    t = 0.0
    for i in range(40):            # 4 s, one person missing in every 5th frame
        t += 0.1
        smoother.add(t, 2 if i % 5 == 0 else 3)
    assert smoother.locked == 3
    for _ in range(40):            # permanently 2 → accepted after one window
        t += 0.1
        smoother.add(t, 2)
    assert smoother.locked == 2


def test_table_lifecycle():
    table = Table(square_table(), occupy_after=3, free_after=5, lock_window=2)
    clock = 0.0

    def run(seconds, count):
        nonlocal clock
        visits = []
        for _ in range(int(seconds * 10)):
            clock += 0.1
            visit = table.update(clock, count)
            if visit:
                visits.append(visit)
        return visits

    run(1, 1)
    assert table.state is TableState.PENDING       # just walking past
    run(2, 0)
    assert table.state is TableState.FREE
    run(4, 2)
    assert table.state is TableState.OCCUPIED and table.guests == 2
    run(60, 2)
    run(2, 0)                                       # briefly hidden → still occupied
    assert table.is_occupied
    run(30, 2)
    visits = run(6, 0)
    assert table.state is TableState.FREE and len(visits) == 1
    assert visits[0].guests == 2 and 95 < visits[0].duration < 100


# ---------------------------------------------------------------------- pipeline
def test_analyzer_pipeline():
    cfg = Config(tables=[square_table(seats=4)], occupy_after_sec=1, free_after_sec=1)
    guests = [Person(Box(50, 50, 120, 250), 0.9), Person(Box(52, 52, 118, 248), 0.6),
              Person(Box(180, 60, 250, 260), 0.8)]
    analyzer = RestaurantAnalyzer(cfg, detector=FakeDetector(guests))
    frame = np.zeros((720, 1280, 3), np.uint8)
    for i in range(30):
        result = analyzer.process(frame, i * 0.1)
    assert len(result.people) == 2                  # duplicate removed
    stats = analyzer.summary(result)
    assert stats["guests"] == 2 and stats["tables_occupied"] == 1 and stats["capacity"] == 0.5


# ---------------------------------------------------------------------- config
def test_config_roundtrip_and_validation():
    cfg = Config(tables=[square_table()])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.json"
        cfg.save(path)
        assert Config.load(path) == cfg
        assert json.loads(path.read_text())["tables"][0]["id"] == "T1"
    cfg.update({"conf": "0.5", "tables": [{"seats": 2, "polygon": [[0, 0], [1, 0], [1, 1]]}]})
    assert cfg.conf == 0.5 and cfg.tables[0].id == "T1" and cfg.total_seats == 2
    try:
        cfg.update({"conf": "abc"})
        raise AssertionError("ConfigError expected")
    except ConfigError:
        pass


def test_format_duration():
    assert format_duration(65) == "1:05" and format_duration(3725) == "1:02:05"


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in list(globals().items()) if name.startswith("test_")]
    for name, fn in tests:
        fn()
        print("OK ", name)
    print(f"\n{len(tests)} Tests bestanden.")
