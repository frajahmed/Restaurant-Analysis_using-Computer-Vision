"""Logik-Tests ohne YOLO/Torch:  python -m pytest tests  (oder python tests/test_logic.py)"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rcv.detection import Person, classify_posture, remove_duplicates  # noqa: E402
from rcv.tables import FREE, OCCUPIED, PENDING, CountSmoother, Table, assign_tables  # noqa: E402


def kp(points):
    """points: {index: (x, y)} → (17,3)-Array, Rest unsichtbar."""
    a = np.zeros((17, 3))
    for i, (x, y) in points.items():
        a[i] = (x, y, 0.9)
    return a


def test_remove_duplicates():
    a = Person((100, 100, 200, 400), 0.9)
    b = Person((105, 102, 198, 395), 0.6)   # fast gleiche Box
    c = Person((110, 100, 190, 220), 0.5)   # Oberkörper in a
    d = Person((500, 100, 600, 400), 0.8)   # andere Person
    kept = remove_duplicates([a, b, c, d])
    assert kept == [a, d]


def test_posture():
    sitting = Person((0, 0, 100, 160), 0.9, kp({5: (30, 20), 6: (70, 20), 11: (35, 100), 12: (65, 100),
                                                  13: (20, 115), 14: (80, 115)}))
    standing = Person((0, 0, 60, 300), 0.9, kp({5: (20, 40), 6: (40, 40), 11: (22, 140), 12: (38, 140),
                                                  13: (22, 210), 14: (38, 210), 15: (22, 290), 16: (38, 290)}))
    assert classify_posture(sitting) == "sitting"
    assert classify_posture(standing) == "standing"


def test_assign_tables():
    tables = [{"id": "T1", "seats": 4, "polygon": [[0, 0], [300, 0], [300, 300], [0, 300]]}]
    inside = Person((100, 50, 200, 250), 0.9)
    outside = Person((500, 50, 600, 250), 0.9)
    waiter = Person((100, 50, 200, 250), 0.9, posture="standing")
    res = assign_tables([inside, outside, waiter], tables, only_sitting=True)
    assert res["T1"] == [inside]


def test_smoother_ignores_flicker():
    s = CountSmoother(window=2.0)
    s.force(3)
    t = 0.0
    for i in range(40):  # 4 s, jedes 5. Bild fällt eine Person weg
        t += 0.1
        s.add(t, 2 if i % 5 == 0 else 3)
    assert s.locked == 3
    for _ in range(40):  # dauerhaft 2 → nach Fenster übernommen
        t += 0.1
        s.add(t, 2)
    assert s.locked == 2


def test_table_lifecycle():
    t = Table("T1", 4, [[0, 0], [1, 1], [0, 1]], occupy_after=3, free_after=5, lock_window=2)
    clock = 0.0

    def run(sec, count):
        nonlocal clock
        out = []
        for _ in range(int(sec * 10)):
            clock += 0.1
            v = t.update(clock, count)
            if v:
                out.append(v)
        return out

    run(1, 1)
    assert t.state == PENDING            # kurz vorbeigelaufen
    run(2, 0)
    assert t.state == FREE
    run(4, 2)
    assert t.state == OCCUPIED and t.guests == 2
    run(60, 2)
    run(2, 0)                             # kurz verdeckt → bleibt belegt
    assert t.occupied
    run(30, 2)
    visits = run(6, 0)
    assert t.state == FREE and len(visits) == 1
    v = visits[0]
    assert v["guests"] == 2 and 95 < v["duration"] < 100


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK ", name)
