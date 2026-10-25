"""2.4 Stehende ignorieren: Kellner, die am Tisch stehen, sollen ihn nicht belegen.
Wir wechseln auf das Pose-Modell (findet Personen UND 17 Körperpunkte) und
prüfen, ob der Oberschenkel eher waagerecht (sitzt) oder senkrecht (steht) ist."""
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rcv.detection import classify_posture, detect_people, load_model, remove_duplicates  # noqa: E402
from rcv.render import draw_people, draw_tables  # noqa: E402
from rcv.tables import assign_tables, build_tables  # noqa: E402
from rcv.utils import VideoSource, draw_panel, load_config, show  # noqa: E402

SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16)]

cfg = load_config()
model = load_model(cfg["pose_model"])
src = VideoSource(cfg["source"], cfg["process_width"])
tables = build_tables(cfg)

while (frame := src.read()) is not None:
    now = src.video_time()
    people = remove_duplicates(detect_people(model, frame, cfg["conf"]), cfg["dedupe_iou"], cfg["dedupe_contain"])
    for p in people:
        p.posture = classify_posture(p, cfg["sit_ratio"])
        if p.keypoints is not None:
            for a, b in SKELETON:
                if p.keypoints[a][2] > 0.35 and p.keypoints[b][2] > 0.35:
                    cv2.line(frame, tuple(map(int, p.keypoints[a][:2])), tuple(map(int, p.keypoints[b][:2])),
                             (255, 200, 0), 2)
    per_table = assign_tables(people, cfg["tables"], only_sitting=True)
    for t in tables:
        t.update(now, len(per_table[t.id]))
    draw_tables(frame, tables, now, show_guests=False)
    draw_people(frame, people)
    sitting = sum(p.posture != "standing" for p in people)
    draw_panel(frame, [f"Sitzend: {sitting}", f"Stehend: {len(people) - sitting}"])
    if not show("2.4 Standing Filter", frame):
        break
src.release()
