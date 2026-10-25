"""2.2 Tischbelegung: Tische als Polygone (aus config.json, siehe tools/mark_tables.py).
Liegt der Referenzpunkt einer Person im Polygon, gehört sie zu diesem Tisch."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rcv.detection import detect_people, load_model, remove_duplicates  # noqa: E402
from rcv.render import draw_people  # noqa: E402
from rcv.tables import assign_tables  # noqa: E402
from rcv.utils import GREEN, RED, VideoSource, draw_panel, draw_polygon, load_config, put_label, show  # noqa: E402

cfg = load_config()
model = load_model(cfg["model"])
src = VideoSource(cfg["source"], cfg["process_width"])

while (frame := src.read()) is not None:
    people = remove_duplicates(detect_people(model, frame, cfg["conf"]), cfg["dedupe_iou"], cfg["dedupe_contain"])
    per_table = assign_tables(people, cfg["tables"])
    busy = 0
    for t in cfg["tables"]:
        occupied = len(per_table[t["id"]]) > 0
        busy += occupied
        color = RED if occupied else GREEN
        draw_polygon(frame, t["polygon"], color)
        put_label(frame, f'{t["id"]} {"belegt" if occupied else "frei"}', t["polygon"][0], color)
    draw_people(frame, people)
    draw_panel(frame, [f"Tische belegt: {busy}/{len(cfg['tables'])}"])
    if not show("2.2 Table Occupancy", frame):
        break
src.release()
