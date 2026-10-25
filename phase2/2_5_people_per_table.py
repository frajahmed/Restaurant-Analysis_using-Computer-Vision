"""2.5 Gäste pro Tisch: Anzahl sitzender Personen im Tisch-Polygon (noch ungeglättet –
die Zahl springt, sobald jemand kurz verdeckt ist. Das lösen wir in 2.6)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rcv.detection import classify_posture, detect_people, load_model, remove_duplicates  # noqa: E402
from rcv.render import STATE_COLOR, draw_people  # noqa: E402
from rcv.tables import assign_tables, build_tables  # noqa: E402
from rcv.utils import VideoSource, draw_panel, draw_polygon, load_config, put_label, show  # noqa: E402

cfg = load_config()
model = load_model(cfg["pose_model"])
src = VideoSource(cfg["source"], cfg["process_width"])
tables = build_tables(cfg)

while (frame := src.read()) is not None:
    now = src.video_time()
    people = remove_duplicates(detect_people(model, frame, cfg["conf"]), cfg["dedupe_iou"], cfg["dedupe_contain"])
    for p in people:
        p.posture = classify_posture(p, cfg["sit_ratio"])
    per_table = assign_tables(people, cfg["tables"], only_sitting=True)
    for t in tables:
        t.update(now, len(per_table[t.id]))
        draw_polygon(frame, t.polygon, STATE_COLOR[t.state])
        put_label(frame, f"{t.id}: {t.raw}/{t.seats} Gaeste", t.polygon[0], STATE_COLOR[t.state])
    draw_people(frame, people)
    draw_panel(frame, [f"Gaeste an Tischen: {sum(t.raw for t in tables)}"])
    if not show("2.5 People Per Table", frame):
        break
src.release()
