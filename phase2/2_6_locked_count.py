"""2.6 Stabile Gästezahl: Statt der Rohzahl pro Bild nehmen wir den häufigsten Wert
der letzten `lock_window_sec` Sekunden. Die angezeigte ('locked') Zahl ändert
sich erst, wenn ein neuer Wert so lange stabil bleibt."""
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
        c = STATE_COLOR[t.state]
        draw_polygon(frame, t.polygon, c)
        put_label(frame, f"{t.id}: {t.guests} Gaeste (roh {t.raw})", t.polygon[0], c)
    draw_people(frame, people)
    draw_panel(frame, [f"Gaeste (stabil): {sum(t.guests for t in tables)}",
                       f"Gaeste (roh): {sum(t.raw for t in tables)}"])
    if not show("2.6 Locked Count", frame):
        break
src.release()
