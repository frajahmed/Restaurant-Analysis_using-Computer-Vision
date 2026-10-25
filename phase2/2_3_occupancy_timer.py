"""2.3 Belegungs-Timer: Ein Tisch gilt erst als belegt, wenn dort `occupy_after_sec`
lang jemand ist (Vorbeilaufen zählt nicht), und erst als frei, wenn er
`free_after_sec` leer war (kurz verdeckt zählt nicht).

  frei ──Person──▶ wartet ──3 s──▶ belegt ──leer──▶ geht? ──6 s──▶ frei
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rcv.detection import detect_people, load_model, remove_duplicates  # noqa: E402
from rcv.render import draw_people, draw_tables  # noqa: E402
from rcv.tables import assign_tables, build_tables  # noqa: E402
from rcv.utils import VideoSource, draw_panel, load_config, show  # noqa: E402

cfg = load_config()
model = load_model(cfg["model"])
src = VideoSource(cfg["source"], cfg["process_width"])
tables = build_tables(cfg)

while (frame := src.read()) is not None:
    now = src.video_time()
    people = remove_duplicates(detect_people(model, frame, cfg["conf"]), cfg["dedupe_iou"], cfg["dedupe_contain"])
    per_table = assign_tables(people, cfg["tables"])
    for t in tables:
        t.update(now, len(per_table[t.id]))
    draw_tables(frame, tables, now, show_guests=False)
    draw_people(frame, people)
    draw_panel(frame, [f"Tische belegt: {sum(t.occupied for t in tables)}/{len(tables)}"])
    if not show("2.3 Occupancy Timer", frame):
        break
src.release()
