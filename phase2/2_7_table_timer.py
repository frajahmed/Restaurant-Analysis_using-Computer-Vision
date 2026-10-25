"""2.7 Tisch-Timer: Jeder Besuch bekommt Start, Ende, Dauer und Gästezahl.
Beendete Besuche werden in der Konsole und in visits.csv protokolliert."""
import csv
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rcv.detection import classify_posture, detect_people, load_model, remove_duplicates  # noqa: E402
from rcv.render import draw_people, draw_tables  # noqa: E402
from rcv.tables import assign_tables, build_tables  # noqa: E402
from rcv.utils import ROOT, VideoSource, draw_panel, fmt_duration, load_config, show  # noqa: E402

cfg = load_config()
model = load_model(cfg["pose_model"])
src = VideoSource(cfg["source"], cfg["process_width"])
tables = build_tables(cfg)
visits = []

log = open(ROOT / "visits.csv", "w", newline="", encoding="utf-8")
writer = csv.writer(log)
writer.writerow(["tisch", "start", "ende", "dauer_sek", "gaeste"])

while (frame := src.read()) is not None:
    now = src.video_time()
    people = remove_duplicates(detect_people(model, frame, cfg["conf"]), cfg["dedupe_iou"], cfg["dedupe_contain"])
    for p in people:
        p.posture = classify_posture(p, cfg["sit_ratio"])
    per_table = assign_tables(people, cfg["tables"], only_sitting=True)
    for t in tables:
        v = t.update(now, len(per_table[t.id]))
        if v:
            visits.append(v)
            print(f"[Besuch] {v['table']}: {v['guests']} Gäste, {fmt_duration(v['duration'])}")
            writer.writerow([v["table"], str(timedelta(seconds=int(v["start"]))),
                             str(timedelta(seconds=int(v["end"]))), round(v["duration"], 1), v["guests"]])
            log.flush()
    draw_tables(frame, tables, now)
    draw_people(frame, people)
    avg = sum(v["duration"] for v in visits) / len(visits) if visits else 0
    draw_panel(frame, [
        f"Gaeste: {sum(t.guests for t in tables)} / {cfg['seats_total']}",
        f"Tische belegt: {sum(t.occupied for t in tables)}/{len(tables)}",
        f"Besuche: {len(visits)}  Schnitt: {fmt_duration(avg)}",
    ])
    if not show("2.7 Table Timer", frame):
        break
src.release()
log.close()
