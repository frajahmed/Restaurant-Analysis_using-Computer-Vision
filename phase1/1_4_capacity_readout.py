"""1.4 Auslastung: Personen im Verhältnis zur Anzahl Sitzplätze, mit Farbbalken."""
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rcv.detection import detect_people, load_model  # noqa: E402
from rcv.utils import GREEN, RED, YELLOW, VideoSource, draw_box, draw_panel, load_config, show  # noqa: E402

cfg = load_config()
seats = cfg["seats_total"]
model = load_model(cfg["model"])
src = VideoSource(cfg["source"], cfg["process_width"])

while (frame := src.read()) is not None:
    people = detect_people(model, frame, cfg["conf"])
    for p in people:
        draw_box(frame, p.box)
    load = len(people) / max(1, seats)
    color = GREEN if load < 0.6 else YELLOW if load < 0.9 else RED
    draw_panel(frame, [f"Gaeste: {len(people)} / {seats}", f"Auslastung: {load:.0%}"])
    # Balken
    x, y, w, h = 12, 80, 260, 14
    cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 60, 60), -1)
    cv2.rectangle(frame, (x, y), (x + int(w * min(load, 1)), y + h), color, -1)
    if not show("1.4 Capacity Readout", frame):
        break
src.release()
