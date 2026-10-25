"""1.3 Personen zählen: Anzahl Boxen = Anzahl Personen im Bild."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rcv.detection import detect_people, load_model  # noqa: E402
from rcv.utils import VideoSource, draw_box, draw_panel, load_config, show  # noqa: E402

cfg = load_config()
model = load_model(cfg["model"])
src = VideoSource(cfg["source"], cfg["process_width"])

while (frame := src.read()) is not None:
    people = detect_people(model, frame, cfg["conf"])
    for p in people:
        draw_box(frame, p.box)
    draw_panel(frame, [f"Personen: {len(people)}"])
    if not show("1.3 People Counting", frame):
        break
src.release()
