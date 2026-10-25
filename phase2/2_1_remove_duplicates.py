"""2.1 Duplikate entfernen: YOLO liefert manchmal zwei Boxen für eine Person
(z.B. ganze Person + nur Oberkörper). Grau = verworfen, Grün = behalten."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rcv.detection import detect_people, load_model, remove_duplicates  # noqa: E402
from rcv.utils import GREEN, GREY, VideoSource, draw_box, draw_panel, load_config, show  # noqa: E402

cfg = load_config()
model = load_model(cfg["model"])
src = VideoSource(cfg["source"], cfg["process_width"])

while (frame := src.read()) is not None:
    raw = detect_people(model, frame, cfg["conf"])
    kept = remove_duplicates(raw, cfg["dedupe_iou"], cfg["dedupe_contain"])
    for p in raw:
        if p not in kept:
            draw_box(frame, p.box, GREY, "doppelt")
    for p in kept:
        draw_box(frame, p.box, GREEN)
    draw_panel(frame, [f"Roh: {len(raw)}", f"Ohne Duplikate: {len(kept)}"])
    if not show("2.1 Remove Duplicates", frame):
        break
src.release()
