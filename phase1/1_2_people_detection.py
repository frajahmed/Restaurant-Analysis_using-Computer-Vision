"""1.2 Personenerkennung: nur Klasse 0 ('person') und nur ab einer Mindest-Konfidenz."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ultralytics import YOLO  # noqa: E402
from rcv.utils import VideoSource, draw_box, load_config, show  # noqa: E402

cfg = load_config()
model = YOLO(cfg["model"])
src = VideoSource(cfg["source"], cfg["process_width"])

while (frame := src.read()) is not None:
    res = model(frame, classes=[0], conf=cfg["conf"], verbose=False)[0]
    for box, conf in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist()):
        draw_box(frame, box, label=f"Person {conf:.2f}")
    if not show("1.2 People Detection", frame):
        break
src.release()
