"""1.1 YOLO-Grundlagen: YOLO11 auf ein Video loslassen und alles anzeigen, was es findet.

Modellgrößen (schnell → genau):  yolo11n  yolo11s  yolo11m  yolo11l  yolo11x
Aufruf:  python phase1/1_1_yolo_basics.py [modell.pt]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ultralytics import YOLO  # noqa: E402
from rcv.utils import VideoSource, draw_panel, load_config, show  # noqa: E402

cfg = load_config()
model = YOLO(sys.argv[1] if len(sys.argv) > 1 else cfg["model"])  # lädt beim 1. Mal automatisch herunter
src = VideoSource(cfg["source"], cfg["process_width"])

while (frame := src.read()) is not None:
    t0 = time.time()
    result = model(frame, verbose=False)[0]      # alle 80 COCO-Klassen
    out = result.plot()                          # Ultralytics zeichnet Boxen + Labels
    fps = 1 / max(1e-6, time.time() - t0)
    draw_panel(out, [f"Objekte: {len(result.boxes)}", f"FPS: {fps:.1f}"])
    if not show("1.1 YOLO Basics", out):
        break
src.release()
