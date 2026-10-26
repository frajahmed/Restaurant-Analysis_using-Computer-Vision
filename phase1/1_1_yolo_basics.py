"""Step 1.1 – YOLO basics: run YOLO11 on a video and show everything it finds.

Model sizes (fast → accurate): yolo11n, yolo11s, yolo11m, yolo11l, yolo11x

Usage:  python main.py 1.1 [model.pt]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rcv.step import VideoStep  # noqa: E402


class YoloBasics(VideoStep):
    """Shows all 80 COCO classes with the plotting built into Ultralytics."""

    title = "1.1 YOLO Basics"

    def __init__(self, model_name: str | None = None):
        super().__init__()
        from ultralytics import YOLO
        # Weights are downloaded automatically on first use.
        self.model = YOLO(model_name or self.config.model)

    def process(self, frame, now):
        started = time.perf_counter()
        result = self.model(frame, verbose=False)[0]          # no class filter yet
        output = result.plot()                                # boxes + labels drawn by Ultralytics
        fps = 1.0 / max(1e-6, time.perf_counter() - started)
        self.renderer.panel(output, [f"Objekte: {len(result.boxes)}", f"FPS: {fps:.1f}"])
        return output


if __name__ == "__main__":
    YoloBasics(sys.argv[1] if len(sys.argv) > 1 else None).run()
