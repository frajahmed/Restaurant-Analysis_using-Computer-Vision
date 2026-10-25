"""Hilfsfunktionen: Config, Videoquelle, Geometrie, Zeichnen."""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"


# ---------------------------------------------------------------- Config
def load_config(path: Path | str = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(cfg: dict, path: Path | str = CONFIG_PATH) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------- Video
class VideoSource:
    """Liest eine Datei, eine Webcam (z.B. "0") oder einen RTSP-Stream.
    Dateien werden endlos wiederholt und skaliert auf process_width."""

    def __init__(self, source: str, width: int | None = 1280, loop: bool = True):
        src = int(source) if str(source).isdigit() else source
        if isinstance(src, str) and not src.startswith(("rtsp", "http")):
            p = Path(src)
            if not p.is_absolute():
                p = ROOT / p
            if not p.exists():
                raise FileNotFoundError(f"Video nicht gefunden: {p}")
            src = str(p)
        self.src = src
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Videoquelle kann nicht geöffnet werden: {source}")
        self.width = width
        self.loop = loop
        self.is_file = isinstance(src, str) and not src.startswith(("rtsp", "http"))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.frame_idx = 0

    def read(self):
        ok, frame = self.cap.read()
        if not ok and self.is_file and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame_idx = 0
            ok, frame = self.cap.read()
        if not ok:
            return None
        self.frame_idx += 1
        return resize_to_width(frame, self.width)

    def video_time(self) -> float:
        """Zeit im Video (Sek.) – für Dateien die Videozeit, sonst Echtzeit."""
        return self.frame_idx / self.fps if self.is_file else time.time()

    def release(self):
        self.cap.release()


def resize_to_width(frame, width):
    if not width or frame.shape[1] == width:
        return frame
    h = int(frame.shape[0] * width / frame.shape[1])
    return cv2.resize(frame, (width, h))


# ---------------------------------------------------------------- Geometrie
def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def containment(small, big) -> float:
    """Anteil von `small`, der in `big` liegt."""
    x1, y1 = max(small[0], big[0]), max(small[1], big[1])
    x2, y2 = min(small[2], big[2]), min(small[3], big[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area = max(1e-6, (small[2] - small[0]) * (small[3] - small[1]))
    return inter / area


def point_in_polygon(pt, polygon) -> bool:
    poly = np.array(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False) >= 0


def anchor_point(box):
    """Referenzpunkt einer Person: Mitte, 70 % nach unten (≈ Hüfte/Sitzfläche)."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, y1 + 0.7 * (y2 - y1))


# ---------------------------------------------------------------- Zeichnen
GREEN = (80, 200, 120)
RED = (60, 60, 230)
YELLOW = (0, 200, 255)
GREY = (160, 160, 160)
WHITE = (255, 255, 255)


def draw_box(img, box, color=GREEN, label: str | None = None):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    if label:
        put_label(img, label, (x1, y1 - 6), color)


def put_label(img, text, org, color=GREEN, scale=0.55):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = int(org[0]), int(max(org[1], h + 4))
    cv2.rectangle(img, (x, y - h - 4), (x + w + 6, y + 4), color, -1)
    cv2.putText(img, text, (x + 3, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), 1, cv2.LINE_AA)


def draw_polygon(img, polygon, color, alpha=0.25):
    pts = np.array(polygon, dtype=np.int32)
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.polylines(img, [pts], True, color, 2)


def draw_panel(img, lines, org=(12, 12)):
    """Halbtransparentes Info-Panel oben links."""
    pad, lh = 10, 26
    w = max(cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0][0] for l in lines) + 2 * pad
    h = lh * len(lines) + pad
    x, y = org
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (x + pad, y + pad + 16 + i * lh),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 2, cv2.LINE_AA)


def fmt_duration(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600:d}:{sec % 3600 // 60:02d}:{sec % 60:02d}" if sec >= 3600 else f"{sec // 60:d}:{sec % 60:02d}"


def show(win, frame) -> bool:
    """Zeigt das Bild; gibt False zurück, wenn q/ESC gedrückt wurde."""
    cv2.imshow(win, frame)
    return (cv2.waitKey(1) & 0xFF) not in (ord("q"), 27)
