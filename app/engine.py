"""Analyse-Engine: läuft im Hintergrund-Thread, verarbeitet die Videoquelle
und stellt das letzte Bild (JPEG) sowie die Live-Zahlen bereit."""
from __future__ import annotations

import threading
import time

import cv2

from rcv.detection import classify_posture, detect_people, load_model, remove_duplicates
from rcv.render import draw_people, draw_tables
from rcv.tables import assign_tables, build_tables
from rcv.utils import VideoSource, load_config

from .db import Database


class Engine:
    SNAPSHOT_EVERY = 60  # Sekunden zwischen Datenbank-Momentaufnahmen

    def __init__(self, db: Database):
        self.db = db
        self.lock = threading.Lock()
        self.jpeg: bytes | None = None
        self.raw_jpeg: bytes | None = None
        self.stats: dict = {"running": False, "error": None}
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._models: dict = {}

    # ------------------------------------------------------------- Steuerung
    def start(self):
        self.stop()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread and self._thread.is_alive():
            self._stop.set()
            self._thread.join(timeout=5)

    def restart(self):
        self.start()

    def _model(self, name):
        if name not in self._models:
            self._models[name] = load_model(name)
        return self._models[name]

    # ------------------------------------------------------------- Schleife
    def _run(self):
        try:
            cfg = load_config()
            model = self._model(cfg["pose_model"])
            src = VideoSource(cfg["source"], cfg.get("process_width", 1280))
        except Exception as e:  # noqa: BLE001
            with self.lock:
                self.stats = {"running": False, "error": str(e)}
            print("[Engine] Fehler:", e)
            return

        tables = build_tables(cfg)
        seats_total = cfg.get("seats_total") or sum(t.seats for t in tables)
        last_snapshot = 0.0
        fps, t_prev = 0.0, time.time()

        while not self._stop.is_set():
            frame = src.read()
            if frame is None:
                time.sleep(0.5)
                continue
            vt, wall = src.video_time(), time.time()

            people = remove_duplicates(detect_people(model, frame, cfg["conf"]),
                                       cfg["dedupe_iou"], cfg["dedupe_contain"])
            for p in people:
                p.posture = classify_posture(p, cfg["sit_ratio"])
            per_table = assign_tables(people, cfg["tables"], only_sitting=True)

            for t in tables:
                v = t.update(vt, len(per_table[t.id]))
                if v:  # Videozeit → Uhrzeit umrechnen
                    self.db.add_visit(v["table"], wall - (vt - v["start"]), wall - (vt - v["end"]), v["guests"])

            raw = frame.copy()
            draw_tables(frame, tables, vt)
            draw_people(frame, people)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1 / max(1e-6, now - t_prev))
            t_prev = now

            guests = sum(t.guests for t in tables if t.occupied)
            occupied = sum(t.occupied for t in tables)
            if wall - last_snapshot >= self.SNAPSHOT_EVERY:
                self.db.add_snapshot(wall, guests, occupied, len(tables))
                last_snapshot = wall

            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            ok2, rjpg = cv2.imencode(".jpg", raw, [cv2.IMWRITE_JPEG_QUALITY, 85])
            with self.lock:
                if ok:
                    self.jpeg = jpg.tobytes()
                if ok2:
                    self.raw_jpeg = rjpg.tobytes()
                self.stats = {
                    "running": True, "error": None, "ts": wall, "fps": round(fps, 1),
                    "frame_w": frame.shape[1], "frame_h": frame.shape[0],
                    "people_total": len(people),
                    "standing": sum(p.posture == "standing" for p in people),
                    "guests": guests, "seats_total": seats_total,
                    "capacity": round(guests / max(1, seats_total), 3),
                    "tables_occupied": occupied, "tables_total": len(tables),
                    "tables": [t.snapshot(vt) for t in tables],
                }

        src.release()
        with self.lock:
            self.stats["running"] = False

    # ------------------------------------------------------------- Zugriff
    def get_jpeg(self, raw=False):
        with self.lock:
            return self.raw_jpeg if raw else self.jpeg

    def get_stats(self):
        with self.lock:
            return dict(self.stats)
