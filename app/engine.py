"""Analysis engine for the dashboard.

Runs the :class:`~rcv.analyzer.RestaurantAnalyzer` in a background thread,
keeps the latest frame and figures ready for the web server and writes
finished visits and regular snapshots to the database.
"""
from __future__ import annotations

import threading
import time
from typing import Any

from rcv.analyzer import RestaurantAnalyzer
from rcv.config import CONFIG_PATH, Config
from rcv.detection import Detector, PersonDetector
from rcv.render import Renderer
from rcv.streaming import BackgroundWorker, FrameBuffer, encode_jpeg
from rcv.tables import Visit
from rcv.video import VideoSource

from .db import Database


class Engine:
    """Owns the processing thread and exposes thread-safe read access to its results."""

    SNAPSHOT_INTERVAL = 60.0  # seconds between occupancy snapshots in the database
    RECONNECT_AFTER = 10      # failed reads in a row before a live source is re-opened

    def __init__(self, database: Database, config_path=CONFIG_PATH):
        self.db = database
        self.config_path = config_path
        self.annotated = FrameBuffer()   # frame with boxes and tables (live view)
        self.raw = FrameBuffer()         # clean frame (background for the table editor)

        self._worker = BackgroundWorker(self._run, name="analysis")
        self._stats_lock = threading.Lock()
        self._stats: dict[str, Any] = {"running": False, "error": None}
        self._detectors: dict[tuple[str, float], Detector] = {}   # cache: model loads are slow

    # ------------------------------------------------------------------ control
    def start(self) -> None:
        """(Re)start the analysis with the current configuration file."""
        self._worker.start()

    def stop(self) -> None:
        self._worker.stop()

    def restart(self) -> None:
        self.start()

    # ------------------------------------------------------------------ read access
    @property
    def stats(self) -> dict[str, Any]:
        with self._stats_lock:
            return dict(self._stats)

    def _set_stats(self, stats: dict[str, Any]) -> None:
        with self._stats_lock:
            self._stats = stats

    # ------------------------------------------------------------------ processing
    def _detector(self, config: Config) -> Detector:
        """Return a cached detector so that a restart does not reload the model."""
        key = (config.pose_model, config.conf)
        if key not in self._detectors:
            self._detectors[key] = PersonDetector(config.pose_model, config.conf)
        return self._detectors[key]

    def _run(self, stop: threading.Event) -> None:
        """Thread body: read → analyse → draw → publish, until ``stop`` is set."""
        try:
            config = Config.load(self.config_path)
            analyzer = RestaurantAnalyzer(config, detector=self._detector(config))
            source = VideoSource(config.source, config.process_width)
        except Exception as exc:  # shown in the dashboard instead of crashing the server
            self._set_stats({"running": False, "error": str(exc)})
            print(f"[Engine] Fehler: {exc}")
            return

        renderer = Renderer()
        last_snapshot = 0.0
        fps, previous = 0.0, time.time()
        failed_reads = 0

        with source:
            while not stop.is_set():
                frame = source.read()
                if frame is None:
                    # Live stream interrupted (or a non-looping file ended): wait, then
                    # re-open the source after several failures in a row.
                    failed_reads += 1
                    if failed_reads >= self.RECONNECT_AFTER:
                        print("[Engine] Keine Bilder – Videoquelle wird neu verbunden …")
                        source.reconnect()
                        failed_reads = 0
                    time.sleep(0.5)
                    continue
                failed_reads = 0

                video_time, wall_time = source.timestamp, time.time()
                result = analyzer.process(frame, video_time)
                for visit in result.finished_visits:
                    self.db.add_visit(self._to_wall_clock(visit, video_time, wall_time))

                self.raw.put(encode_jpeg(frame, 85))
                renderer.tables(frame, analyzer.tables, video_time)
                renderer.people(frame, result.people)
                self.annotated.put(encode_jpeg(frame, 75))

                # Exponential moving average keeps the FPS display calm.
                now = time.time()
                fps = 0.9 * fps + 0.1 / max(1e-6, now - previous)
                previous = now

                summary = analyzer.summary(result)
                if wall_time - last_snapshot >= self.SNAPSHOT_INTERVAL:
                    self.db.add_snapshot(wall_time, summary["guests"],
                                         summary["tables_occupied"], summary["tables_total"])
                    last_snapshot = wall_time

                self._set_stats({**summary, "running": True, "error": None, "ts": wall_time,
                                 "fps": round(fps, 1), "frame_w": frame.shape[1], "frame_h": frame.shape[0]})

        with self._stats_lock:
            self._stats["running"] = False

    @staticmethod
    def _to_wall_clock(visit: Visit, video_time: float, wall_time: float) -> Visit:
        """Convert a visit from video time to Unix time so the database stores real clock times."""
        offset = wall_time - video_time
        return Visit(visit.table_id, visit.start + offset, visit.end + offset, visit.guests)
