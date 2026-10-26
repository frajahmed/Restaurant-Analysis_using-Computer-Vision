"""Helpers for streaming processed video to a browser (phase 3 and the dashboard)."""
from __future__ import annotations

import asyncio
import threading
from typing import AsyncIterator, Callable

import cv2
import numpy as np

#: Content type of a Motion-JPEG stream. Browsers show it directly in an ``<img>`` tag.
MJPEG_MEDIA_TYPE = "multipart/x-mixed-replace; boundary=frame"


def encode_jpeg(frame: np.ndarray, quality: int = 80) -> bytes | None:
    """Encode a BGR image as JPEG bytes, or return ``None`` on failure."""
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buffer.tobytes() if ok else None


class FrameBuffer:
    """Thread-safe holder for the latest JPEG frame.

    The processing thread calls :meth:`put`; every connected browser reads
    the newest frame through :meth:`mjpeg_stream`. Old frames are simply
    replaced, so slow clients never slow down the analysis.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None

    def put(self, jpeg: bytes | None) -> None:
        if jpeg is not None:
            with self._lock:
                self._jpeg = jpeg

    def get(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    async def mjpeg_stream(self, interval: float = 0.04) -> AsyncIterator[bytes]:
        """Yield an endless MJPEG stream; each new frame is sent exactly once."""
        last: bytes | None = None
        while True:
            jpeg = self.get()
            if jpeg is not None and jpeg is not last:
                last = jpeg
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            await asyncio.sleep(interval)


class BackgroundWorker:
    """Runs a function repeatedly in a daemon thread until :meth:`stop` is called.

    The function receives a :class:`threading.Event`; it should return
    (or check the event regularly) once the event is set.
    """

    def __init__(self, target: Callable[[threading.Event], None], name: str = "worker"):
        self._target = target
        self._name = name
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start the thread (stops a running one first)."""
        self.stop()
        self._stop.clear()
        self._thread = threading.Thread(target=self._target, args=(self._stop,),
                                        name=self._name, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Ask the thread to stop and wait for it."""
        if self.is_running:
            self._stop.set()
            self._thread.join(timeout=timeout)
