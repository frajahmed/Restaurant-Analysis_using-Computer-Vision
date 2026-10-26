"""Video input: file, webcam or network stream behind one interface."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .config import ROOT


class VideoSource:
    """Reads frames from a video file, a webcam (``"0"``) or an RTSP/HTTP stream.

    * Files can loop forever (handy for demos).
    * Every frame is resized to ``width`` pixels so table coordinates stay
      valid regardless of the original resolution.

    Usable as a context manager and iterator::

        with VideoSource("videos/restaurant.mp4") as src:
            for frame in src:
                ...
    """

    def __init__(self, source: str, width: int | None = 1280, loop: bool = True):
        self.width = width
        self.loop = loop
        # Frames read since opening. Not reset when a file loops, so that
        # `timestamp` keeps increasing and timers never jump backwards.
        self.frame_index = 0

        self._source = source
        self._target = self._resolve(source)
        self.is_file = isinstance(self._target, str) and not self._target.startswith(("rtsp", "http"))
        self._cap = self._open()
        self.fps: float = self._cap.get(cv2.CAP_PROP_FPS) or 25.0

    # ------------------------------------------------------------------ internal helpers
    @staticmethod
    def _resolve(source: str) -> int | str:
        """Turn the config value into something OpenCV can open."""
        source = str(source).strip()
        if source.isdigit():                       # webcam index
            return int(source)
        if source.startswith(("rtsp", "http")):    # network stream
            return source
        path = Path(source)
        if not path.is_absolute():                 # relative to the project root
            path = ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"Video nicht gefunden: {path}")
        return str(path)

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self._target)
        if not cap.isOpened():
            raise RuntimeError(f"Videoquelle kann nicht geöffnet werden: {self._source}")
        return cap

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        if not self.width or frame.shape[1] == self.width:
            return frame
        height = int(frame.shape[0] * self.width / frame.shape[1])
        return cv2.resize(frame, (self.width, height))

    # ------------------------------------------------------------------ public API
    def read(self) -> np.ndarray | None:
        """Return the next frame, or ``None`` when the source has ended."""
        ok, frame = self._cap.read()
        if not ok and self.is_file and self.loop:
            # End of file reached → rewind to the start (frame_index keeps counting).
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
        if not ok:
            return None
        self.frame_index += 1
        return self._resize(frame)

    @property
    def timestamp(self) -> float:
        """Time in seconds: video time for files, wall-clock time for live sources.

        Using video time keeps timers correct for files even when processing
        runs slower or faster than real time.
        """
        return self.frame_index / self.fps if self.is_file else time.time()

    def reconnect(self) -> bool:
        """Re-open the source, e.g. after a network camera dropped out.

        Returns:
            ``True`` if the source could be opened again.
        """
        self._cap.release()
        try:
            self._cap = self._open()
            return True
        except RuntimeError:
            return False

    def release(self) -> None:
        """Close the video source."""
        self._cap.release()

    def __iter__(self) -> Iterator[np.ndarray]:
        while (frame := self.read()) is not None:
            yield frame

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, *_exc) -> None:
        self.release()
