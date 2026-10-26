"""Base class for the learning steps in ``phase1/`` and ``phase2/``.

Every step follows the same loop: read a frame → process it → show it.
:class:`VideoStep` implements that loop once; a step only overrides
:meth:`VideoStep.process` and therefore contains nothing but its own idea.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .config import Config
from .render import Renderer, Window
from .video import VideoSource


class VideoStep(ABC):
    """Runs a processing step on the configured video source and shows the result.

    Subclasses set :attr:`title`, create their helpers in ``__init__``
    (after calling ``super().__init__()``) and implement :meth:`process`.
    """

    title: str = "Restaurant CV"

    def __init__(self, config: Config | None = None):
        self.config = config or Config.load()
        self.renderer = Renderer()

    @abstractmethod
    def process(self, frame: np.ndarray, now: float) -> np.ndarray:
        """Process one frame and return the image to display."""

    def on_finish(self) -> None:
        """Optional hook, called once after the loop ends (e.g. to close files)."""

    def run(self) -> None:
        """Main loop. Ends when the video ends or q/ESC is pressed."""
        try:
            with VideoSource(self.config.source, self.config.process_width) as source, \
                    Window(self.title) as window:
                for frame in source:
                    if not window.show(self.process(frame, source.timestamp)):
                        break
        finally:
            self.on_finish()
