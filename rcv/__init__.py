"""rcv – Restaurant Computer Vision.

Reusable building blocks shared by the learning steps (``phase1`` – ``phase3``)
and the dashboard (``app``):

==================  =====================================================
Module              Contents
==================  =====================================================
``config``          :class:`Config`, :class:`TableConfig`
``geometry``        :class:`Box`, :class:`Polygon`
``video``           :class:`VideoSource`
``detection``       :class:`PersonDetector`, :class:`DuplicateFilter`,
                    :class:`PostureClassifier`, :class:`Person`
``tables``          :class:`Table`, :class:`TableManager`,
                    :class:`CountSmoother`, :class:`Visit`
``analyzer``        :class:`RestaurantAnalyzer` (the full pipeline)
``render``          :class:`Renderer`, :class:`Window`
==================  =====================================================
"""
from .analyzer import FrameResult, RestaurantAnalyzer
from .config import CONFIG_PATH, ROOT, Config, ConfigError, TableConfig
from .detection import DuplicateFilter, Keypoint, Person, PersonDetector, Posture, PostureClassifier
from .geometry import Box, Polygon
from .render import Color, Renderer, Window, format_duration
from .tables import CountSmoother, Table, TableManager, TableState, Visit
from .video import VideoSource

__all__ = [
    "ROOT", "CONFIG_PATH", "Config", "ConfigError", "TableConfig",
    "Box", "Polygon", "VideoSource",
    "Keypoint", "Person", "Posture", "PersonDetector", "DuplicateFilter", "PostureClassifier",
    "CountSmoother", "Table", "TableManager", "TableState", "Visit",
    "FrameResult", "RestaurantAnalyzer",
    "Color", "Renderer", "Window", "format_duration",
]
