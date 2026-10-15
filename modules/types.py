# modules/types.py
# M0: shared data types passed between modules.
# Coordinates are normalised to [0, 1] (x by frame width, y by frame height)
# so thresholds do not depend on video resolution.

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np


@dataclass
class Frame:
    index: int              # index in the source video
    t: float                # seconds since start of video
    ts: datetime            # real wall-clock time of this frame
    image: np.ndarray       # BGR, lives in RAM only


@dataclass
class Hand:
    label: str                  # "Left" / "Right" (after FLIP_HANDEDNESS)
    landmarks: np.ndarray       # (21, 2) normalised x, y
    score: float

    @property
    def wrist(self) -> np.ndarray:
        return self.landmarks[0]

    @property
    def box(self) -> tuple[float, float, float, float]:
        x, y = self.landmarks[:, 0], self.landmarks[:, 1]
        return float(x.min()), float(y.min()), float(x.max()), float(y.max())

    @property
    def center(self) -> np.ndarray:
        x1, y1, x2, y2 = self.box
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

    @property
    def size(self) -> float:
        """Hand scale: diagonal of the landmark box (normalised units)."""
        x1, y1, x2, y2 = self.box
        return float(max(np.hypot(x2 - x1, y2 - y1), 1e-3))


@dataclass
class Detection:
    obj: str                                    # canonical object name, e.g. "wallet"
    conf: float
    box: tuple[float, float, float, float]      # normalised x1, y1, x2, y2

    @property
    def center(self) -> np.ndarray:
        x1, y1, x2, y2 = self.box
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])


@dataclass
class FrameObs:
    """Everything perception saw in one frame (no pixels)."""
    t: float
    hands: list[Hand] = field(default_factory=list)
    detections: list[Detection] = field(default_factory=list)
    ego: np.ndarray = field(default_factory=lambda: np.zeros(2))   # camera motion (dx, dy) normalised


@dataclass
class Event:
    """Brief's event contract: Event(ts, object, action, landmark|None, confidence, note|None)."""
    ts: datetime
    object: str
    action: str                     # "pickup" | "putdown"
    landmark: str | None = None
    confidence: float = 0.0
    note: str | None = None
    t: float = 0.0                  # seconds since start of video (for evaluation)
