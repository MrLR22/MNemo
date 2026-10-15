# modules/video_source.py
# M1: read video (file or camera), sample at PROCESS_FPS, keep a RAM-only rolling buffer.
# Privacy (M9): frames are never written to disk here or anywhere in the runtime path.

from collections import deque
from datetime import datetime, timedelta

import cv2

from config import PROCESS_FPS, PROCESS_WIDTH, BUFFER_SECONDS
from modules.types import Frame


def get_frames(source, start_time: datetime | None = None, fps: float = PROCESS_FPS):
    """
    Yield Frame objects sampled at `fps`.

    source      video path, or an int camera index for live capture
    start_time  real time the recording started (recorded files). Defaults to now.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"Could not open video source: {source}")

    live = isinstance(source, int)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if start_time is None:
        start_time = datetime.now()
    # Read sequentially and skip with grab(): much faster and more exact than seeking.
    # Frames are picked by TIME (one every 1/fps s), not "every Nth frame": with N rounded, a 24 fps
    # video would give 12 fps, and different datasets would get different effective rates.
    period = 1.0 / fps
    next_t = 0.0

    index = 0
    try:
        while True:
            if not cap.grab():
                break
            if index / src_fps + 1e-6 >= next_t:
                next_t += period
                ok, image = cap.retrieve()
                if not ok:
                    break
                if image.shape[1] > PROCESS_WIDTH:
                    scale = PROCESS_WIDTH / image.shape[1]
                    image = cv2.resize(image, (PROCESS_WIDTH, round(image.shape[0] * scale)),
                                       interpolation=cv2.INTER_AREA)
                if live:
                    ts = datetime.now()
                    t = (ts - start_time).total_seconds()
                else:
                    t = index / src_fps
                    ts = start_time + timedelta(seconds=t)
                yield Frame(index=index, t=t, ts=ts, image=image)
            index += 1
    finally:
        cap.release()


class FrameBuffer:
    """Rolling RAM-only buffer of the last BUFFER_SECONDS of frames."""

    def __init__(self, seconds: float = BUFFER_SECONDS, fps: float = PROCESS_FPS):
        self.frames: deque[Frame] = deque(maxlen=max(1, int(seconds * fps)))

    def push(self, frame: Frame):
        self.frames.append(frame)

    def latest(self) -> Frame | None:
        return self.frames[-1] if self.frames else None

    def at(self, t: float) -> Frame | None:
        """Frame closest to video time t, if still in the buffer."""
        if not self.frames:
            return None
        return min(self.frames, key=lambda f: abs(f.t - t))

    def clear(self):
        self.frames.clear()
