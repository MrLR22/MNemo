# M9 privacy guard: the runtime path must never write image or video files.
# Runs the real pipeline (MediaPipe + YOLO) on a generated video with every image-writing
# function patched to fail, and checks no image/video file appeared anywhere in the project.

import os

import cv2
import numpy as np
import pytest

import config as C

ROOT = C.ROOT
MEDIA_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".mp4", ".avi", ".mov", ".mkv", ".npy"}
SKIP_DIRS = {".git", ".mnemovenv", "legacy", "__pycache__", "data"}


def media_files(root):
    found = set()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for f in filenames:
            if os.path.splitext(f)[1].lower() in MEDIA_EXT:
                found.add(os.path.join(dirpath, f))
    return found


@pytest.fixture
def video(tmp_path):
    path = str(tmp_path / "synthetic.avi")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30, (320, 240))
    for i in range(90):
        im = np.full((240, 320, 3), 40, np.uint8)
        cv2.rectangle(im, (20 + i, 100), (80 + i, 160), (0, 200, 255), -1)
        w.write(im)
    w.release()
    return path


def test_runtime_writes_no_images(video, tmp_path, monkeypatch):
    from PIL import Image
    from modules.pipeline import MnemoPipeline
    from modules.storage import EventStore

    def forbidden(*a, **k):
        raise AssertionError("runtime path tried to write an image/video")

    before_root = media_files(ROOT)
    before_tmp = media_files(str(tmp_path))
    monkeypatch.setattr(cv2, "imwrite", forbidden)
    monkeypatch.setattr(cv2, "VideoWriter", forbidden)
    monkeypatch.setattr(Image.Image, "save", forbidden)
    monkeypatch.setattr(np, "save", forbidden)

    db = str(tmp_path / "events.db")
    pipe = MnemoPipeline(mode="rules", store=EventStore(db), use_landmarks=os.path.exists(C.LANDMARK_INDEX))
    events, stats = pipe.run(video, verbose=False)

    assert stats["frames"] == 30                         # 3 s at PROCESS_FPS=10
    assert events == []                                  # no hands → no events
    assert media_files(ROOT) == before_root
    assert media_files(str(tmp_path)) == before_tmp
    assert len(pipe.buffer.frames) == 0                  # RAM buffer released after the run


def test_source_code_has_no_image_writes():
    """Static check of runtime modules (tools/ and detect/ are offline utilities)."""
    runtime = ["modules", "query", "mnemo.py"]
    bad = []
    for item in runtime:
        paths = [os.path.join(ROOT, item)] if item.endswith(".py") else [
            os.path.join(ROOT, item, f) for f in os.listdir(os.path.join(ROOT, item)) if f.endswith(".py")]
        for p in paths:
            src = open(p).read()
            # (landmarks.py's np.savez stores CLIP embeddings of enrollment clips, not images)
            for needle in ("imwrite", "VideoWriter", "Image.save", "np.save("):
                if needle in src:
                    bad.append((p, needle))
    assert bad == [], bad
