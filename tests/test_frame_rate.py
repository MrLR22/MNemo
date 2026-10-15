# Different datasets have different frame rates (Charades ~24-30, EPIC 50-60, phone 30).
# Sampling and motion features must not depend on the source frame rate.

import cv2
import numpy as np
import pytest

from modules.features import FeatureExtractor, GROUPS, HAND_DIM
from modules.video_source import get_frames
from tests.synth import hand_at, obs
from modules.types import FrameObs


@pytest.mark.parametrize("src_fps", [24, 25, 29.97, 30, 50, 60])
def test_sampling_is_10_fps_for_any_source_rate(tmp_path, src_fps):
    path = str(tmp_path / f"v{src_fps}.avi")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), src_fps, (64, 48))
    for _ in range(int(round(src_fps * 5))):           # 5 seconds
        w.write(np.zeros((48, 64, 3), np.uint8))
    w.release()
    times = [f.t for f in get_frames(path)]
    assert len(times) == 50, (src_fps, len(times))     # 10 fps x 5 s
    gaps = np.diff(times)
    assert abs(gaps.mean() - 0.1) < 0.005
    assert gaps.max() - gaps.min() < 1.0 / src_fps + 1e-6   # at most one source frame of jitter


def test_hand_velocity_feature_independent_of_frame_gap():
    """A hand moving at 0.2 frame-widths per second must give the same velocity feature whether
    frames are 0.1 s apart or 0.0833 s apart (24 fps source)."""
    vel_idx = 43                                          # slot 0 velocity x (Left hand)
    out = []
    for gap in (0.1, 1 / 12):
        fx = FeatureExtractor()
        f = None
        for k in range(5):
            t = k * gap
            o = FrameObs(t=t, hands=[hand_at(0.3 + 0.2 * t, 0.8, label="Left")], detections=[],
                         ego=np.zeros(2))
            f = fx(o)
        out.append(f[vel_idx])
    assert out[0] == pytest.approx(out[1], rel=1e-3)
    assert out[0] == pytest.approx(0.2 * 0.1 * 10.0, rel=1e-3)   # 0.02 per 0.1 s x VEL_SCALE
