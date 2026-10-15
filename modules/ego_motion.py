# modules/ego_motion.py
# Camera (body) motion estimate. On a chest camera everything moves when the wearer
# walks or turns, so object/hand speeds must be measured relative to the background.
# Sparse Lucas-Kanade on a downscaled frame, hands masked out, median displacement.

import cv2
import numpy as np

_WIDTH = 320


class EgoMotionEstimator:
    def __init__(self):
        self.prev_gray = None

    def reset(self):
        self.prev_gray = None

    def update(self, image_bgr: np.ndarray, hand_boxes=()) -> np.ndarray:
        """Return camera motion (dx, dy) in normalised frame units since the previous call."""
        h, w = image_bgr.shape[:2]
        scale = _WIDTH / w
        gray = cv2.cvtColor(cv2.resize(image_bgr, (_WIDTH, int(h * scale))), cv2.COLOR_BGR2GRAY)
        prev, self.prev_gray = self.prev_gray, gray
        if prev is None:
            return np.zeros(2)

        gh, gw = gray.shape
        mask = np.full((gh, gw), 255, np.uint8)
        for x1, y1, x2, y2 in hand_boxes:      # hands move on their own; exclude them
            mask[int(y1 * gh):int(y2 * gh) + 1, int(x1 * gw):int(x2 * gw) + 1] = 0

        pts = cv2.goodFeaturesToTrack(prev, maxCorners=150, qualityLevel=0.01, minDistance=8, mask=mask)
        if pts is None or len(pts) < 8:
            return np.zeros(2)
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev, gray, pts, None)
        good = status.reshape(-1) == 1
        if good.sum() < 8:
            return np.zeros(2)
        d = (nxt[good] - pts[good]).reshape(-1, 2)
        dx, dy = np.median(d, axis=0)
        return np.array([dx / gw, dy / gh])
