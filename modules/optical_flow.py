import cv2
import numpy as np

class OpticalFlowAnalyzer:
    def __init__(self, magnitude_threshold: float = 1.5, crop_padding: int = 20):
        self.magnitude_threshold = magnitude_threshold
        self.crop_padding = crop_padding

    def analyze(self, frame1: np.ndarray, frame2: np.ndarray, hand_box: tuple | list = None) -> dict:
        """
        Computes movement direction and magnitude between two frames.
        Args:
            frame1: Previous BGR frame.
            frame2: Current BGR frame.
            hand_box: Optional bounding box [x1, y1, x2, y2].
        """
        h, w = frame1.shape[:2]
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        if hand_box is not None:
            x1, y1, x2, y2 = hand_box
            x1 = max(0, int(x1) - self.crop_padding)
            y1 = max(0, int(y1) - self.crop_padding)
            x2 = min(w, int(x2) + self.crop_padding)
            y2 = min(h, int(y2) + self.crop_padding)
            if x2 > x1 and y2 > y1:
                roi1 = gray1[y1:y2, x1:x2]
                roi2 = gray2[y1:y2, x1:x2]
            else:
                roi1, roi2 = gray1, gray2
        else:
            roi1, roi2 = gray1, gray2
        flow = cv2.calcOpticalFlowFarneback(
            prev=roi1,
            next=roi2,
            flow=None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        u = flow[..., 0]
        v = flow[..., 1]
        pixel_magnitudes = np.sqrt(u**2 + v**2)
        motion_mask = pixel_magnitudes > (self.magnitude_threshold * 0.5)

        if np.any(motion_mask):
            med_x = float(np.median(u[motion_mask]))
            med_y = float(np.median(v[motion_mask]))
            magnitude = float(np.sqrt(med_x**2 + med_y**2))
        else:
            med_x, med_y, magnitude = 0.0, 0.0, 0.0

        direction = self._classify_direction(med_x, med_y, magnitude)

        return {
            "direction": direction,
            "magnitude": round(magnitude, 3),
            "vector_x": round(med_x, 3),
            "vector_y": round(med_y, 3)
        }

    def _classify_direction(self, vx: float, vy: float, magnitude: float) -> str:
        if magnitude < self.magnitude_threshold:
            return "stationary"

        if abs(vy) > abs(vx):
            return "upward" if vy < 0 else "downward"
        else:
            return "right" if vx > 0 else "left"
