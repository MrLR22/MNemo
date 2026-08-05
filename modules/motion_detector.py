import cv2


class MotionDetector:
    def __init__(self, sensitivity=5000, threshold_limit=25):
        self.sensitivity = sensitivity
        self.threshold_limit = threshold_limit
        self.prev_gray = None

    def is_active(self, frame):
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.GaussianBlur(current_gray, (21, 21), 0)

        if self.prev_gray is None:
            self.prev_gray = current_gray
            return False

        frame_delta = cv2.absdiff(self.prev_gray, current_gray)

        thresh = cv2.threshold(
            frame_delta, self.threshold_limit, 255, cv2.THRESH_BINARY
        )[1]

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.dilate(thresh, kernel, iterations=2)

        non_zero_count = cv2.countNonZero(thresh)
        motion_detected = non_zero_count > self.sensitivity

        self.prev_gray = current_gray
        return motion_detected
