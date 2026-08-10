import mediapipe as mp
import cv2


class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.mp_draw = mp.solutions.drawing_utils
        print(f"[HandDetector] mediapipe hands loaded")

    def detect(self, frame, draw=True, padding=20):
        h, w = frame.shape[:2]

        # Fast BGR to RGB conversion using NumPy slicing
        rgb = frame[:, :, ::-1]
        results = self.hands.process(rgb)

        hands = []

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                xs = [lm.x * w for lm in hand_landmarks.landmark]
                ys = [lm.y * h for lm in hand_landmarks.landmark]

                x1 = max(0, int(min(xs)) - padding)
                y1 = max(0, int(min(ys)) - padding)
                x2 = min(w, int(max(xs)) + padding)
                y2 = min(h, int(max(ys)) + padding)
                wrist = (
                    int(hand_landmarks.landmark[0].x * w),
                    int(hand_landmarks.landmark[0].y * h),
                )
                index_tip = (
                    int(hand_landmarks.landmark[8].x * w),
                    int(hand_landmarks.landmark[8].y * h),
                )

                # Extract Left/Right label
                label = handedness.classification[0].label
                hands.append(
                    {
                        "label": label,
                        "box": (x1, y1, x2, y2),
                        "wrist": wrist,
                        "index_tip": index_tip,
                        "center": ((x1 + x2) // 2, (y1 + y2) // 2),
                    }
                )
        return hands
