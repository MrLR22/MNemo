import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandDetector:
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.IMAGE,
        )

        self.detector = vision.HandLandmarker.create_from_options(options)
        print("[HandDetector] mediapipe Tasks API loaded")

    def detect(self, frame, draw=True, padding=20):
        h, w = frame.shape[:2]

        rgb = frame[:, :, ::-1]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        results = self.detector.detect(mp_image)

        hands = []

        if results.hand_landmarks:
            for hand_landmarks, handedness in zip(
                results.hand_landmarks, results.handedness
            ):
                xs = [lm.x * w for lm in hand_landmarks]
                ys = [lm.y * h for lm in hand_landmarks]

                x1 = max(0, int(min(xs)) - padding)
                y1 = max(0, int(min(ys)) - padding)
                x2 = min(w, int(max(xs)) + padding)
                y2 = min(h, int(max(ys)) + padding)

                wrist = (int(hand_landmarks[0].x * w), int(hand_landmarks[0].y * h))

                index_tip = (int(hand_landmarks[8].x * w), int(hand_landmarks[8].y * h))

                label = handedness[0].category_name
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
