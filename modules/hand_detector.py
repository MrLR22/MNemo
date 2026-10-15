# modules/hand_detector.py
# M2: MediaPipe Hands (Tasks API) → list[Hand] with normalised landmarks.
# Only the wearer's own hands are kept (see OWN_HAND_MIN_WRIST_Y in config).

import sys
import types

# MediaPipe imports `sounddevice` for its audio tasks. On this machine PortAudio's JACK/PipeWire
# start-up gets the whole process SIGKILLed (verified with gdb: kill comes from outside, 148 MB RSS).
# Mnemo uses no audio, so register an empty stub before mediapipe is imported.
if "sounddevice" not in sys.modules:
    sys.modules["sounddevice"] = types.ModuleType("sounddevice")

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from config import HAND_MODEL, HAND_MIN_CONFIDENCE, FLIP_HANDEDNESS, OWN_HAND_MIN_WRIST_Y
from modules.types import Hand

_FLIP = {"Left": "Right", "Right": "Left"}


class HandDetector:
    def __init__(self, own_hands_only: bool = True):
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL),
            num_hands=2,
            min_hand_detection_confidence=HAND_MIN_CONFIDENCE,
            min_hand_presence_confidence=HAND_MIN_CONFIDENCE,
            min_tracking_confidence=HAND_MIN_CONFIDENCE,
            running_mode=vision.RunningMode.IMAGE,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.own_hands_only = own_hands_only

    def detect(self, image_bgr: np.ndarray) -> list[Hand]:
        rgb = np.ascontiguousarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        result = self.detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

        hands = []
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            pts = np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float32).clip(0.0, 1.0)
            label = handedness[0].category_name
            if FLIP_HANDEDNESS:
                label = _FLIP.get(label, label)
            hand = Hand(label=label, landmarks=pts, score=float(handedness[0].score))
            if self.own_hands_only and hand.wrist[1] < OWN_HAND_MIN_WRIST_Y:
                continue        # wrist too high in frame: probably someone else's hand
            hands.append(hand)
        return hands

    def close(self):
        self.detector.close()
