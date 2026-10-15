# modules/perception.py
# M2: one call per frame → FrameObs (hands, tracked objects, camera motion). No pixels leave here.

from modules.ego_motion import EgoMotionEstimator
from modules.hand_detector import HandDetector
from modules.object_detector import ObjectDetector
from modules.types import Frame, FrameObs


class Perception:
    def __init__(self, any_object: bool = False, own_hands_only: bool = True, yolo_weights=None):
        self.hands = HandDetector(own_hands_only=own_hands_only)
        self.objects = ObjectDetector(weights=yolo_weights, any_object=any_object)
        self.ego = EgoMotionEstimator()

    def reset(self):
        self.ego.reset()

    def __call__(self, frame: Frame) -> FrameObs:
        hands = self.hands.detect(frame.image)
        dets = self.objects.detect(frame.image)
        ego = self.ego.update(frame.image, [h.box for h in hands])
        return FrameObs(t=frame.t, hands=hands, detections=dets, ego=ego)
