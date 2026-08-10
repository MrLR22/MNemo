import cv2
from modules.motion_detector import MotionDetector
from modules.object_detector import ObjectDetector

class MnemoPipeline:
    def __init__(self):
        self.motion_detector = MotionDetector()
        self.object_detector = ObjectDetector()
        print("[MnemoPipeline] Initialized pipeline.")

    def process_frame(self, frame, timestamp):
        """
        Process a single frame through the pipeline.
        Returns a dictionary with detection results.
        """
        results = {
            "timestamp": timestamp,
            "motion": False,
            "objects": []
        }

        # 1. Check for motion
        has_motion = self.motion_detector.is_active(frame)
        results["motion"] = has_motion

        # 2. If motion is detected, run object detection
        if has_motion:
            detections = self.object_detector.detect(frame)
            results["objects"] = detections

        return results
