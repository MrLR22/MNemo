import cv2
from modules.action_classifier import ActionClassifier
from modules.hand_detector import HandDetector
from modules.motion_detector import MotionDetector
from modules.object_detector import ObjectDetector
from modules.optical_flow import OpticalFlowAnalyzer


class MnemoPipeline:
    def __init__(self):
        self.motion_detector = MotionDetector()
        self.object_detector = ObjectDetector()
        self.hand_detector = HandDetector()
        self.optical_flow_analyzer = OpticalFlowAnalyzer()
        self.action_classifier = ActionClassifier()
        self.prev_frame = None
        print("[MnemoPipeline] Initialized pipeline.")

    def process_frame(self, frame, timestamp):
        """
        Process a single frame through the pipeline.
        Returns a dictionary with detection results.
        """
        results = {"timestamp": timestamp, "motion": False, "objects": [], "hands": [], "actions": []}

        # 1. Check for motion
        has_motion = self.motion_detector.is_active(frame)
        results["motion"] = has_motion

        # 2. If motion is detected, run object detection
        if has_motion:
            detections = self.object_detector.detect(frame)
            hands = self.hand_detector.detect(frame)
            
            # Optical Flow
            if hasattr(self, 'prev_frame') and self.prev_frame is not None:
                for hand in hands:
                    hand_box = hand["box"]
                    flow = self.optical_flow_analyzer.analyze(self.prev_frame, frame, hand_box)
                    hand["flow"] = flow
            
            results["objects"] = detections
            results["hands"] = hands
            results["actions"] = self.action_classifier.classify(frame)

        # Always update previous frame for flow calculation
        self.prev_frame = frame.copy()

        return results
