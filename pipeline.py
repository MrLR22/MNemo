import cv2
from modules.action_classifier import ActionClassifier
from modules.event_binder import EventBinder
from modules.hand_detector import HandDetector
from modules.location_detector import LocationDetector
from modules.motion_detector import MotionDetector
from modules.object_detector import ObjectDetector
from modules.optical_flow import OpticalFlowAnalyzer
from config import VIDEO_SOURCE


class MnemoPipeline:
    def __init__(self):
        self.motion_detector = MotionDetector()
        self.object_detector = ObjectDetector()
        self.hand_detector = HandDetector()
        self.optical_flow_analyzer = OpticalFlowAnalyzer()
        self.action_classifier = ActionClassifier()
        self.location_detector = LocationDetector(
            model=self.action_classifier.model,
            preprocess=self.action_classifier.preprocess,
        )
        self.event_binder = EventBinder()
        self.prev_frame = None
        self._prev_timestamp = None
        print("[MnemoPipeline] Initialized pipeline.")

    def process_frame(self, frame, timestamp):
        """
        Process a single frame through the pipeline.
        Returns a dictionary with detection results.
        """
        results = {
            "timestamp": timestamp,
            "motion": False,
            "objects": [],
            "hands": [],
            "actions": [],
            "room": {},
            "surface": {},
            "event": None,
        }

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
            results["room"] = self.location_detector.detect_room(frame)

            # Surface detection — use the first detected object's box if available
            obj_box = detections[0]["box"] if detections else None
            results["surface"] = self.location_detector.detect_surface(frame, obj_box)

            # Aggregate flow from all hands — pick the one with highest magnitude
            flow_result = max(
                (h["flow"] for h in hands if "flow" in h),
                key=lambda f: f["magnitude"],
                default={"direction": "stationary", "magnitude": 0.0, "vector_x": 0.0, "vector_y": 0.0},
            )

            # Bind all signals into one EventRecord (or None)
            results["event"] = self.event_binder.bind(
                yolo_detections=detections,
                hand_detections=hands,
                flow_result=flow_result,
                action_scores=results["actions"],
                location_result={"room": results["room"], "surface": results["surface"]},
                timestamp_start=self._prev_timestamp or timestamp,
                timestamp_end=timestamp,
                video_source=VIDEO_SOURCE,
            )

        # Always update previous frame + timestamp for flow calculation
        self.prev_frame = frame.copy()
        self._prev_timestamp = timestamp

        return results
