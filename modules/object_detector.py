from ultralytics import YOLO
from config import YOLO_MODEL


class ObjectDetector:
    def __init__(self):
        self.model = YOLO(YOLO_MODEL)
        self.conf_threshold = 0.5
        self.names = self.model.names
        print(f"[ObjectDetector] loaded model : {self.model}")

    def detect(self, frame):
        results = self.model(frame, verbose=False, conf=self.conf_threshold)
        detections = []

        # Iterate over the results generator to satisfy Pyright
        for result in results:
            for box in result.boxes:
                label = self.names[int(box.cls.item())]
                confidence = float(box.conf.item())

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                detections.append(
                    {
                        "label": label,
                        "confidence": confidence,
                        "box": (x1, y1, x2, y2),
                        "center": ((x1 + x2) // 2, (y1 + y2) // 2),
                    }
                )

        return detections

