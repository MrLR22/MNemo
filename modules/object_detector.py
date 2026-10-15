# modules/object_detector.py
# M2: object detection → list[Detection] for the tracked personal objects only.
#
# Backends (config.OBJECT_BACKEND):
#   "finetuned"  YOLOv8n fine-tuned on your photos (models/mnemo_objects.pt). Fastest, most accurate.
#   "yoloworld"  YOLO-World open-vocabulary detector: finds objects from their NAMES, no training data.
#                ~216 ms/frame on the laptop CPU (vs ~53 ms for yolov8n), so slower than real time.
#   "coco"       stock COCO yolov8n; only phone (and bottle as a rough pill-bottle stand-in).
#   "auto"       finetuned if the file exists, else yoloworld.
# any_object=True (public-dataset feature extraction) always uses fast COCO yolov8n and reports any
# portable object, because HOI-TCN's features do not depend on the object's identity.

import os

import numpy as np

import config as C
from modules.types import Detection

# Classes that are never a hand-held "object"
_NOT_PORTABLE = {"person", "chair", "couch", "bed", "dining table", "toilet", "tv", "refrigerator",
                 "oven", "sink", "bench", "car", "bus", "truck", "train", "airplane", "boat"}


def resolve_backend(backend: str | None = None) -> str:
    backend = backend or C.OBJECT_BACKEND
    if backend == "auto":
        return "finetuned" if os.path.exists(C.FINETUNED_YOLO) else "yoloworld"
    return backend


class ObjectDetector:
    def __init__(self, weights: str | None = None, any_object: bool = False, backend: str | None = None):
        self.any_object = any_object
        if any_object:
            self.backend = "coco"
        elif weights is not None:
            self.backend = "finetuned"
        else:
            self.backend = resolve_backend(backend)

        if self.backend == "yoloworld":
            from ultralytics import YOLOWorld
            self.model = YOLOWorld(C.YOLOWORLD_MODEL)
            prompts = [C.YOLOWORLD_PROMPTS[o] for o in C.TARGET_OBJECTS]
            self.model.set_classes(prompts)
            self.prompt_to_obj = {C.YOLOWORLD_PROMPTS[o]: o for o in C.TARGET_OBJECTS}
            self.conf = C.YOLOWORLD_CONFIDENCE
        else:
            from ultralytics import YOLO
            if self.backend == "finetuned":
                path = weights or C.FINETUNED_YOLO
                if not os.path.exists(path):
                    raise FileNotFoundError(f"No fine-tuned detector at {path}")
            else:
                path = C.COCO_YOLO
            self.model = YOLO(path)
            self.conf = C.YOLO_CONFIDENCE
            if self.backend == "coco" and not any_object:
                print("[ObjectDetector] WARNING: COCO weights can only detect "
                      f"{sorted(set(C.COCO_CLASS_MAP.values()))}.")
        self.names = self.model.names

    def _canonical(self, name: str) -> str | None:
        if self.any_object:
            return None if name in _NOT_PORTABLE else name
        if self.backend == "yoloworld":
            return self.prompt_to_obj.get(name)
        if self.backend == "finetuned":
            return name if name in C.TARGET_OBJECTS else None
        return C.COCO_CLASS_MAP.get(name)

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        h, w = image_bgr.shape[:2]
        # agnostic NMS: one label per box (YOLO-World otherwise labels one lid as both 'lid' and 'bowl')
        result = self.model(image_bgr, conf=self.conf, imgsz=C.YOLO_IMGSZ, agnostic_nms=True, verbose=False)[0]
        detections = []
        for box in result.boxes:
            obj = self._canonical(self.names[int(box.cls.item())])
            if obj is None:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(Detection(obj=obj, conf=float(box.conf.item()),
                                        box=(x1 / w, y1 / h, x2 / w, y2 / h)))
        # Keep the most confident detection per object (one wallet, one phone, ...)
        best: dict[str, Detection] = {}
        for d in detections:
            if d.obj not in best or d.conf > best[d.obj].conf:
                best[d.obj] = d
        return list(best.values())
