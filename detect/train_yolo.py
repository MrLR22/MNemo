# detect/train_yolo.py
# Fine-tune YOLOv8n on your tracked objects (keys, wallet, phone, pill_bottle).
#
# 1. Take ~150 varied photos per object FROM THE CHEST CAMERA (on tables, in hand, partly
#    covered by fingers, different rooms and lighting). Also photos with NO objects.
# 2. Label boxes in Roboflow or CVAT. Class names must be exactly: keys, wallet, phone, pill_bottle
# 3. Export in "YOLOv8" format → you get a folder with data.yaml, train/, valid/ (and test/).
# 4. python -m detect.train_yolo --data path/to/data.yaml
#    (CPU works but is slow; on Colab use --device 0)
# The best weights are copied to models/mnemo_objects.pt, which the pipeline picks up automatically.

import argparse
import os
import shutil

import yaml
from ultralytics import YOLO

import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="data.yaml from the Roboflow/CVAT YOLOv8 export")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cpu", help="'cpu' or GPU index like 0")
    args = ap.parse_args()

    with open(args.data) as f:
        names = yaml.safe_load(f)["names"]
    names = list(names.values()) if isinstance(names, dict) else list(names)
    unknown = [n for n in names if n not in C.TARGET_OBJECTS]
    if unknown:
        raise SystemExit(f"Class names {unknown} are not in TARGET_OBJECTS {C.TARGET_OBJECTS}. "
                         "Rename them in the labelling tool before exporting.")

    model = YOLO(C.COCO_YOLO)       # start from COCO-pretrained yolov8n
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device=args.device, project="runs/mnemo_yolo", name="train", exist_ok=True,
                patience=20)
    metrics = model.val()
    print(f"mAP50 = {metrics.box.map50:.3f}, mAP50-95 = {metrics.box.map:.3f}")

    best = os.path.join("runs", "mnemo_yolo", "train", "weights", "best.pt")
    os.makedirs(os.path.dirname(C.FINETUNED_YOLO), exist_ok=True)
    shutil.copy(best, C.FINETUNED_YOLO)
    print(f"Copied {best} → {C.FINETUNED_YOLO}")


if __name__ == "__main__":
    main()
