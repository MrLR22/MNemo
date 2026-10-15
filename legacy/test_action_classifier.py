"""Quick accuracy check with expanded prompts."""

import cv2
from modules.action_classifier import ActionClassifier
from config import VIDEO_SOURCE

cap = cv2.VideoCapture(VIDEO_SOURCE)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

classifier = ActionClassifier()

positions = [int(total * f) for f in [0.1, 0.25, 0.4, 0.6, 0.8]]

for pos in positions:
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    ret, frame = cap.read()
    if not ret:
        continue
    ts = pos / fps
    results = classifier.classify(frame)
    print(f"\n{'='*60}")
    print(f"Frame {pos}  ({ts:.1f}s)")
    for i, r in enumerate(results[:5], 1):
        print(f"  {i}. [{r['score']:.4f}]  {r['action']}")

cap.release()
