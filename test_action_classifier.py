"""Quick diagnostic: sample 5 frames, print ALL scores to see the distribution."""

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
    print(f"\n{'='*70}")
    print(f"Frame {pos}  ({ts:.1f}s)")
    print(f"{'='*70}")
    for i, r in enumerate(results, 1):
        print(f"  {i:2}. [{r['score']:.4f}]  {r['action']}")

cap.release()
