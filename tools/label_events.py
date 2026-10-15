# tools/label_events.py
# Fast keyboard labelling of pick-up / put-down instants in your own chest-camera videos.
# Writes rows to data/own/labels.csv:  video,time_s,object,action,landmark
#
#   python -m tools.label_events data/own/videos/day1_morning.mp4
#
# Keys:
#   1-4      choose object (1 keys, 2 wallet, 3 phone, 4 pill_bottle)
#   p        mark PICK-UP of the chosen object at the current time
#   d        mark PUT-DOWN of the chosen object at the current time
#   u        undo last mark
#   space    pause / play          a / s   back / forward 2 s        , / .   back / forward 1 frame (paused)
#   q        save and quit
# Mark the moment the object leaves the surface (pick-up) or is released (put-down).
# The landmark column is optional: fill it in later for location accuracy (e.g. entryway_table).

import csv
import os
import sys

import cv2

import config as C

KEYS = {ord(str(i + 1)): o for i, o in enumerate(C.TARGET_OBJECTS)}


def main(video: str, labels_csv: str = C.OWN_LABELS_CSV):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        sys.exit(f"Cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    name = os.path.basename(video)
    marks, obj, paused = [], C.TARGET_OBJECTS[0], False
    frame, idx = None, 0

    cv2.namedWindow("label", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("label", 960, 540)
    while True:
        if not paused or frame is None:
            ok, frame = cap.read()
            if not ok:
                paused = True
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n_frames - 1))
                ok, frame = cap.read()
                if not ok:
                    break
            idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        t = idx / fps
        im = frame.copy()
        cv2.putText(im, f"{t:7.2f}s  object: {obj}  marks: {len(marks)}  {'PAUSED' if paused else ''}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        if marks:
            m = marks[-1]
            cv2.putText(im, f"last: {m[1]:.2f}s {m[3]} {m[2]}", (10, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("label", im)
        k = cv2.waitKey(0 if paused else max(1, int(1000 / fps))) & 0xFF
        if k == ord("q"):
            break
        if k in KEYS:
            obj = KEYS[k]
        elif k in (ord("p"), ord("d")):
            marks.append((name, round(t, 2), obj, "pickup" if k == ord("p") else "putdown"))
            print(f"  {t:7.2f}s {marks[-1][3]:7s} {obj}")
        elif k == ord("u") and marks:
            print(f"  undo {marks.pop()}")
        elif k == ord(" "):
            paused = not paused
        elif k in (ord("a"), ord("s"), ord(","), ord(".")):
            step = {ord("a"): -2 * fps, ord("s"): 2 * fps, ord(","): -1, ord("."): 1}[k]
            target = int(min(max(0, idx + step), n_frames - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ok, frame = cap.read()
            idx = target
    cap.release()
    cv2.destroyAllWindows()

    if marks:
        os.makedirs(os.path.dirname(labels_csv), exist_ok=True)
        new = not os.path.exists(labels_csv)
        with open(labels_csv, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["video", "time_s", "object", "action", "landmark"])
            for m in sorted(marks, key=lambda m: m[1]):
                w.writerow([*m, ""])
        print(f"Saved {len(marks)} marks to {labels_csv}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python -m tools.label_events VIDEO   (keys are listed at the top of this file)")
    main(sys.argv[1])
