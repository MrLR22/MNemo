# evaluation/evaluate_pipeline.py
# M10: end-to-end evaluation on labelled test videos (your own footage).
#
#   python -m evaluation.evaluate_pipeline --videos data/own/test_videos --labels data/own/test_labels.csv \
#       --mode rules          (or --mode model)
#
# Reports (brief section 10):
#   - pick-up / put-down event precision, recall, F1 (object must match, +-tolerance seconds)
#   - landmark top-1 accuracy and "unknown" rate on correctly detected put-downs (if the
#     labels CSV has a landmark column filled in)
#   - "where is X?" accuracy: after each labelled put-down, does the store's latest put-down
#     for that object point to the right landmark?
#   - CPU FPS and peak RAM

import argparse
import csv
import os
from collections import defaultdict

from model.metrics import match_events
from modules.pipeline import MnemoPipeline
from modules.storage import EventStore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--mode", choices=["rules", "model"], default="rules")
    ap.add_argument("--tolerance", type=float, default=1.5)
    args = ap.parse_args()

    gt = defaultdict(list)
    with open(args.labels) as f:
        for r in csv.DictReader(f):
            gt[r["video"]].append((float(r["time_s"]), r["action"], r["object"], (r.get("landmark") or "").strip()))

    pipe = MnemoPipeline(mode=args.mode, store=EventStore(":memory:"))
    all_pred, all_gt, offset = [], [], 0.0
    lm_total = lm_correct = lm_unknown = 0
    where_total = where_correct = 0
    fps, ram = [], []
    for video, labels in sorted(gt.items()):
        path = os.path.join(args.videos, video)
        events, stats = pipe.run(path, verbose=False)
        fps.append(stats["fps"])
        ram.append(stats["peak_ram_mb"])
        pred = [(e.t + offset, e.action, e.object) for e in events]
        g = [(t + offset, t + offset, a, o) for t, a, o, _ in labels]
        all_pred += pred
        all_gt += g
        print(f"{video}: {len(events)} predicted, {len(labels)} labelled, {stats['fps']} fps")

        # landmark accuracy on put-downs that were detected
        for t, a, o, lm in labels:
            if a != "putdown" or not lm:
                continue
            near = [e for e in events if e.action == "putdown" and e.object == o
                    and abs(e.t - t) <= args.tolerance]
            if near:
                lm_total += 1
                if near[0].landmark is None:
                    lm_unknown += 1
                elif near[0].landmark == lm:
                    lm_correct += 1
            # "where is X?" right after this put-down: latest put-down event before t+tol
            before = [e for e in events if e.object == o and e.t <= t + args.tolerance]
            where_total += 1
            if before and before[-1].action == "putdown" and before[-1].landmark == lm:
                where_correct += 1
        offset += 1e5

    m = match_events(all_pred, all_gt, tol_s=args.tolerance, use_object=True)
    print(f"\n=== {args.mode} mode, tolerance +-{args.tolerance}s, object must match ===")
    for k in ("pickup", "putdown", "overall"):
        r = m[k]
        print(f"{k:8s} P {r['precision']:.3f}  R {r['recall']:.3f}  F1 {r['f1']:.3f}  "
              f"(tp {r['tp']}, fp {r['fp']}, fn {r['fn']})")
    if lm_total:
        print(f"landmark top-1 on detected put-downs: {lm_correct / lm_total:.3f} "
              f"(unknown rate {lm_unknown / lm_total:.3f}, n={lm_total})")
    if where_total:
        print(f"'where is X?' accuracy after each labelled put-down: {where_correct / where_total:.3f} "
              f"(n={where_total})")
    print(f"CPU throughput: {min(fps)}-{max(fps)} processed fps, peak RAM {max(ram)} MB")


if __name__ == "__main__":
    main()
