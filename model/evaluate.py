# model/evaluate.py
# Test-set evaluation of a trained HOI-TCN on feature files, with modality ablation.
#
#   python -m model.evaluate --checkpoint models/hoi_tcn.pt --features data/features/epic data/features/ego4d --split test
#
# --split test uses exactly the held-out test videos recorded in the checkpoint by train.py.
# The decision threshold stored in the checkpoint (chosen on validation) is used, so the
# numbers are honest. Results are also broken down per dataset.

import argparse
import json

import torch

from model.architecture import load_model
from model.data import dataset_of, load_videos
from model.train import evaluate_videos

ABLATIONS = [(), ("object",), ("ego",), ("object", "ego"), ("hands",)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--features", nargs="+", required=True)
    ap.add_argument("--split", choices=["test", "val", "all"], default="test")
    ap.add_argument("--tolerance", type=float, default=1.0, help="seconds")
    ap.add_argument("--json", default=None, help="also write results to this file")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    thr = ckpt.get("threshold", 0.5)
    model = load_model(args.checkpoint)
    vids = load_videos(args.features)
    if args.split != "all":
        names = set(ckpt.get(f"{args.split}_videos", []))
        vids = [v for v in vids if v["name"] in names]
        if not vids:
            raise SystemExit(f"No {args.split} videos from the checkpoint were found in {args.features}")
    else:
        used = set(ckpt.get("val_videos", [])) | set(ckpt.get("train_videos", []))
        overlap = used & {v["name"] for v in vids}
        if overlap:
            print(f"WARNING: {len(overlap)} of these videos were used for training/validation; "
                  "do not report them as test results. Use --split test.")
    print(f"{len(vids)} {args.split} videos, threshold {thr}, tolerance +-{args.tolerance}s\n")

    out = {}
    print(f"{'variant':28s} | {'pickup F1':>9s} | {'putdown F1':>10s} | {'P':>6s} | {'R':>6s} | {'F1':>6s}")
    print("-" * 80)
    for drop in ABLATIONS:
        m = evaluate_videos(model, vids, drop_groups=drop, thresholds=(thr,), tol_s=args.tolerance)[thr]
        name = "full model" if not drop else "without " + "+".join(drop)
        out[name] = m
        o = m["overall"]
        print(f"{name:28s} | {m['pickup']['f1']:9.3f} | {m['putdown']['f1']:10.3f} | "
              f"{o['precision']:6.3f} | {o['recall']:6.3f} | {o['f1']:6.3f}")

    print("\nPer dataset (full model):")
    for ds in sorted({dataset_of(v) for v in vids}):
        dv = [v for v in vids if dataset_of(v) == ds]
        m = evaluate_videos(model, dv, thresholds=(thr,), tol_s=args.tolerance)[thr]
        out[f"dataset:{ds}"] = m
        o = m["overall"]
        print(f"  {ds:12s} {len(dv):4d} videos | P {o['precision']:.3f} R {o['recall']:.3f} F1 {o['f1']:.3f} "
              f"(pickup F1 {m['pickup']['f1']:.3f}, putdown F1 {m['putdown']['f1']:.3f})")
    print("\n(ablation zeroes a modality at test time; for the report, retraining without it is stronger)")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
