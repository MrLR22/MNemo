# evaluation/tune_gate.py
# Choose the trigger settings (HOI threshold, hand-motion threshold, keep-going windows) on the
# VALIDATION people of HOI-TCN (P04 P21 P22 P27 P31), never on the test people, so the benchmark on
# the test people stays fair.
#
# No VLM needed: for every setting we only count
#   calls      fraction of 4-s windows sent to the VLM
#   coverage   fraction of ALL annotated EPIC actions (any verb) that overlap a called window
# and pick the setting with the fewest calls that reaches --target coverage. A fixed-rate baseline
# (every n-th window) at the same number of calls is printed next to it.
#
#   python -m evaluation.tune_gate --clips /kaggle/working/val_clips --annotations EPIC_100_train.csv \
#       --checkpoint models/hoi_tcn.pt --out results/gate_config.json

import argparse
import glob
import itertools
import json
import os

import numpy as np

import config as C
from evaluation.gating_experiment import coverage, select
from evaluation.log_accuracy import clip_name_parts, load_truth
from modules.hoi_gate import HOIGate, gate_decisions

THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
HANGOVERS = (0, 1, 2)
MOTION_PCTS = (None, 30, 50, 70, 85)            # motion threshold = this percentile of window speeds


def evaluate(signals: list[dict], actions: list[list[tuple]], threshold, motion_thr, hangover, window_s):
    sel = [gate_decisions(s["hoi"], threshold, motion=s["motion"], motion_threshold=motion_thr,
                          hangover=hangover) for s in signals]
    calls = sum(x.sum() for x in sel) / max(1, sum(len(x) for x in sel))
    return float(calls), coverage(sel, actions, window_s)


def tune(signals: list[dict], actions: list[list[tuple]], target: float = 0.9,
         window_s: float = C.VLM_WINDOW_S) -> tuple[dict, list[dict]]:
    allm = np.concatenate([s["motion"] for s in signals]) if signals else np.zeros(1)
    rows = []
    for th, hang, pct in itertools.product(THRESHOLDS, HANGOVERS, MOTION_PCTS):
        mthr = None if pct is None else float(np.percentile(allm, pct))
        calls, cov = evaluate(signals, actions, th, mthr, hang, window_s)
        rows.append({"threshold": th, "hangover": hang, "motion_pct": pct, "motion_threshold": mthr,
                     "calls": calls, "coverage": cov})
    ok = [r for r in rows if r["coverage"] >= target]
    best = min(ok, key=lambda r: (r["calls"], -r["coverage"])) if ok else max(rows, key=lambda r: (r["coverage"], -r["calls"]))
    best = dict(best)
    uni = select([s["hoi"] for s in signals], "uniform", best["calls"])
    best["uniform_coverage_same_calls"] = coverage(uni, actions, window_s)
    return best, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True, help="folder with <video>_w<start>.mp4 clips of validation people")
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--checkpoint", default=C.MODEL_CHECKPOINT)
    ap.add_argument("--target", type=float, default=0.9)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gate = HOIGate(args.checkpoint)
    signals, actions = [], []
    for k, v in enumerate(sorted(glob.glob(os.path.join(args.clips, "*_w*.mp4"))), 1):
        sig, st = gate.signals(v)
        vid, t0 = clip_name_parts(v)
        dur = len(sig["hoi"]) * C.VLM_WINDOW_S
        truth = load_truth(args.annotations, vid, t0, dur)
        signals.append(sig)
        actions.append([(g["start"], g["stop"], g["narration"]) for g in truth])
        print(f"  [{k}] {os.path.basename(v)}: {len(truth)} actions, {st['gate_seconds']}s")
    best, rows = tune(signals, actions, args.target)
    print(f"\n{sum(len(a) for a in actions)} actions in {len(signals)} validation clips\n")
    print(f"{'HOI thr':>7s} {'keep-going':>10s} {'motion':>8s} | {'calls':>6s} {'coverage':>8s}")
    show = [r for r in sorted(rows, key=lambda r: (r["calls"], -r["coverage"])) if r["coverage"] >= args.target - 0.1][:12]
    for r in show:                                     # the 12 cheapest settings near the target
        m = "off" if r["motion_pct"] is None else f"p{r['motion_pct']}"
        print(f"{r['threshold']:7.1f} {r['hangover']:10d} {m:>8s} | {r['calls']:6.0%} {r['coverage']:8.0%}")
    print(f"\nCHOSEN: HOI threshold {best['threshold']}, keep-going {best['hangover']}, motion "
          f"{'off' if best['motion_pct'] is None else 'p' + str(best['motion_pct'])} → VLM on {best['calls']:.0%} "
          f"of windows, {best['coverage']:.0%} of actions covered "
          f"(every-n-th window with the same calls: {best['uniform_coverage_same_calls']:.0%})")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"chosen": best, "all": rows, "target": args.target}, f, indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
