# evaluation/gating_experiment.py
# EXPERIMENT A — does an interaction trigger let the expensive VLM run on fewer windows without
# missing what the user did?
#
# The video is cut into VLM windows (4 s, as in stage 1). Each "gate" gives every window a score;
# at a given call rate r the top-r fraction of windows is sent to the VLM. An action counts as
# COVERED if at least one selected window overlaps it. Ground truth = EPIC-KITCHENS annotations of
# ALL actions (open, close, wash, cut, ... not only pick/put), so the trigger is tested on more than
# it was trained on. Same idea of "action coverage vs. VLM call rate" as Jalayer et al. 2026
# (acoustic triage, arXiv 2608.22359), so results are comparable in spirit.
#
# Gates compared:
#   hoi      our HOI-TCN: max over the window of P(pick-up or put-down)
#   hands    fraction of frames with a hand visible (MediaPipe only, no learning)
#   motion   mean hand speed relative to the background
#   uniform  every 1/r-th window (what a fixed-rate captioner does)
#   random   random windows (sanity baseline)
#
#   python -m evaluation.gating_experiment --features data/features/epic_cont \
#       --checkpoint models/hoi_tcn.pt --annotations data/demo/EPIC_100_train.csv --out results/gating

import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np

from modules.features import HAND_DIM

_trapz = getattr(np, "trapezoid", None) or np.trapz      # numpy 2 / numpy 1
GATES = ["hoi", "hands", "motion", "uniform", "random"]
RATES = [round(0.05 * i, 2) for i in range(1, 21)]


def secs(ts: str) -> float:
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def load_all_actions(path: str) -> dict[str, list[tuple]]:
    """EPIC annotations, every verb: {video_id: [(start, stop, narration)]}."""
    out = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            out[r["video_id"]].append((secs(r["start_timestamp"]), secs(r["stop_timestamp"]), r["narration"]))
    return out


def clip_actions(actions: list[tuple], t0: float, duration: float) -> list[tuple]:
    """Actions overlapping [t0, t0+duration], re-timed and clipped to the clip."""
    return [(max(0.0, s - t0), min(duration, e - t0), n) for s, e, n in actions if e > t0 and s < t0 + duration]


def window_scores(times: np.ndarray, feats: np.ndarray, probs: np.ndarray | None, window_s: float,
                  rng: np.random.Generator) -> dict[str, np.ndarray]:
    duration = float(times[-1]) + (times[1] - times[0] if len(times) > 1 else 0.1)
    n_win = max(1, math.ceil(duration / window_s - 1e-9))
    idx = np.minimum((times / window_s).astype(int), n_win - 1)
    hand_present = feats[:, [0, HAND_DIM]].max(1)
    v1 = np.linalg.norm(feats[:, 43:45], axis=1)
    v2 = np.linalg.norm(feats[:, HAND_DIM + 43:HAND_DIM + 45], axis=1)
    speed = np.maximum(v1, v2)
    act = (1.0 - probs[:, 0]) if probs is not None else np.zeros(len(times))
    s = {g: np.zeros(n_win) for g in ("hoi", "hands", "motion")}
    for w in range(n_win):
        m = idx == w
        if m.any():
            s["hoi"][w] = act[m].max()
            s["hands"][w] = hand_present[m].mean()
            s["motion"][w] = speed[m].mean()
    s["random"] = rng.random(n_win)
    return s


def select(scores: list[np.ndarray], gate: str, rate: float) -> list[np.ndarray]:
    """Boolean selection per video at call rate `rate` (fraction of all windows)."""
    if gate == "uniform":
        out = []
        for sc in scores:
            k = np.arange(len(sc))
            out.append(np.floor((k + 1) * rate) > np.floor(k * rate))       # evenly spread, exact rate
        return out
    allv = np.concatenate(scores)
    n_sel = max(1, int(round(rate * len(allv))))
    order = np.argsort(-allv, kind="stable")
    chosen = np.zeros(len(allv), bool)
    chosen[order[:n_sel]] = True
    out, i = [], 0
    for sc in scores:
        out.append(chosen[i:i + len(sc)])
        i += len(sc)
    return out


def coverage(selected: list[np.ndarray], actions: list[list[tuple]], window_s: float) -> float:
    hit = total = 0
    for sel, acts in zip(selected, actions):
        for s, e, _ in acts:
            total += 1
            w0, w1 = int(s // window_s), int(max(s, e - 1e-6) // window_s)
            hit += bool(sel[w0:min(w1, len(sel) - 1) + 1].any())
    return hit / total if total else 0.0


def curves(scores_by_gate: dict, actions: list, window_s: float) -> dict:
    out = {}
    for g, sc in scores_by_gate.items():
        cov = [coverage(select(sc, g, r), actions, window_s) for r in RATES]
        need = {}
        for target in (0.8, 0.9, 0.95):
            ok = [r for r, c in zip(RATES, cov) if c >= target]
            need[f"rate_for_{int(target * 100)}pct"] = ok[0] if ok else None
        out[g] = {"coverage": cov, "auc": float(_trapz(cov, RATES) / (RATES[-1] - RATES[0])), **need}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", required=True, help="feature dirs of CONTINUOUS clips")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--annotations", required=True, help="EPIC_100_train.csv (all verbs)")
    ap.add_argument("--window-s", type=float, default=4.0)
    ap.add_argument("--vlm-seconds", type=float, default=None,
                    help="measured VLM time per window (from colab/mnemo_vlm.ipynb) → real-time estimate")
    ap.add_argument("--out", default="results/gating")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from model.architecture import load_model
    from model.train import predict_video
    model = load_model(args.checkpoint)
    ann = load_all_actions(args.annotations)
    rng = np.random.default_rng(args.seed)

    files = sorted(f for d in args.features for f in glob.glob(os.path.join(d, "*.npz")))
    scores = {g: [] for g in GATES}
    actions, names, total_s = [], [], 0.0
    for f in files:
        z = np.load(f)
        clip = os.path.basename(f)[:-4].split("__", 1)[-1]             # <group>__<video>_w<start>
        video, _, t0 = clip.rpartition("_w")
        times, feats = z["times"], z["feats"]
        if len(times) < 2:
            continue
        probs = predict_video(model, feats)
        s = window_scores(times, feats, probs, args.window_s, rng)
        duration = float(times[-1]) + float(times[1] - times[0])
        acts = clip_actions(ann.get(video, []), float(t0), duration)
        for g in ("hoi", "hands", "motion", "random"):
            scores[g].append(s[g])
        scores["uniform"].append(s["random"])                           # only its length is used
        actions.append(acts)
        names.append(clip)
        total_s += duration
    if not names:
        raise SystemExit(f"No feature files in {args.features}")
    n_win = sum(len(x) for x in scores["hoi"])
    n_act = sum(len(a) for a in actions)
    print(f"{len(names)} continuous clips, {total_s / 60:.1f} min, {n_win} windows of {args.window_s:.0f}s, "
          f"{n_act} annotated actions (all verbs)\n")

    res = curves(scores, actions, args.window_s)
    print(f"{'gate':8s} | {'cov@10%':>7s} {'cov@25%':>7s} {'cov@50%':>7s} | {'AUC':>5s} | calls for 80% / 90% / 95% coverage")
    print("-" * 86)
    for g in GATES:
        r = res[g]
        c = dict(zip(RATES, r["coverage"]))
        fmt = lambda x: f"{x:.0%}" if x is not None else "never"
        print(f"{g:8s} | {c[0.1]:7.1%} {c[0.25]:7.1%} {c[0.5]:7.1%} | {r['auc']:.3f} | "
              f"{fmt(r['rate_for_80pct'])} / {fmt(r['rate_for_90pct'])} / {fmt(r['rate_for_95pct'])}")

    if args.vlm_seconds:
        print(f"\nReal-time estimate with {args.vlm_seconds:.1f}s of VLM time per {args.window_s:.0f}s window:")
        for g in ("hoi", "uniform"):
            r = res[g]["rate_for_90pct"]
            if r:
                print(f"  {g:8s}: {r:.0%} of windows for 90% coverage → real-time factor "
                      f"{r * args.vlm_seconds / args.window_s:.2f} (below 1.0 = live)")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "gating_results.json"), "w") as f:
        json.dump({"rates": RATES, "results": res, "clips": names, "minutes": total_s / 60,
                   "windows": n_win, "actions": n_act, "window_s": args.window_s}, f, indent=1)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6.5, 4.5))
        for g in GATES:
            plt.plot([r * 100 for r in RATES], [c * 100 for c in res[g]["coverage"]],
                     label=f"{g} (AUC {res[g]['auc']:.2f})", lw=2.5 if g == "hoi" else 1.5)
        plt.xlabel("VLM calls (% of 4-s windows)")
        plt.ylabel("actions covered (%)")
        plt.title("Interaction-triggered VLM: coverage vs. compute")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, "gating_curve.png"), dpi=150)
        print(f"\nSaved {args.out}/gating_curve.png and gating_results.json")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
