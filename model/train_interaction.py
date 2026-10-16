# model/train_interaction.py
# Teach the trigger ANY hand action, not only pick-up / put-down.
#
# Why: the first HOI-TCN only learned take/put, so cutting, stirring, sliding food or washing scored
# low and the trigger skipped those windows (seen on EPIC: chopping chilli scored 0.2-0.5).
#
# What: start from the trained HOI-TCN, add a small "act" head (one output per frame: "the hands are
# doing some annotated action"), and fine-tune on the SAME EPIC feature files with labels from ALL
# EPIC-100 verbs. The pick/put head keeps training, so it is not forgotten.
# The train / validation / test people are copied from the original checkpoint (test people stay unseen).
# Which score the trigger should use ("event" = old, "act" = new, "max" = both) is chosen on the
# VALIDATION people by window coverage, then reported on the test people.
#
#   python -m model.train_interaction --features data/drive_features/features/epic \
#       --annotations EPIC_100_train.csv --init models/hoi_tcn.pt --out models/hoi_tcn_act.pt

import argparse
import csv
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from evaluation.gating_experiment import coverage, select
from model.architecture import HOITCN, NUM_CLASSES
from model.data import WindowDataset, load_videos
from model.train import class_weights, evaluate_videos, predict_activity
from modules.hoi_gate import window_hoi_scores

RATES = (0.3, 0.4, 0.5, 0.6, 0.7)
WINDOW_S = 4.0


def _ts(s: str) -> float:
    h, m, x = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(x)


def load_all_actions(path: str) -> dict[str, list[tuple]]:
    out = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            out[r["video_id"]].append((_ts(r["start_timestamp"]), _ts(r["stop_timestamp"])))
    return out


def exact_start(v: dict, actions: list[tuple]) -> float | None:
    """The clip's true start in the source video. File names keep only whole seconds, but the stored
    pick/put ends (gt_end, relative) must line up with EPIC's absolute stop times: pick the start that
    matches the most of them."""
    w = float(v["name"].rpartition("_w")[2])
    stops = np.array([e for _, e in actions])
    gt = np.asarray(v["gt_end"], float)
    if len(gt) == 0 or len(stops) == 0:
        return None
    cands = {round(e - g, 3) for g in gt for e in stops if w - 0.01 <= e - g < w + 1.0}
    best, best_n = None, 0
    for t0 in cands:
        n = sum(np.any(np.abs(stops - t0 - g) < 0.02) for g in gt)
        if n > best_n:
            best, best_n = t0, n
    return best if best_n >= max(1, 0.8 * len(gt)) else None


def act_labels(v: dict, actions: list[tuple], t0: float) -> np.ndarray:
    t = v["times"]
    y = np.zeros(len(t), np.float32)
    for s, e in actions:
        y[(t >= s - t0) & (t <= e - t0)] = 1.0
    return y


def window_coverage(model, vids, score: str, rates=RATES) -> dict[float, float]:
    """Share of ALL annotated actions that overlap a selected 4-s window, at each call rate."""
    sc, acts = [], []
    for v in vids:
        p = predict_activity(model, v["feats"], score=score)
        sc.append(window_hoi_scores(v["times"], p, WINDOW_S))
        acts.append(v["_spans"])
    return {r: coverage(select(sc, "hoi", r), acts, WINDOW_S) for r in rates}


def uniform_coverage(vids, rates=RATES) -> dict[float, float]:
    sc = [np.zeros(int(np.floor(v["times"][-1] / WINDOW_S)) + 1) for v in vids]
    return {r: coverage(select(sc, "uniform", r), [v["_spans"] for v in vids], WINDOW_S) for r in rates}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="folder with the EPIC feature files (<P>__<video>_w<start>.npz)")
    ap.add_argument("--annotations", required=True, help="EPIC_100_train.csv")
    ap.add_argument("--init", required=True, help="the trained pick/put HOI-TCN (its people split is reused)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--act-weight", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    ckpt = torch.load(args.init, map_location="cpu", weights_only=False)
    split = {n: k for k in ("train", "val", "test") for n in ckpt[f"{k}_videos"]}
    acts = load_all_actions(args.annotations)
    vids, skipped = [], []
    for v in load_videos([args.features]):
        video = v["name"].split("__", 1)[1].rpartition("_w")[0]
        t0 = exact_start(v, acts.get(video, []))
        if t0 is None or v["name"] not in split:
            skipped.append(v["name"])
            continue
        v["act"] = act_labels(v, acts[video], t0)
        dur = float(v["times"][-1])
        v["_spans"] = [(max(0.0, s - t0), min(dur, e - t0), "") for s, e in acts[video] if e - t0 > 0 and s - t0 < dur]
        v["_split"] = split[v["name"]]
        vids.append(v)
    tr = [v for v in vids if v["_split"] == "train"]
    va = [v for v in vids if v["_split"] == "val"]
    te = [v for v in vids if v["_split"] == "test"]
    pos = np.concatenate([v["act"] for v in tr]).mean()
    print(f"pieces: train {len(tr)}, val {len(va)}, test {len(te)} (skipped {len(skipped)}: {skipped[:5]})")
    print(f"train frames inside an annotated action: {pos:.0%}")

    hp = dict(ckpt["hparams"], act_head=True)
    model = HOITCN(**hp)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    assert all(k.startswith("act_head") for k in missing) and not unexpected, (missing, unexpected)
    weights, _ = class_weights(tr, "cpu")
    pos_w = torch.tensor((1 - pos) / max(pos, 1e-3))
    ds = WindowDataset(tr, samples_per_epoch=4000, seed=args.seed, with_act=True)
    dl = DataLoader(ds, batch_size=32)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)

    def val_score(m):
        cov = window_coverage(m, va, "act")
        return float(np.mean(list(cov.values()))), cov

    base_ev = evaluate_videos(model, va)
    print(f"before: val pick/put F1 {max(r['overall']['f1'] for r in base_ev.values()):.3f}, "
          f"val coverage (old score) {np.mean(list(window_coverage(model, va, 'event').values())):.3f}")
    best, bad, history = -1.0, 0, []
    for ep in range(1, args.epochs + 1):
        model.train()
        t, tot = time.time(), 0.0
        for x, y, hold, hmask, lmask, act, amask in dl:
            ev, _, al = model(x, return_act=True)
            ce = F.cross_entropy(ev.reshape(-1, NUM_CLASSES), y.reshape(-1), weight=weights, reduction="none")
            loss = (ce * lmask.reshape(-1)).sum() / lmask.sum().clamp(min=1)
            bce = F.binary_cross_entropy_with_logits(al, act, pos_weight=pos_w, reduction="none")
            loss = loss + args.act_weight * (bce * amask).sum() / amask.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        sc, cov = val_score(model)
        f1 = max(r["overall"]["f1"] for r in evaluate_videos(model, va).values())
        history.append({"epoch": ep, "loss": tot / len(dl), "val_coverage": sc, "val_f1": f1})
        print(f"ep {ep:2d} loss {tot / len(dl):.3f} | val coverage (act) {sc:.3f} "
              + " ".join(f"{int(r * 100)}%:{c:.2f}" for r, c in cov.items()) + f" | val pick/put F1 {f1:.3f} | {time.time() - t:.0f}s")
        if sc > best:
            best, bad = sc, 0
            torch.save({"model_state_dict": model.state_dict(), "hparams": hp, "threshold": ckpt.get("threshold"),
                        "epoch": ep, "train_videos": ckpt["train_videos"], "val_videos": ckpt["val_videos"],
                        "test_videos": ckpt["test_videos"], "init": args.init}, args.out)
        else:
            bad += 1
            if bad >= args.patience:
                break

    # choose the trigger score on VALIDATION people, then report everything on TEST people
    final = HOITCN(**hp)
    saved = torch.load(args.out, map_location="cpu", weights_only=False)
    final.load_state_dict(saved["model_state_dict"])
    old = HOITCN(**ckpt["hparams"])
    old.load_state_dict(ckpt["model_state_dict"])
    val = {s: float(np.mean(list(window_coverage(final, va, s).values()))) for s in ("event", "act", "max")}
    choice = max(val, key=val.get)
    saved["gate_score"] = choice
    saved["val_gate_coverage"] = val
    rows = {"every-nth window": uniform_coverage(te), "old trigger (pick/put)": window_coverage(old, te, "event"),
            f"new trigger ({choice})": window_coverage(final, te, choice)}
    saved["test_gate_coverage"] = {k: {str(r): c for r, c in v.items()} for k, v in rows.items()}
    test_f1 = {n: max(r["overall"]["f1"] for r in evaluate_videos(m, te).values()) for n, m in (("old", old), ("new", final))}
    saved["test_pickput_f1"] = test_f1
    torch.save(saved, args.out)
    with open(os.path.splitext(args.out)[0] + "_report.json", "w") as f:
        json.dump({"history": history, "val_gate_coverage": val, "gate_score": choice,
                   "test_gate_coverage": saved["test_gate_coverage"], "test_pickput_f1": test_f1}, f, indent=1)

    print(f"\nvalidation coverage by score: " + ", ".join(f"{k} {v:.3f}" for k, v in val.items()) + f" → use '{choice}'")
    print(f"\nTEST people {sorted({v['name'].split('__')[0] for v in te})}: share of ALL annotated actions "
          f"inside the windows sent to the VLM")
    print(f"{'':28s}" + "".join(f"{int(r * 100):>6d}%" for r in RATES) + "   (VLM calls)")
    for k, v in rows.items():
        print(f"{k:28s}" + "".join(f"{c:7.0%}" for c in v.values()))
    print(f"\npick/put event F1 on test people: old {test_f1['old']:.3f}, new {test_f1['new']:.3f} (must not drop much)")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
