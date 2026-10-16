# model/train.py
# Train HOI-TCN.
#
#   Stage 1, pre-train on EPIC-KITCHENS:
#     python -m model.train --features data/features/epic --out models/hoi_tcn_epic.pt
#   Stage 2, fine-tune on your own chest-camera footage:
#     python -m model.train --features data/features/own --init models/hoi_tcn_epic.pt \
#         --lr 3e-4 --out models/hoi_tcn.pt
#
# Validation is split BY VIDEO. The best checkpoint is chosen by event-level F1, and the
# decision threshold is tuned on the validation videos and stored in the checkpoint.

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.architecture import HOITCN, NUM_CLASSES
from model.data import WindowDataset, group_of, gt_events, load_videos, split_videos
from model.metrics import decode_events, match_events

THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


@torch.no_grad()
def predict_video(model, feats: np.ndarray, device="cpu", drop_groups=()) -> np.ndarray:
    model.eval()
    x = torch.from_numpy(feats).float().unsqueeze(0).to(device)
    ev, _ = model(x, drop_groups=drop_groups)
    return torch.softmax(ev[0], -1).cpu().numpy()


@torch.no_grad()
def predict_activity(model, feats: np.ndarray, device="cpu", score: str | None = None) -> np.ndarray:
    """Per-frame trigger score: 'event' = 1 - P(none) (pick/put), 'act' = P(any action),
    'max' = the larger of the two. Default: the model's own choice (made on validation people)."""
    model.eval()
    score = score or getattr(model, "gate_score", "event")
    x = torch.from_numpy(feats).float().unsqueeze(0).to(device)
    ev, _, act = model(x, return_act=True)
    event = 1.0 - torch.softmax(ev[0], -1)[:, 0].cpu().numpy()
    if act is None or score == "event":
        return event
    a = torch.sigmoid(act[0]).cpu().numpy()
    return a if score == "act" else np.maximum(a, event)


def evaluate_videos(model, vids, device="cpu", drop_groups=(), thresholds=THRESHOLDS, tol_s=1.0):
    """Event-level metrics at each threshold, summed over videos. Returns {thr: metrics}."""
    probs = [predict_video(model, v["feats"], device, drop_groups) for v in vids]
    results = {}
    for thr in thresholds:
        pred, gt = [], []
        offset = 0.0
        for v, p in zip(vids, probs):       # concatenate videos on one time axis for matching
            pred += [(t + offset, a) for t, a, _ in decode_events(p, v["times"], thr)]
            gt += [(s + offset, e + offset, a, o) for s, e, a, o in gt_events(v)]
            offset += float(v["times"][-1]) + 100.0 if len(v["times"]) else 100.0
        results[thr] = match_events(pred, gt, tol_s=tol_s)
    return results


def class_weights(vids, device):
    counts = np.bincount(np.concatenate([v["event"] for v in vids]), minlength=NUM_CLASSES).astype(float)
    w = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    return torch.tensor(w / w.mean(), dtype=torch.float32, device=device), counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="+", required=True, help="feature dirs (e.g. data/features/own)")
    ap.add_argument("--out", default="models/hoi_tcn.pt")
    ap.add_argument("--init", default=None, help="checkpoint to fine-tune from")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--samples-per-epoch", type=int, default=4000)
    ap.add_argument("--hold-weight", type=float, default=0.5)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--test-fraction", type=float, default=0.15)
    ap.add_argument("--val-videos", nargs="*", default=None, help="explicit validation video names")
    ap.add_argument("--test-videos", nargs="*", default=None, help="explicit test video names")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    vids = load_videos(args.features)
    train_v, val_v, test_v = split_videos(vids, args.val_fraction, args.test_fraction, args.seed,
                                          args.val_videos, args.test_videos)
    if not val_v:
        raise ValueError("Need at least 2 groups (people/videos) so one can be held out for validation.")
    print(f"[train] split by person/source video → train {len(train_v)}, val {len(val_v)}, "
          f"test {len(test_v)} videos")
    print(f"[train]   val groups:  {sorted({group_of(v) for v in val_v})}")
    print(f"[train]   test groups: {sorted({group_of(v) for v in test_v})}  (never used here)")

    weights, counts = class_weights(train_v, device)
    print(f"[train] frame counts none/pickup/putdown = {counts.astype(int).tolist()}, "
          f"class weights = {weights.cpu().numpy().round(2).tolist()}")

    if args.init:
        ckpt = torch.load(args.init, map_location=device, weights_only=False)
        hparams = ckpt["hparams"]
        model = HOITCN(**hparams).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[train] fine-tuning from {args.init}")
    else:
        hparams = {}
        model = HOITCN().to(device)
    print(f"[train] params={model.count_parameters():,} receptive field={model.receptive_field} frames")

    ds = WindowDataset(train_v, window=args.window, samples_per_epoch=args.samples_per_epoch, seed=args.seed)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_f1, best_thr, bad, history = -1.0, 0.5, 0, []
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for x, y, hold, hmask, lmask in dl:
            x, y, hold, hmask, lmask = (a.to(device) for a in (x, y, hold, hmask, lmask))
            ev, hl = model(x)
            ce = F.cross_entropy(ev.reshape(-1, NUM_CLASSES), y.reshape(-1), weight=weights, reduction="none")
            loss = (ce * lmask.reshape(-1)).sum() / lmask.sum().clamp(min=1)
            if hmask.sum() > 0:
                bce = F.binary_cross_entropy_with_logits(hl, hold, reduction="none")
                loss = loss + args.hold_weight * (bce * hmask).sum() / hmask.sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
            nb += 1
        sched.step()

        res = evaluate_videos(model, val_v, device)
        thr = max(res, key=lambda k: res[k]["overall"]["f1"])
        m = res[thr]
        history.append({"epoch": epoch, "loss": tot / nb, "thr": thr, **{a: m[a] for a in m}})
        print(f"ep {epoch:3d} loss {tot / nb:.4f} | val F1 {m['overall']['f1']:.3f} "
              f"(P {m['overall']['precision']:.3f} R {m['overall']['recall']:.3f}) @thr {thr} "
              f"| pickup F1 {m['pickup']['f1']:.3f} putdown F1 {m['putdown']['f1']:.3f} | {time.time() - t0:.0f}s")

        if m["overall"]["f1"] > best_f1:
            best_f1, best_thr, bad = m["overall"]["f1"], thr, 0
            torch.save({"model_state_dict": model.state_dict(), "hparams": hparams,
                        "threshold": thr, "val_metrics": m, "epoch": epoch,
                        "val_videos": [v["name"] for v in val_v],
                        "test_videos": [v["name"] for v in test_v],
                        "train_videos": [v["name"] for v in train_v],
                        "train_features": args.features},
                       args.out)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[train] early stop (no val improvement for {args.patience} epochs)")
                break

    with open(os.path.splitext(args.out)[0] + "_history.json", "w") as f:
        json.dump(history, f, indent=1)
    print(f"[train] best val event F1 = {best_f1:.3f} at threshold {best_thr} → {args.out}")
    print("[train] These are VALIDATION numbers (used to pick the checkpoint). Report test numbers:\n"
          f"        python -m model.evaluate --checkpoint {args.out} --features {' '.join(args.features)} --split test")


if __name__ == "__main__":
    main()
