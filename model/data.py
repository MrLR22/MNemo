# model/data.py
# Frame labels + training windows for HOI-TCN.
#
# Each video becomes one .npz in data/features/<dataset>/:
#   feats      (N, 105) float32   from modules/features.py
#   times      (N,)     seconds
#   event      (N,)     0 none / 1 pickup / 2 putdown
#   hold       (N,)     1 if a tracked object is in hand (own data only)
#   hold_mask  (N,)     1 where `hold` is known (0 for EPIC-KITCHENS)
#   gt_start, gt_end, gt_action, gt_object   ground-truth events for event-level metrics
#
# Labelling rule (same idea for both datasets):
#   own footage  : you mark the instant of each pick-up/put-down (tools/label_events.py);
#                  frames in [t - OWN_PRE, t + OWN_POST] get that class.
#   EPIC-KITCHENS: "take"/"put" segments [start, stop]; frames in the last EPIC_TAIL_S seconds
#                  of the segment (+ OWN_POST after it) get the class. The end of a take/put
#                  segment is where the object leaves/arrives, which matches the instant you
#                  mark on your own footage. This is a design choice: check it on a few clips.

import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

OWN_PRE, OWN_POST = 0.3, 0.7
EPIC_TAIL_S = 1.0
CLASS_ID = {"pickup": 1, "putdown": 2}


def label_point_events(times, events):
    """events: [(t, action, object)] → (event labels, hold, gt arrays)."""
    lab = np.zeros(len(times), np.int64)
    for t, action, _ in sorted(events):
        lab[(times >= t - OWN_PRE) & (times <= t + OWN_POST)] = CLASS_ID[action]

    hold = np.zeros(len(times), np.float32)
    by_obj = {}
    for t, action, obj in sorted(events):
        by_obj.setdefault(obj, []).append((t, action))
    for obj, evs in by_obj.items():
        start = times[0] if evs[0][1] == "putdown" else None    # already holding at video start
        for t, action in evs:
            if action == "pickup":
                start = t
            elif start is not None:
                hold[(times >= start) & (times < t)] = 1.0
                start = None
        if start is not None:
            hold[times >= start] = 1.0
    gt = {
        "gt_start": np.array([e[0] for e in events], np.float32),
        "gt_end": np.array([e[0] for e in events], np.float32),
        "gt_action": np.array([e[1] for e in events]),
        "gt_object": np.array([e[2] for e in events]),
    }
    return lab, hold, gt


def label_segments(times, segments):
    """segments: [(start, stop, action, noun)] (EPIC) → (event labels, gt arrays)."""
    lab = np.zeros(len(times), np.int64)
    for s, e, action, _ in sorted(segments):
        lab[(times >= max(s, e - EPIC_TAIL_S)) & (times <= e + OWN_POST)] = CLASS_ID[action]
    gt = {
        "gt_start": np.array([max(s, e - EPIC_TAIL_S) for s, e, _, _ in segments], np.float32),
        "gt_end": np.array([e for _, e, _, _ in segments], np.float32),
        "gt_action": np.array([a for _, _, a, _ in segments]),
        "gt_object": np.array([n for _, _, _, n in segments]),
    }
    return lab, gt


def load_videos(feature_dirs: list[str]) -> list[dict]:
    vids = []
    for d in feature_dirs:
        for path in sorted(glob.glob(os.path.join(d, "*.npz"))):
            z = np.load(path, allow_pickle=False)
            v = {k: z[k] for k in z.files}
            v["name"] = os.path.splitext(os.path.basename(path))[0]
            v["path"] = path
            vids.append(v)
    if not vids:
        raise FileNotFoundError(f"No feature files in {feature_dirs}. Run model/extract_features.py first.")
    return vids


def gt_events(v: dict) -> list[tuple]:
    return [(float(s), float(e), str(a), str(o)) for s, e, a, o in
            zip(v["gt_start"], v["gt_end"], v["gt_action"], v["gt_object"])]


def group_of(v: dict) -> str:
    """Files are named <group>__<video>; the group is the person (EPIC participant, Charades subject)
    or source video (Ego4D video_uid). Videos without '__' are their own group."""
    return v["name"].split("__")[0]


def dataset_of(v: dict) -> str:
    return os.path.basename(os.path.dirname(v["path"]))


def split_videos(vids: list[dict], val_fraction: float = 0.15, test_fraction: float = 0.15,
                 seed: int = 0, val_names=None, test_names=None):
    """
    Split into train / val / test by GROUP, separately within each dataset, so the same person or
    source video never appears in two splits (no leakage), and every dataset is represented in each
    split when it has >= 3 groups. Explicit names override the random split.
    """
    if val_names or test_names:
        val_names, test_names = set(val_names or []), set(test_names or [])
        val = [v for v in vids if v["name"] in val_names]
        test = [v for v in vids if v["name"] in test_names]
        train = [v for v in vids if v["name"] not in val_names | test_names]
        return train, val, test
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for ds in sorted({dataset_of(v) for v in vids}):
        dv = [v for v in vids if dataset_of(v) == ds]
        groups = sorted({group_of(v) for v in dv})
        rng.shuffle(groups)
        n = len(groups)
        n_test = max(1, round(n * test_fraction)) if n >= 3 and test_fraction > 0 else 0
        n_val = max(1, round(n * val_fraction)) if n >= 2 and val_fraction > 0 else 0
        test_g, val_g = set(groups[:n_test]), set(groups[n_test:n_test + n_val])
        for v in dv:
            g = group_of(v)
            (test if g in test_g else val if g in val_g else train).append(v)
    return train, val, test


class WindowDataset(Dataset):
    """
    Random training windows. Half of the windows are centred near an event so the rare
    pickup/putdown frames are seen often enough. The first WARMUP frames of each window get
    no loss (the model has no past context there, unlike at inference time).
    """

    WARMUP = 20

    def __init__(self, vids: list[dict], window: int = 128, samples_per_epoch: int = 4000,
                 event_fraction: float = 0.5, seed: int = 0, with_act: bool = False):
        self.with_act = with_act          # also return v["act"] (any-action labels) + its mask
        self.vids = [v for v in vids if len(v["feats"]) >= 8]
        self.window = window
        self.n = samples_per_epoch
        self.event_fraction = event_fraction
        self.rng = np.random.default_rng(seed)
        self.event_pos = [(vi, i) for vi, v in enumerate(self.vids) for i in np.flatnonzero(v["event"] > 0)]
        lengths = np.array([len(v["feats"]) for v in self.vids], float)
        self.vid_p = lengths / lengths.sum()

    def __len__(self):
        return self.n

    def __getitem__(self, _):
        W = self.window
        if self.event_pos and self.rng.random() < self.event_fraction:
            vi, pos = self.event_pos[self.rng.integers(len(self.event_pos))]
            start = pos - self.rng.integers(self.WARMUP + 5, W - 5)
        else:
            vi = self.rng.choice(len(self.vids), p=self.vid_p)
            start = self.rng.integers(-W // 2, max(1, len(self.vids[vi]["feats"]) - W // 2))
        v = self.vids[vi]
        N = len(v["feats"])
        start = int(np.clip(start, 0, max(0, N - W)))
        sl = slice(start, start + W)
        x = np.zeros((W, v["feats"].shape[1]), np.float32)
        y = np.zeros(W, np.int64)
        hold = np.zeros(W, np.float32)
        hmask = np.zeros(W, np.float32)
        lmask = np.zeros(W, np.float32)
        n = min(W, N - start)
        x[:n], y[:n] = v["feats"][sl], v["event"][sl]
        hold[:n], hmask[:n] = v["hold"][sl], v["hold_mask"][sl]
        lmask[:n] = 1.0
        if start > 0:
            lmask[:self.WARMUP] = 0.0
            hmask[:self.WARMUP] = 0.0
        out = (torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(hold),
               torch.from_numpy(hmask), torch.from_numpy(lmask))
        if self.with_act:
            act = np.zeros(W, np.float32)
            amask = np.zeros(W, np.float32)
            if "act" in v:
                act[:n] = v["act"][sl]
                amask[:n] = lmask[:n]
            out += (torch.from_numpy(act), torch.from_numpy(amask))
        return out
