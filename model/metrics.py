# model/metrics.py
# Event decoding + event-level precision/recall/F1. Shared by training, model evaluation
# and full-pipeline evaluation so all numbers in the report are computed the same way.

import numpy as np

ACTIONS = ("pickup", "putdown")


def decode_events(probs: np.ndarray, times: np.ndarray, threshold: float, min_gap_s: float = 2.0):
    """
    probs (T, 3) softmax over {none, pickup, putdown}. A segment = consecutive frames with
    P(action) >= threshold; one event per segment at its peak. Events of the same action closer
    than min_gap_s are merged (higher peak wins). Returns [(t, action, peak_prob)].
    """
    events = []
    for k, action in enumerate(ACTIONS, start=1):
        p = probs[:, k]
        above = p >= threshold
        i, T = 0, len(p)
        segs = []
        while i < T:
            if above[i]:
                j = i
                while j + 1 < T and above[j + 1]:
                    j += 1
                m = i + int(np.argmax(p[i:j + 1]))
                segs.append((float(times[m]), action, float(p[m])))
                i = j + 1
            else:
                i += 1
        merged = []
        for s in segs:
            if merged and s[0] - merged[-1][0] < min_gap_s:
                if s[2] > merged[-1][2]:
                    merged[-1] = s
            else:
                merged.append(s)
        events += merged
    return sorted(events)


def match_events(pred, gt, tol_s: float = 1.0, use_object: bool = False):
    """
    pred: [(t, action[, object])]
    gt:   [(t_start, t_end, action[, object])]  (t_start == t_end for point labels)
    A prediction matches an unmatched GT of the same action (and object if use_object) when
    t is inside [t_start - tol, t_end + tol]. Greedy by time distance.
    Returns per-action and overall {tp, fp, fn, precision, recall, f1}.
    """
    used = [False] * len(gt)
    tp = {a: 0 for a in ACTIONS}
    fp = {a: 0 for a in ACTIONS}
    for p in sorted(pred, key=lambda x: x[0]):
        t, action = p[0], p[1]
        best, best_d = None, None
        for i, g in enumerate(gt):
            if used[i] or g[2] != action:
                continue
            if use_object and len(p) > 2 and len(g) > 3 and p[2] != g[3]:
                continue
            if g[0] - tol_s <= t <= g[1] + tol_s:
                d = 0.0 if g[0] <= t <= g[1] else min(abs(t - g[0]), abs(t - g[1]))
                if best is None or d < best_d:
                    best, best_d = i, d
        if best is None:
            fp[action] = fp.get(action, 0) + 1
        else:
            used[best] = True
            tp[action] += 1
    fn = {a: sum(1 for i, g in enumerate(gt) if g[2] == a and not used[i]) for a in ACTIONS}

    def prf(tp_, fp_, fn_):
        p = tp_ / (tp_ + fp_) if tp_ + fp_ else 0.0
        r = tp_ / (tp_ + fn_) if tp_ + fn_ else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return {"tp": tp_, "fp": fp_, "fn": fn_, "precision": round(p, 4), "recall": round(r, 4),
                "f1": round(f, 4)}

    out = {a: prf(tp[a], fp[a], fn[a]) for a in ACTIONS}
    out["overall"] = prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    return out
