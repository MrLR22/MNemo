import numpy as np

from evaluation.gating_experiment import RATES, clip_actions, coverage, curves, select, window_scores
from tools.fetch_epic_windows import plan_random


def test_clip_actions_retimed_and_clipped():
    acts = [(95, 105, "open fridge"), (110, 115, "close fridge"), (300, 310, "wash cup")]
    assert clip_actions(acts, t0=100, duration=60) == [(0.0, 5, "open fridge"), (10, 15, "close fridge")]


def test_select_rates_and_uniform_spread():
    sc = [np.arange(10, dtype=float), np.arange(10, dtype=float)]
    sel = select(sc, "hoi", 0.25)
    assert sum(s.sum() for s in sel) == 5                       # 25% of 20 windows, globally ranked
    u = select(sc, "uniform", 0.5)
    assert all(s.sum() == 5 for s in u) and not u[0][0:2].all()  # every other window


def test_coverage_counts_overlapping_windows():
    sel = [np.array([False, True, False, False])]               # window 1 = [4, 8) s
    acts = [[(5, 6, "a"), (9, 10, "b"), (3, 4.5, "c")]]          # a: inside, b: not, c: overlaps window 1
    assert coverage(sel, acts, 4.0) == 2 / 3


def test_informative_gate_beats_uniform_and_random():
    """Actions happen only in 20% of windows; a gate that sees them reaches 100% coverage at 20% calls."""
    rng = np.random.default_rng(0)
    scores = {g: [] for g in ("hoi", "hands", "motion", "uniform", "random")}
    actions = []
    for v in range(5):
        n = 50
        active = rng.random(n) < 0.2
        acts = [(w * 4 + 1, w * 4 + 2, "x") for w in np.flatnonzero(active)]
        good = active.astype(float) + rng.random(n) * 0.1
        for g in scores:
            scores[g].append(good if g == "hoi" else rng.random(n))
        actions.append(acts)
    res = curves(scores, actions, 4.0)
    assert res["hoi"]["rate_for_95pct"] <= 0.25
    assert res["hoi"]["auc"] > res["uniform"]["auc"] and res["hoi"]["auc"] > res["random"]["auc"]
    assert res["uniform"]["coverage"][-1] == 1.0                # all windows → everything covered


def test_window_scores_shapes():
    times = np.arange(0, 12, 0.1)
    feats = np.zeros((len(times), 105), np.float32)
    feats[40:60, 0] = 1.0                                        # hand visible in window 1 only
    probs = np.tile([1.0, 0.0, 0.0], (len(times), 1))
    probs[45, :] = [0.2, 0.7, 0.1]
    s = window_scores(times, feats, probs, 4.0, np.random.default_rng(0))
    assert len(s["hoi"]) == 3 and s["hoi"].argmax() == 1 and s["hands"].argmax() == 1


def test_plan_random_no_overlap_and_spread():
    segs = {f"P{p:02d}_0{v}": [(t, t + 1, "pickup", "x") for t in range(0, 1200, 30)] for p in (1, 2, 3) for v in (1, 2)}
    dur = {k: 1200.0 for k in segs}
    w = plan_random(segs, dur, n_windows=6, window_s=600, seed=0)
    assert len(w) == 6 and {x["video"][:3] for x in w} == {"P01", "P02", "P03"}
    for a in w:
        for b in w:
            if a is not b and a["video"] == b["video"]:
                assert abs(a["start"] - b["start"]) >= 600
