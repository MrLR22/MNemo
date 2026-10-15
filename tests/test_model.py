import os
import subprocess
import sys

import numpy as np
import torch

from model.architecture import HOITCN, StreamingModel, load_model
from model.data import label_point_events
from modules.features import FeatureExtractor
from tests.synth import det_at, hand_at, obs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_causal_and_streaming_equivalent():
    torch.manual_seed(0)
    m = HOITCN().eval()
    x = torch.randn(1, 100, 105)
    full, _ = m(x)
    x2 = x.clone()
    x2[:, 70:] = torch.randn(1, 30, 105)
    full2, _ = m(x2)
    assert torch.allclose(full[:, :70], full2[:, :70], atol=1e-5)     # future can't change the past
    s = StreamingModel(m)
    stream = torch.stack([s.step(x[0, t]) for t in range(100)])
    assert torch.allclose(stream, torch.softmax(full[0], -1), atol=1e-5)


def _synthetic_video(rng, n_events=6):
    """Feature sequence of a wallet being picked up and put down at random places."""
    fx = FeatureExtractor()
    feats, events, t = [], [], 0
    pos = np.array([0.5, 0.7])
    hand = np.array([0.2, 0.9])

    def step(h, o, held=False):
        nonlocal t
        noise = rng.normal(0, 0.002, 2)
        feats.append(fx(obs(t, [hand_at(*(h + noise))], [det_at(*(o + noise))])))
        t += 1

    for _ in range(n_events // 2):
        for k in range(int(rng.integers(10, 30))):           # idle
            step(hand + rng.normal(0, 0.003, 2), pos)
        for k in range(10):                                   # reach
            step(hand + (pos - hand) * (k + 1) / 10, pos)
        for k in range(4):
            step(pos, pos)
        events.append((t / 10.0, "pickup", "wallet"))
        target = rng.uniform([0.2, 0.4], [0.8, 0.8])
        for k in range(12):                                   # carry
            p = pos + (target - pos) * (k + 1) / 12
            step(p, p)
        pos = target
        for k in range(4):
            step(pos, pos)
        events.append((t / 10.0, "putdown", "wallet"))
        hand = rng.uniform([0.1, 0.8], [0.9, 0.95])
        for k in range(10):                                   # leave
            step(pos + (hand - pos) * (k + 1) / 10, pos)
    times = np.arange(len(feats), dtype=np.float32) / 10.0
    lab, hold, gt = label_point_events(times, events)
    return dict(feats=np.stack(feats), times=times, event=lab, hold=hold, hold_mask=np.ones_like(hold),
                focus=np.array(["wallet"] * len(times)), **gt)


def test_training_pipeline_learns_synthetic_task(tmp_path):
    """End-to-end: feature files → model.train → checkpoint → model.evaluate. The synthetic task
    is easy, so a working pipeline must reach a high event F1 on held-out videos."""
    rng = np.random.default_rng(0)
    d = tmp_path / "feats"
    d.mkdir()
    for i in range(10):                    # 10 "people", one video each
        np.savez(d / f"P{i:02d}__vid{i}.npz", **_synthetic_video(rng, n_events=8))
    ckpt = tmp_path / "m.pt"
    r = subprocess.run([sys.executable, "-m", "model.train", "--features", str(d), "--out", str(ckpt),
                        "--epochs", "12", "--samples-per-epoch", "600", "--device", "cpu"],
                       cwd=ROOT, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
    c = torch.load(ckpt, weights_only=False)
    assert c["val_metrics"]["overall"]["f1"] >= 0.8, r.stdout[-1500:]
    assert len(c["test_videos"]) >= 1 and not set(c["test_videos"]) & set(c["train_videos"])
    m = load_model(str(ckpt))
    assert m.threshold == c["threshold"]
    r = subprocess.run([sys.executable, "-m", "model.evaluate", "--checkpoint", str(ckpt), "--features", str(d),
                        "--split", "test", "--json", str(tmp_path / "res.json")],
                       cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    import json
    res = json.load(open(tmp_path / "res.json"))
    assert res["full model"]["overall"]["f1"] >= 0.8, r.stdout       # held-out test videos
    r = subprocess.run([sys.executable, "-m", "model.evaluate", "--checkpoint", str(ckpt), "--features", str(d),
                        "--split", "all"], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert "WARNING" in r.stdout          # evaluating on training videos must warn
