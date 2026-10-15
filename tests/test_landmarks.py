import numpy as np

import config as C
from modules.landmarks import LandmarkIndex, check_index


def _unit(v):
    return v / np.linalg.norm(v)


def make_index(rng):
    centers = {"entryway_table": _unit(rng.normal(size=512)), "desk": _unit(rng.normal(size=512))}
    embs, labels, srcs = [], [], []
    for name, c in centers.items():
        for s in range(3):                       # 3 clips per landmark
            for _ in range(5):                   # 5 frames per clip
                embs.append(_unit(c + rng.normal(scale=0.02, size=512)))
                labels.append(name)
                srcs.append(f"{name}/clip{s}")
    return LandmarkIndex(np.array(embs), labels, srcs), centers


def test_resolve_known_and_unknown():
    rng = np.random.default_rng(0)
    idx, centers = make_index(rng)
    name, sim = idx.resolve(_unit(centers["desk"] + rng.normal(scale=0.02, size=512)))
    assert name == "desk" and sim > C.LANDMARK_MIN_SIM
    name, sim = idx.resolve(_unit(rng.normal(size=512)))       # a place never enrolled
    assert name is None


def test_save_load_roundtrip(tmp_path):
    idx, _ = make_index(np.random.default_rng(1))
    p = str(tmp_path / "idx.npz")
    idx.save(p)
    idx2 = LandmarkIndex.load(p)
    assert idx2.names == idx.names and np.allclose(idx2.embeddings, idx.embeddings)


def test_leave_one_source_out_check():
    idx, _ = make_index(np.random.default_rng(2))
    r = check_index(idx)
    assert r["top1_accuracy"] == 1.0 and r["unknown_rate"] == 0.0
    assert r["sources_per_landmark"] == {"desk": 3, "entryway_table": 3}
