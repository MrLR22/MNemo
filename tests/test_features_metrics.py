import numpy as np

from model.data import label_point_events
from model.metrics import decode_events, match_events
from modules.features import FEATURE_DIM, GROUPS, FeatureExtractor
from tests.synth import det_at, hand_at, obs, pick_and_place


def test_feature_shape_and_groups():
    fx = FeatureExtractor()
    feats = np.stack([fx(o) for o in pick_and_place()])
    assert feats.shape == (70, FEATURE_DIM) == (70, 105)
    assert GROUPS["ego"].stop == FEATURE_DIM
    assert np.isfinite(feats).all()
    assert feats[:, 90].min() == 1.0                 # object present in every frame


def test_ego_motion_is_removed_from_object_velocity():
    fx = FeatureExtractor()
    o = GROUPS["object"].start
    f = [fx(obs(i, [hand_at(0.5 - 0.02 * i, 0.8)], [det_at(0.5 - 0.02 * i, 0.6)], ego=(-0.02, 0)))
         for i in range(5)]
    assert np.allclose(f[-1][o + 8:o + 10], 0, atol=1e-5)


def test_no_hands_no_objects_is_all_zero_except_flags():
    fx = FeatureExtractor()
    f = fx(obs(0))
    assert not f.any()


def test_point_labels_and_holding():
    times = np.arange(0, 20, 0.1)
    lab, hold, gt = label_point_events(times, [(5.0, "pickup", "keys"), (12.0, "putdown", "keys")])
    assert set(lab[(times > 4.75) & (times < 5.65)]) == {1}
    assert set(lab[(times > 11.75) & (times < 12.65)]) == {2}
    assert hold[(times > 5.1) & (times < 11.9)].min() == 1.0
    assert hold[times < 4.9].max() == 0.0 and hold[times > 12.1].max() == 0.0


def test_decode_and_match():
    times = np.arange(0, 30, 0.1)
    p = np.zeros((len(times), 3))
    p[:, 0] = 1
    p[50:55] = [0.2, 0.8, 0.0]
    p[200:204] = [0.2, 0.0, 0.8]
    p[250:252] = [0.4, 0.6, 0.0]           # false pickup
    pred = decode_events(p, times, 0.5)
    assert [(round(t, 1), a) for t, a, _ in pred] == [(5.0, "pickup"), (20.0, "putdown"), (25.0, "pickup")]
    m = match_events([(t, a) for t, a, _ in pred],
                     [(5.2, 5.2, "pickup", "x"), (20.3, 20.3, "putdown", "x"), (28.0, 28.0, "putdown", "x")])
    assert m["pickup"]["tp"] == 1 and m["pickup"]["fp"] == 1
    assert m["putdown"]["tp"] == 1 and m["putdown"]["fn"] == 1
    assert m["overall"]["precision"] == round(2 / 3, 4)
