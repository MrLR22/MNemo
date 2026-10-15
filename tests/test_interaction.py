import numpy as np

from modules.interaction import ModelEngine, RuleEngine
from tests.synth import det_at, hand_at, obs, pick_and_place


def run(seq):
    eng = RuleEngine()
    out = []
    for o in seq:
        out += eng.update(o)
    return out


def test_pickup_then_putdown():
    evs = run(pick_and_place())
    assert [(e.object, e.action) for e in evs] == [("wallet", "pickup"), ("wallet", "putdown")]
    pick, put = evs
    assert 2.5 <= pick.t <= 4.0                       # while moving together (frames 25-39)
    assert pick.location_t <= 2.5                     # location frame = where it was resting
    assert 4.5 <= put.t <= 6.0                        # after the hand leaves (frames 45-54)


def test_brush_is_not_an_event():
    """Hand sweeps past the wallet without moving it."""
    seq = []
    for i in range(40):
        hx = 0.1 + 0.03 * i
        seq.append(obs(i, [hand_at(hx, 0.7)], [det_at(0.5, 0.7)]))
    assert run(seq) == []


def test_nudge_is_not_a_pickup():
    """Hand touches the wallet, the wallet shifts once, hand leaves."""
    seq = []
    for i in range(40):
        ox = 0.5 if i < 15 else 0.52
        h = (0.5, 0.7) if 10 <= i < 20 else (0.2, 0.9)
        seq.append(obs(i, [hand_at(*h)], [det_at(ox, 0.7)]))
    assert run(seq) == []


def test_walking_with_hand_near_object_is_not_a_pickup():
    """Camera moves (wearer walks); hand and object shift together in the image only because of
    camera motion. Ego-motion compensation must cancel it."""
    seq = []
    for i in range(40):
        shift = 0.015 * i
        seq.append(obs(i, [hand_at(0.5 - shift, 0.72)], [det_at(0.5 - shift, 0.7)], ego=(-0.015, 0.0)))
    assert run(seq) == []


def test_without_ego_compensation_the_same_scene_would_fire():
    """Sanity check that the previous test is meaningful."""
    seq = []
    for i in range(40):
        shift = 0.015 * i
        seq.append(obs(i, [hand_at(0.5 - shift, 0.72)], [det_at(0.5 - shift, 0.7)], ego=(0.0, 0.0)))
    assert [e.action for e in run(seq)] == ["pickup"]


def test_keys_hidden_in_fist_counts_as_pickup():
    seq = []
    for i in range(40):
        if i < 10:
            seq.append(obs(i, [hand_at(0.5, 0.7)], [det_at(0.5, 0.7, obj="keys", s=0.03)]))
        else:   # keys disappear into the hand, hand moves away
            seq.append(obs(i, [hand_at(0.5 - 0.03 * (i - 9), 0.7)], []))
    evs = run(seq)
    assert [(e.object, e.action) for e in evs] == [("keys", "pickup")]
    assert evs[0].note == "object hidden in hand"


def test_two_objects_tracked_independently():
    a = pick_and_place("wallet")
    b = [obs(o.t * 10, o.hands, o.detections + [det_at(0.9, 0.3, obj="phone")]) for o in a]
    evs = run(b)
    assert {e.object for e in evs} == {"wallet"}


def test_model_engine_segments_and_holding_fallback():
    eng = ModelEngine(threshold=0.5)
    probs = np.zeros((60, 3))
    probs[:, 0] = 1.0
    probs[10:15] = [0.1, 0.8, 0.1]          # pickup segment
    probs[40:44] = [0.1, 0.1, 0.8]          # putdown segment (object not visible)
    out = []
    for i in range(60):
        out += eng.update(i / 10, probs[i], "wallet" if 10 <= i < 15 else None, 10 <= i < 15)
    assert [(e.object, e.action) for e in out] == [("wallet", "pickup"), ("wallet", "putdown")]
    assert out[1].note == "object not visible at put-down"
    assert eng.holding() == []
