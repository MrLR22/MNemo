# tests/synth.py
# Synthetic hand/object tracks for testing the interaction logic without video.

import numpy as np

from modules.types import Detection, FrameObs, Hand

FPS = 10.0
HAND_W = 0.10           # hand landmark box width/height (normalised)


def hand_at(cx, cy, label="Right", w=HAND_W):
    """21 landmarks spread over a w x w box centred at (cx, cy); wrist = bottom-centre."""
    rng = np.random.default_rng(0)
    pts = rng.uniform(-w / 2, w / 2, size=(21, 2))
    pts[0] = [0.0, w / 2]                    # wrist at the bottom
    pts[1] = [-w / 2, -w / 2]                # make sure the box is exactly w x w
    pts[2] = [w / 2, w / 2]
    pts += [cx, cy]
    return Hand(label=label, landmarks=pts.astype(np.float32), score=0.95)


def det_at(cx, cy, obj="wallet", s=0.06, conf=0.8):
    return Detection(obj=obj, conf=conf, box=(cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2))


def obs(i, hands=(), dets=(), ego=(0.0, 0.0)):
    return FrameObs(t=i / FPS, hands=list(hands), detections=list(dets), ego=np.array(ego, float))


def lerp(a, b, k, n):
    return a + (b - a) * min(k, n) / n


def pick_and_place(obj="wallet"):
    """
    0-9   wallet still at (0.5, 0.7), hand far away
    10-19 hand moves onto the wallet
    20-24 hand rests on the wallet (contact)
    25-39 hand and wallet move together to (0.3, 0.5)            → pickup
    40-44 resting at the new place, still in hand
    45-54 hand leaves upward-right, wallet stays at (0.3, 0.5)    → putdown
    55-69 nothing happens
    """
    seq = []
    ox, oy = 0.5, 0.7
    for i in range(70):
        if i < 10:
            h, o = (0.15, 0.9), (ox, oy)
        elif i < 20:
            h, o = (lerp(0.15, ox, i - 10, 10), lerp(0.9, oy, i - 10, 10)), (ox, oy)
        elif i < 25:
            h, o = (ox, oy), (ox, oy)
        elif i < 40:
            p = (lerp(ox, 0.3, i - 25, 15), lerp(oy, 0.5, i - 25, 15))
            h, o = p, p
        elif i < 45:
            h, o = (0.3, 0.5), (0.3, 0.5)
        elif i < 55:
            h, o = (lerp(0.3, 0.7, i - 45, 10), lerp(0.5, 0.85, i - 45, 10)), (0.3, 0.5)
        else:
            h, o = (0.7, 0.85), (0.3, 0.5)
        seq.append(obs(i, [hand_at(*h)], [det_at(*o, obj=obj)]))
    return seq
