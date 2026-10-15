# modules/features.py
# Per-frame feature vector for the HOI-TCN model.
# THE SAME CODE is used for EPIC-KITCHENS pre-training, fine-tuning on your own footage,
# and live inference, so the model always sees identically computed inputs.
#
# Input is a FrameObs (hands + detections + camera motion), never pixels.
#
# Layout (105 dims):
#   hands  [0:90]    2 slots (Left, Right) x 45:
#                    present, wrist x, wrist y, 20 landmarks relative to wrist / hand size (40),
#                    hand velocity relative to background (2)
#   object [90:103]  the tracked object nearest a hand:
#                    present, conf, offset to nearest wrist / hand size (2), box w, h,
#                    IoU with that hand box, distance / hand size, object velocity (2),
#                    object-minus-hand velocity (2), time since last seen
#   ego    [103:105] camera motion (background displacement)
#
# Velocities are displacement per 1/PROCESS_FPS seconds (normalised by the real time between frames,
# so videos with different frame rates give the same numbers) and scaled by VEL_SCALE.
# Background displacement ("ego") is subtracted so walking/turning does not look like hand motion.

import numpy as np

from config import PROCESS_FPS
from modules.types import FrameObs, Hand, Detection

VEL_SCALE = 10.0
MAX_REL = 5.0
MISSING_CAP_FRAMES = 30

HAND_DIM = 45
OBJ_DIM = 13
EGO_DIM = 2
FEATURE_DIM = 2 * HAND_DIM + OBJ_DIM + EGO_DIM          # 105
GROUPS = {
    "hands": slice(0, 2 * HAND_DIM),
    "object": slice(2 * HAND_DIM, 2 * HAND_DIM + OBJ_DIM),
    "ego": slice(2 * HAND_DIM + OBJ_DIM, FEATURE_DIM),
}


def box_iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def hand_slots(hands: list[Hand]) -> list[Hand | None]:
    """Put hands into fixed [Left, Right] slots. Duplicate labels fall back to x order."""
    slots: list[Hand | None] = [None, None]
    labels = [h.label for h in hands]
    if len(hands) == 2 and labels[0] == labels[1]:
        a, b = sorted(hands, key=lambda h: h.wrist[0])
        return [a, b]
    for h in hands[:2]:
        slots[0 if h.label == "Left" else 1] = h
    return slots


def nearest_hand(det: Detection, hands: list[Hand]) -> tuple[Hand | None, float]:
    """Nearest hand to a detection and the distance in units of that hand's size."""
    best, best_d = None, float("inf")
    for h in hands:
        d = float(np.linalg.norm(det.center - h.center)) / h.size
        if d < best_d:
            best, best_d = h, d
    return best, best_d


class FeatureExtractor:
    """Stateful (velocities need the previous frame). Call reset() between videos."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.prev_wrist: list[np.ndarray | None] = [None, None]
        self.prev_obj_center: dict[str, np.ndarray] = {}
        self.last_seen_frame: dict[str, int] = {}
        self.frame_no = 0
        self.prev_t: float | None = None
        self.focus_obj: str | None = None      # object the object-block describes this frame

    def __call__(self, obs: FrameObs) -> np.ndarray:
        f = np.zeros(FEATURE_DIM, dtype=np.float32)
        # displacement → velocity per 1/PROCESS_FPS s, whatever the actual gap between frames
        dt = (obs.t - self.prev_t) if self.prev_t is not None else 1.0 / PROCESS_FPS
        rate = float(np.clip((1.0 / PROCESS_FPS) / max(dt, 1e-3), 0.25, 4.0))
        self.prev_t = obs.t
        ego = np.asarray(obs.ego, dtype=np.float32)

        # ── hands ──
        hand_vel: list[np.ndarray] = [np.zeros(2), np.zeros(2)]
        for i, h in enumerate(hand_slots(obs.hands)):
            o = i * HAND_DIM
            if h is None:
                self.prev_wrist[i] = None
                continue
            f[o] = 1.0
            f[o + 1:o + 3] = h.wrist
            f[o + 3:o + 43] = ((h.landmarks[1:] - h.wrist) / h.size).reshape(-1)
            if self.prev_wrist[i] is not None:
                hand_vel[i] = ((h.wrist - self.prev_wrist[i]) - ego) * rate
                f[o + 43:o + 45] = hand_vel[i] * VEL_SCALE
            self.prev_wrist[i] = h.wrist.copy()

        # ── object nearest to a hand (or most confident if no hands) ──
        o = GROUPS["object"].start
        for d in obs.detections:
            self.last_seen_frame[d.obj] = self.frame_no
        chosen, chosen_hand, chosen_dist = None, None, MAX_REL
        if obs.detections:
            if obs.hands:
                for d in obs.detections:
                    h, dist = nearest_hand(d, obs.hands)
                    if chosen is None or dist < chosen_dist:
                        chosen, chosen_hand, chosen_dist = d, h, dist
            else:
                chosen = max(obs.detections, key=lambda d: d.conf)
        if chosen is not None:
            self.focus_obj = chosen.obj
        if chosen is not None:
            f[o] = 1.0
            f[o + 1] = chosen.conf
            if chosen_hand is not None:
                rel = (chosen.center - chosen_hand.wrist) / chosen_hand.size
                f[o + 2:o + 4] = np.clip(rel, -MAX_REL, MAX_REL) / MAX_REL
                f[o + 6] = box_iou(chosen.box, chosen_hand.box)
            f[o + 4] = chosen.box[2] - chosen.box[0]
            f[o + 5] = chosen.box[3] - chosen.box[1]
            f[o + 7] = min(chosen_dist, MAX_REL) / MAX_REL
            prev = self.prev_obj_center.get(chosen.obj)
            if prev is not None:
                obj_vel = ((chosen.center - prev) - ego) * rate
                f[o + 8:o + 10] = obj_vel * VEL_SCALE
                if chosen_hand is not None:
                    hv = hand_vel[0] if hand_slots(obs.hands)[0] is chosen_hand else hand_vel[1]
                    f[o + 10:o + 12] = (obj_vel - hv) * VEL_SCALE
        if self.focus_obj is not None:
            missing = self.frame_no - self.last_seen_frame.get(self.focus_obj, self.frame_no)
            f[o + 12] = min(missing, MISSING_CAP_FRAMES) / MISSING_CAP_FRAMES

        self.prev_obj_center = {d.obj: d.center.copy() for d in obs.detections}

        # ── camera motion ──
        f[GROUPS["ego"]] = ego * rate * VEL_SCALE

        self.frame_no += 1
        return f
