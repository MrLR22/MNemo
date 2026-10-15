# modules/interaction.py
# M3: decides pick-up vs put-down over time (brief section 6).
#
#   IDLE ──(contact for CONTACT_FRAMES, then object moves WITH the hand)──► HOLDING   → "pickup"
#   IDLE ──(contact, then object vanishes into the grip as the hand moves away)──► HOLDING → "pickup"
#   HOLDING ──(object visible, still, and hand gone for RELEASE_FRAMES)──► IDLE      → "putdown"
#
# Two engines share one output format (list[PendingEvent]):
#   RuleEngine   hand-written state machine (baseline, no training needed)
#   ModelEngine  HOI-TCN probabilities + peak picking (learned)

from collections import deque
from dataclasses import dataclass, field

import numpy as np

import config as C
from modules.features import nearest_hand
from modules.types import FrameObs, Hand


@dataclass
class PendingEvent:
    object: str
    action: str             # "pickup" | "putdown"
    t: float                # when the decision was made (video seconds)
    location_t: float       # which frame shows the object's resting place
    confidence: float
    note: str | None = None


def _hand_by_label(hands: list[Hand], label: str) -> Hand | None:
    for h in hands:
        if h.label == label:
            return h
    return None


@dataclass
class _ObjTrack:
    state: str = "IDLE"
    prev_center: np.ndarray | None = None
    contact_frames: int = 0
    contact_start_t: float = 0.0
    together_frames: int = 0
    gap_frames: int = 0              # frames since the object was last detected
    last_center: np.ndarray | None = None
    last_contact_hand: str | None = None
    last_contact_wrist: np.ndarray | None = None
    release_frames: int = 0
    confs: deque = field(default_factory=lambda: deque(maxlen=20))


class RuleEngine:
    def __init__(self, objects: list[str] | None = None):
        self.objects = objects or list(C.TARGET_OBJECTS)
        self.reset()

    def reset(self):
        self.tracks = {o: _ObjTrack() for o in self.objects}
        self.prev_wrists: dict[str, np.ndarray] = {}
        self.prev_t: float | None = None
        self.last_fired: dict[tuple[str, str], float] = {}

    def holding(self) -> list[str]:
        return [o for o, tr in self.tracks.items() if tr.state == "HOLDING"]

    def _fire(self, out, obj, action, t, location_t, tr, note=None):
        key = (obj, action)
        if t - self.last_fired.get(key, -1e9) < C.EVENT_COOLDOWN_S:
            return
        self.last_fired[key] = t
        conf = float(np.mean(tr.confs)) if tr.confs else 0.5
        out.append(PendingEvent(obj, action, t, location_t, round(conf, 3), note))

    def update(self, obs: FrameObs) -> list[PendingEvent]:
        out: list[PendingEvent] = []
        ego = np.asarray(obs.ego)
        # speeds per 1/PROCESS_FPS s, independent of the actual frame gap (same as features.py)
        dt = (obs.t - self.prev_t) if self.prev_t is not None else 1.0 / C.PROCESS_FPS
        rate = float(np.clip((1.0 / C.PROCESS_FPS) / max(dt, 1e-3), 0.25, 4.0))
        self.prev_t = obs.t

        hand_vel = {}
        for h in obs.hands:
            prev = self.prev_wrists.get(h.label)
            hand_vel[h.label] = None if prev is None else ((h.wrist - prev) - ego) * rate
        self.prev_wrists = {h.label: h.wrist.copy() for h in obs.hands}

        dets = {d.obj: d for d in obs.detections}
        for obj, tr in self.tracks.items():
            det = dets.get(obj)
            if det is None:
                tr.gap_frames += 1
                tr.prev_center = None
                tr.together_frames = 0
                tr.release_frames = 0
                self._check_occluded_grasp(obj, tr, obs, out)
                continue

            tr.confs.append(det.conf)
            obj_vel = None if tr.prev_center is None else ((det.center - tr.prev_center) - ego) * rate
            tr.prev_center = det.center.copy()
            tr.last_center = det.center.copy()
            tr.gap_frames = 0

            hand, dist = nearest_hand(det, obs.hands)
            in_contact = hand is not None and dist < C.CONTACT_DIST

            if tr.state == "IDLE":
                if in_contact:
                    if tr.contact_frames == 0:
                        tr.contact_start_t = obs.t
                    tr.contact_frames += 1
                    tr.last_contact_hand = hand.label
                    tr.last_contact_wrist = hand.wrist.copy()
                    hv = hand_vel.get(hand.label)
                    if self._moving_together(obj_vel, hv):
                        tr.together_frames += 1
                    else:
                        tr.together_frames = 0
                    if (tr.contact_frames >= C.CONTACT_FRAMES
                            and tr.together_frames >= C.MOVE_TOGETHER_FRAMES):
                        tr.state = "HOLDING"
                        tr.release_frames = 0
                        self._fire(out, obj, "pickup", obs.t, tr.contact_start_t, tr)
                else:
                    tr.contact_frames = 0
                    tr.together_frames = 0

            else:   # HOLDING
                # hysteresis: needs to be clearly away from the hand to count as released
                released = hand is None or dist > C.CONTACT_DIST * 1.5
                still = obj_vel is not None and np.linalg.norm(obj_vel) < C.STILL_SPEED
                if released and still:
                    tr.release_frames += 1
                else:
                    tr.release_frames = 0
                if tr.release_frames >= C.RELEASE_FRAMES:
                    tr.state = "IDLE"
                    tr.contact_frames = 0
                    tr.together_frames = 0
                    self._fire(out, obj, "putdown", obs.t, obs.t, tr)
        return out

    @staticmethod
    def _moving_together(obj_vel, hand_vel) -> bool:
        if obj_vel is None or hand_vel is None:
            return False
        so, sh = np.linalg.norm(obj_vel), np.linalg.norm(hand_vel)
        if so < C.MOVE_MIN_SPEED or sh < C.MOVE_MIN_SPEED:
            return False
        return float(obj_vel @ hand_vel / (so * sh)) > C.MOVE_DIR_COS

    def _check_occluded_grasp(self, obj, tr: _ObjTrack, obs: FrameObs, out):
        """Small objects (keys) disappear inside the fist. If the object vanished right after a
        contact streak and the same hand has since moved away from where the object was, treat
        it as picked up."""
        if tr.state != "IDLE":
            return
        if tr.gap_frames > C.OCCLUDED_GRASP_FRAMES * 3:
            tr.contact_frames = 0       # contact is too old to explain the disappearance
            return
        if tr.contact_frames < C.CONTACT_FRAMES or tr.last_contact_wrist is None:
            return
        hand = _hand_by_label(obs.hands, tr.last_contact_hand or "")
        if hand is None:
            return
        moved = np.linalg.norm(hand.wrist - tr.last_contact_wrist) / hand.size
        if tr.gap_frames >= C.OCCLUDED_GRASP_FRAMES and moved > 1.0:
            tr.state = "HOLDING"
            tr.release_frames = 0
            self._fire(out, obj, "pickup", obs.t, tr.contact_start_t, tr,
                       note="object hidden in hand")


class ModelEngine:
    """
    Turns per-frame HOI-TCN probabilities into events.
    A segment starts when P(action) rises above the threshold and ends when it falls below;
    one event is emitted per segment with confidence = peak probability.
    The object is the tracked object nearest the hand during the segment; for a put-down with
    no visible object we fall back to the object currently held.
    """

    ACTIONS = ("pickup", "putdown")      # model classes 1 and 2 (0 = none)

    def __init__(self, threshold: float = C.EVENT_MODEL_THRESHOLD):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.active = {a: None for a in self.ACTIONS}   # action -> segment dict
        self.held: list[str] = []
        self.last_fired: dict[tuple[str, str], float] = {}

    def holding(self) -> list[str]:
        return list(self.held)

    def update(self, t: float, probs: np.ndarray, focus_obj: str | None,
               obj_near_hand: bool, final: bool = False) -> list[PendingEvent]:
        out = []
        for k, action in enumerate(self.ACTIONS, start=1):
            p = float(probs[k])
            seg = self.active[action]
            if p >= self.threshold and not final:
                if seg is None:
                    seg = self.active[action] = {"start": t, "peak": p, "peak_t": t, "objs": []}
                if p > seg["peak"]:
                    seg["peak"], seg["peak_t"] = p, t
                if focus_obj and obj_near_hand:
                    seg["objs"].append(focus_obj)
            elif seg is not None:
                self.active[action] = None
                ev = self._close(action, seg, t)
                if ev:
                    out.append(ev)
        return out

    def _close(self, action, seg, t_end) -> PendingEvent | None:
        obj = max(set(seg["objs"]), key=seg["objs"].count) if seg["objs"] else None
        note = None
        if obj is None and action == "putdown" and self.held:
            obj, note = self.held[-1], "object not visible at put-down"
        if obj is None:
            return None
        key = (obj, action)
        if seg["peak_t"] - self.last_fired.get(key, -1e9) < C.EVENT_COOLDOWN_S:
            return None
        self.last_fired[key] = seg["peak_t"]
        if action == "pickup":
            if obj not in self.held:
                self.held.append(obj)
            location_t = seg["start"]
        else:
            if obj in self.held:
                self.held.remove(obj)
            location_t = t_end
        return PendingEvent(obj, action, seg["peak_t"], location_t, round(seg["peak"], 3), note)
