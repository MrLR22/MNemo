# modules/hoi_gate.py
# The interaction trigger (novelty N1): decides WHICH 4-s windows are worth sending to the VLM.
#
# Cheap perception (MediaPipe hands + COCO YOLOv8n + optical flow) runs on every frame at 10 fps,
# HOI-TCN turns it into a per-frame score, and each VLM window gets the max over the window of
#   1 - P(none)                (pick-up / put-down; the first model), or
#   P(any hand action)         (the act head, trained on ALL EPIC verbs: model/train_interaction.py),
# whichever the checkpoint says was best on the validation people (model.gate_score).
# A window goes to the VLM when
#   - its HOI score >= GATE_THRESHOLD                                   (pick-up / put-down seen), or
#   - the previous window was triggered (GATE_HANGOVER windows)         (actions come in bursts), or
#   - hands are visible and moving fast enough (GATE_MOTION)            (stirring, cutting, sliding: not
#     pick/put, which is all HOI-TCN was trained on), or
#   - nothing was described for GATE_HEARTBEAT_S seconds               (keeps place and object states fresh).
#
# HOI-TCN is causal, so a window's score is known as soon as the window ends; computing all scores
# first (as here, for offline videos) gives the same decisions as doing it live.

import time

import numpy as np

import config as C
from modules.features import HAND_DIM


def window_hoi_scores(times: np.ndarray, probs: np.ndarray, window_s: float) -> np.ndarray:
    """Per-window max of the frame score. probs: (N, 3) event probabilities → 1 - P(none), or (N,) an
    already computed per-frame score (model.train.predict_activity). Window k covers [k*w, (k+1)*w)."""
    if len(times) == 0:
        return np.zeros(0)
    n_win = int(np.floor(float(times[-1]) / window_s)) + 1
    idx = np.minimum((times / window_s).astype(int), n_win - 1)
    act = 1.0 - probs[:, 0] if probs.ndim == 2 else probs
    out = np.zeros(n_win)
    np.maximum.at(out, idx, act)
    return out


def window_motion(times: np.ndarray, feats: np.ndarray, window_s: float) -> np.ndarray:
    """Per window: mean over frames of the fastest visible hand's speed (0 when no hand is visible).
    Speeds are the features' hand velocities: image fraction per 0.1 s, camera motion removed, x VEL_SCALE."""
    if len(times) == 0:
        return np.zeros(0)
    n_win = int(np.floor(float(times[-1]) / window_s)) + 1
    idx = np.minimum((times / window_s).astype(int), n_win - 1)
    v1 = np.linalg.norm(feats[:, 43:45], axis=1) * (feats[:, 0] > 0)
    v2 = np.linalg.norm(feats[:, HAND_DIM + 43:HAND_DIM + 45], axis=1) * (feats[:, HAND_DIM] > 0)
    speed = np.maximum(v1, v2)
    tot, cnt = np.zeros(n_win), np.zeros(n_win)
    np.add.at(tot, idx, speed)
    np.add.at(cnt, idx, 1)
    return tot / np.maximum(cnt, 1)


def gate_decisions(scores: np.ndarray, threshold: float = C.GATE_THRESHOLD,
                   heartbeat_windows: int | None = None, motion: np.ndarray | None = None,
                   motion_threshold: float | None = None, hangover: int | None = None) -> np.ndarray:
    """Boolean per window: call the VLM? (rules in the header). motion=None or motion_threshold=None
    switches the motion rule off; hangover=0 switches the keep-going rule off."""
    if heartbeat_windows is None:
        heartbeat_windows = max(1, round(C.GATE_HEARTBEAT_S / C.VLM_WINDOW_S))
    hangover = C.GATE_HANGOVER if hangover is None else hangover
    call = np.zeros(len(scores), bool)
    since = heartbeat_windows                     # describe the very first window: sets the scene
    last_trig = -10 ** 9
    for k, s in enumerate(scores):
        moving = motion is not None and motion_threshold is not None and k < len(motion) \
            and motion[k] >= motion_threshold
        triggered = s >= threshold or moving
        if triggered or k - last_trig <= hangover or since >= heartbeat_windows:
            call[k] = True
            since = 1
        else:
            since += 1
        if triggered:
            last_trig = k
    return call


class HOIGate:
    """Loads HOI-TCN + perception once; .signals(video) → per-window HOI score and hand motion."""

    def __init__(self, checkpoint: str = C.MODEL_CHECKPOINT, window_s: float = C.VLM_WINDOW_S):
        from model.architecture import load_model
        from modules.perception import Perception
        self.model = load_model(checkpoint)
        # same settings as the public-dataset training data (model/extract_features.py)
        self.perception = Perception(any_object=True, own_hands_only=False)
        self.window_s = window_s

    def signals(self, video: str, max_seconds: float | None = None) -> tuple[dict, dict]:
        """{'hoi': per-window HOI score, 'motion': per-window hand speed}, timing stats."""
        from model.extract_features import run_video
        from model.train import predict_activity
        t0 = time.time()
        feats, times, _ = run_video(video, self.perception, max_seconds)
        probs = predict_activity(self.model, feats) if len(times) else np.zeros(0)
        sec = time.time() - t0
        video_s = float(times[-1]) + 1.0 / C.PROCESS_FPS if len(times) else 0.0
        sig = {"hoi": window_hoi_scores(times, probs, self.window_s),
               "motion": window_motion(times, feats, self.window_s)}
        return sig, {"gate_seconds": round(sec, 1), "frames": len(times),
                     "gate_real_time_factor": round(sec / video_s, 2) if video_s else None}

    def scores(self, video: str, max_seconds: float | None = None) -> tuple[np.ndarray, dict]:
        sig, stats = self.signals(video, max_seconds)
        return sig["hoi"], stats
