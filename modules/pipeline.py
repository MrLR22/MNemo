# modules/pipeline.py
# Full runtime path:  M1 frames → M2 perception → M3 interaction (rules or HOI-TCN)
#                     → M5 landmark → M6 store.
# Privacy (M9): frames live only in the RAM FrameBuffer; only text events reach the store.

import os
import resource
import time
from datetime import datetime, timedelta

import cv2

import config as C
from modules.features import FeatureExtractor, nearest_hand
from modules.interaction import RuleEngine, ModelEngine, PendingEvent
from modules.perception import Perception
from modules.storage import EventStore
from modules.types import Event, Frame, FrameObs
from modules.video_source import FrameBuffer, get_frames


class MnemoPipeline:
    def __init__(self, mode: str = "rules", store: EventStore | None = None,
                 use_landmarks: bool = True, checkpoint: str = C.MODEL_CHECKPOINT,
                 perception: Perception | None = None):
        if mode not in ("rules", "model"):
            raise ValueError("mode must be 'rules' or 'model'")
        self.mode = mode
        self.store = store
        self.perception = perception or Perception()
        self.buffer = FrameBuffer()
        self.features = FeatureExtractor()
        self.last_probs = None          # model mode: latest P(none, pickup, putdown), for display

        if mode == "rules":
            self.engine = RuleEngine()
        else:
            from model.architecture import StreamingModel, load_model
            if not os.path.exists(checkpoint):
                raise FileNotFoundError(f"No trained model at {checkpoint}. Train it or use --mode rules.")
            model = load_model(checkpoint)
            self.stream = StreamingModel(model)
            self.engine = ModelEngine(threshold=model.threshold or C.EVENT_MODEL_THRESHOLD)

        # Place resolver: your enrolled landmarks if they exist, else zero-shot generic places.
        self.embedder = self.index = None
        if use_landmarks:
            from modules.landmarks import ClipEmbedder, LandmarkIndex, ZeroShotPlaces
            self.embedder = ClipEmbedder()
            if os.path.exists(C.LANDMARK_INDEX):
                self.index = LandmarkIndex.load(C.LANDMARK_INDEX)
            else:
                self.index = ZeroShotPlaces(self.embedder)
                print("[Pipeline] No enrolled landmarks: using zero-shot generic places "
                      f"({', '.join(self.index.names)}).")

    def reset(self):
        self.buffer.clear()
        self.perception.reset()
        self.features.reset()
        self.engine.reset()
        if self.mode == "model":
            self.stream.reset()

    # ── per frame ──
    def process(self, frame: Frame, final: bool = False) -> tuple[FrameObs, list[Event]]:
        self.buffer.push(frame)
        obs = self.perception(frame)
        feat = self.features(obs)
        if self.mode == "rules":
            pending = self.engine.update(obs)
        else:
            probs = self.stream.step(feat).numpy()
            self.last_probs = probs
            focus = self.features.focus_obj
            near = any(d.obj == focus and nearest_hand(d, obs.hands)[1] < 2 * C.CONTACT_DIST
                       for d in obs.detections)
            pending = self.engine.update(frame.t, probs, focus, near, final=final)
        events = [self._finalise(p, frame) for p in pending]
        if self.store:
            for ev in events:
                self.store.add(ev, source=getattr(self, "source_name", ""))
        return obs, events

    def _finalise(self, p: PendingEvent, now: Frame) -> Event:
        landmark, note = None, p.note
        if self.index is not None:
            loc_frame = self.buffer.at(p.location_t) or now
            emb = self.embedder.embed([loc_frame.image])[0]
            landmark, _ = self.index.resolve(emb)
        ts = now.ts - timedelta(seconds=now.t - p.t)      # event time, not decision time
        return Event(ts=ts, object=p.object, action=p.action, landmark=landmark,
                     confidence=p.confidence, note=note, t=p.t)

    # ── whole video / camera ──
    def run(self, source, start_time: datetime | None = None, show: bool = False,
            verbose: bool = True) -> tuple[list[Event], dict]:
        self.reset()
        self.source_name = str(source)
        events, n, t0 = [], 0, time.time()
        last = None
        try:
            for frame in get_frames(source, start_time):
                obs, evs = self.process(frame)
                events += evs
                n += 1
                last = frame
                if verbose:
                    for e in evs:
                        print(f"  [{e.t:7.2f}s] {e.action:7s} {e.object:12s} @ {e.landmark or 'unknown':18s} "
                              f"conf={e.confidence:.2f}" + (f"  ({e.note})" if e.note else ""))
                if show and not self._show(frame, obs, evs):
                    break
            if self.mode == "model" and last is not None:      # close any open segment
                _, evs = self.process(last, final=True)
                events += evs
        finally:
            if show:
                cv2.destroyAllWindows()
            self.buffer.clear()
        wall = time.time() - t0
        stats = {"frames": n, "seconds": round(wall, 1), "fps": round(n / wall, 2) if wall else 0.0,
                 "peak_ram_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
                 "events": len(events)}
        return events, stats

    def _show(self, frame: Frame, obs: FrameObs, evs: list[Event]) -> bool:
        """Display only (never saved). Returns False when the user presses q."""
        im = frame.image.copy()
        h, w = im.shape[:2]
        for hand in obs.hands:
            for x, y in hand.landmarks:
                cv2.circle(im, (int(x * w), int(y * h)), 2, (255, 0, 0), -1)
        for d in obs.detections:
            x1, y1, x2, y2 = d.box
            cv2.rectangle(im, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (0, 255, 0), 2)
            cv2.putText(im, f"{d.obj} {d.conf:.2f}", (int(x1 * w), max(15, int(y1 * h) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        held = ", ".join(self.engine.holding()) or "-"
        cv2.putText(im, f"t={frame.t:6.1f}s  holding: {held}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        for i, e in enumerate(evs):
            cv2.putText(im, f"{e.action.upper()} {e.object} @ {e.landmark or 'unknown'}", (10, 55 + 28 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Mnemo", im)
        return (cv2.waitKey(1) & 0xFF) != ord("q")
