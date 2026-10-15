# tools/demo_video.py
# Make a DEMO video: runs the full Mnemo pipeline on a clip and saves an annotated copy that plays
# smoothly (processing itself is slower than real time on a laptop CPU).
#
#   python -m tools.demo_video CLIP.mp4 --objects cup plate bowl lid pan knife --labels segments.csv
#
# Shows: hand skeletons, tracked objects, live P(pick-up) / P(put-down) bars (model mode), what is
# being held, and each detected event with its place. With --labels (an EPIC segments.csv from
# tools/fetch_epic_windows.py) the true pick-ups/put-downs are shown too, so viewers can compare.
# Events go into a separate demo database, so afterwards you can ask:
#   python mnemo.py --db outputs/demo.db ask "where is the plate?"
#
# This is an offline demo tool and the only place a video is written; the runtime itself
# (mnemo.py run) never writes images or video.

import argparse
import csv
import os
import time

import cv2
import numpy as np

import config as C

HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
              (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
OUT_W = 960


def load_labels(path: str, clip: str, duration: float = 60.0) -> list[tuple]:
    """True events for the clip: from a segments.csv (fetch_epic_windows) or directly from
    EPIC_100_train.csv, using the clip name <video>_w<start> to find its time range."""
    if not path:
        return []
    name = os.path.splitext(os.path.basename(clip))[0]
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if rows and "narration_id" in rows[0]:              # the EPIC annotation file itself
        video, _, start = name.rpartition("_w")
        t0 = float(start)
        out = []
        for r in rows:
            if r["video_id"] == video and r["verb_class"] in ("0", "1"):     # 0 = take, 1 = put
                h, m, sec = r["stop_timestamp"].split(":")
                end = int(h) * 3600 + int(m) * 60 + float(sec) - t0
                if 0 <= end <= duration:
                    out.append((end, "pickup" if r["verb_class"] == "0" else "putdown", r["noun"]))
        return sorted(out)
    return [(float(r["end_s"]), r["action"], r["object"]) for r in rows if r["video"] == name]


def nice(s: str | None) -> str:
    return (s or "unknown place").replace("_", " ")


def bar(im, x, y, w, h, value, color, label, thr=None):
    cv2.rectangle(im, (x, y), (x + w, y + h), (60, 60, 60), -1)
    cv2.rectangle(im, (x, y), (x + int(w * float(value)), y + h), color, -1)
    if thr is not None:
        tx = x + int(w * thr)
        cv2.line(im, (tx, y - 3), (tx, y + h + 3), (255, 255, 255), 2)
    cv2.putText(im, f"{label} {value:.2f}", (x + 6, y + h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                cv2.LINE_AA)


def render(frame, obs, pipe, recent, truth_now, flash) -> np.ndarray:
    im = frame.image.copy()
    h, w = im.shape[:2]
    for hand in obs.hands:
        pts = [(int(x * w), int(y * h)) for x, y in hand.landmarks]
        for a, b in HAND_EDGES:
            cv2.line(im, pts[a], pts[b], (255, 180, 0), 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(im, p, 3, (255, 255, 255), -1)
    for d in obs.detections:
        x1, y1, x2, y2 = (int(d.box[0] * w), int(d.box[1] * h), int(d.box[2] * w), int(d.box[3] * h))
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(im, f"{d.obj} {d.conf:.2f}", (x1, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 220, 0), 1, cv2.LINE_AA)

    im = cv2.resize(im, (OUT_W, int(h * OUT_W / w)))
    H = im.shape[0]
    top = im.copy()
    cv2.rectangle(top, (0, 0), (OUT_W, 92), (0, 0, 0), -1)
    cv2.addWeighted(top, 0.6, im, 0.4, 0, im)
    held = ", ".join(pipe.engine.holding()) or "-"
    cv2.putText(im, f"MNEMO  t={frame.t:5.1f}s   mode: {pipe.mode}   holding: {held}", (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    if pipe.last_probs is not None:
        thr = pipe.engine.threshold
        bar(im, 12, 40, 300, 20, pipe.last_probs[1], (0, 140, 255), "P(pick-up)", thr)
        bar(im, 330, 40, 300, 20, pipe.last_probs[2], (255, 80, 200), "P(put-down)", thr)
    if truth_now:
        cv2.putText(im, "TRUE: " + truth_now, (12, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1,
                    cv2.LINE_AA)

    if flash:
        e = flash
        txt = f"{'PICK-UP' if e.action == 'pickup' else 'PUT-DOWN'}: {nice(e.object)}  @ {nice(e.landmark)}"
        cv2.rectangle(im, (0, H // 2 - 30), (OUT_W, H // 2 + 20), (0, 0, 180), -1)
        cv2.putText(im, txt, (20, H // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    log = im.copy()
    cv2.rectangle(log, (0, H - 26 * 4 - 10), (OUT_W, H), (0, 0, 0), -1)
    cv2.addWeighted(log, 0.6, im, 0.4, 0, im)
    for i, e in enumerate(recent[-4:]):
        cv2.putText(im, f"{e.ts:%H:%M:%S}  {e.action:7s} {nice(e.object):12s} @ {nice(e.landmark)}  (conf {e.confidence:.2f})",
                    (12, H - 26 * (4 - i) + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default=None, help="annotated output video (default: outputs/demo_<name>.mp4)")
    ap.add_argument("--objects", nargs="+", default=None, help="objects to track, e.g. cup plate bowl lid")
    ap.add_argument("--labels", default=None, help="true events: segments.csv or EPIC_100_train.csv (optional)")
    ap.add_argument("--checkpoint", default=C.MODEL_CHECKPOINT)
    ap.add_argument("--mode", choices=["model", "rules"], default=None,
                    help="default: model if the checkpoint exists, else rules")
    ap.add_argument("--db", default=os.path.join(C.ROOT, "outputs", "demo.db"))
    ap.add_argument("--start", default=None, help='real start time, e.g. "2026-10-17 09:00"')
    args = ap.parse_args()

    if args.objects:
        C.set_objects(args.objects)
    from datetime import datetime
    from modules.pipeline import MnemoPipeline
    from modules.storage import EventStore
    from modules.video_source import get_frames

    mode = args.mode or ("model" if os.path.exists(args.checkpoint) else "rules")
    store = EventStore(args.db)
    store.clear()                                       # a demo DB holds one demo run
    pipe = MnemoPipeline(mode=mode, store=store, checkpoint=args.checkpoint)
    pipe.reset()
    pipe.source_name = os.path.basename(args.video)
    cap = cv2.VideoCapture(args.video)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 10.0)
    cap.release()
    labels = load_labels(args.labels, args.video, duration)
    print(f"[demo] mode={mode}  objects={C.TARGET_OBJECTS}  true events in clip: {len(labels)}")

    out = args.out or os.path.join(C.ROOT, "outputs", f"demo_{os.path.splitext(os.path.basename(args.video))[0]}.mp4")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    writer = None
    events, flash, flash_until, n, t0 = [], None, -1.0, 0, time.time()
    start = datetime.fromisoformat(args.start) if args.start else None
    last = None
    for frame in get_frames(args.video, start):
        obs, evs = pipe.process(frame)
        last = frame
        for e in evs:
            events.append(e)
            flash, flash_until = e, frame.t + 1.5
            print(f"  [{e.t:6.1f}s] {e.action:7s} {e.object:10s} @ {e.landmark or 'unknown':16s} conf={e.confidence:.2f}")
        truth = [f"{a} {o}" for t_end, a, o in labels if abs(frame.t - t_end) <= 0.7]
        im = render(frame, obs, pipe, events, ", ".join(truth), flash if frame.t <= flash_until else None)
        if writer is None:
            writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), C.PROCESS_FPS, (im.shape[1], im.shape[0]))
        writer.write(im)
        n += 1
        if n % 50 == 0:
            print(f"  ... {n} frames, {n / (time.time() - t0):.1f} frames/s", flush=True)
    if mode == "model" and last is not None:
        _, evs = pipe.process(last, final=True)
        events += evs
    if writer is not None:
        writer.release()

    print(f"\n[demo] {n} frames in {time.time() - t0:.0f}s → {out}")
    print(f"[demo] detected {len(events)} events (true events in clip: {len(labels) or 'not given'})")
    if events:
        o = events[-1].object.replace("_", " ")
        print(f'[demo] now try:  python mnemo.py --db "{args.db}" ask "where is the {o}?"')


if __name__ == "__main__":
    main()
