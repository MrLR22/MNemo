# model/extract_features.py
# One-time offline step: video → per-frame features + labels (.npz), using the SAME perception and
# feature code as the live system, so training and inference inputs are identical.
#
# Every dataset is converted to one common form:  video → [(start_s, end_s, action, object)]
# with action in {pickup, putdown}. Output files are named  <group>__<video>.npz  where <group> is
# the person/source video, so train/val/test can be split by person (no leakage).
#
#   python -m model.extract_features epic     --videos D --annotations EPIC_100_train.csv --verb-classes EPIC_100_verb_classes.csv
#   python -m model.extract_features ego4d    --videos D --annotations fho_lta_train.json
#   python -m model.extract_features charades --videos D --annotations CharadesEgo_v1_train.csv --classes Charades_v1_classes.txt
#   python -m model.extract_features segments --videos D --annotations my_segments.csv   (video,start_s,end_s,action,object,group)
#   python -m model.extract_features own      (your labelled footage: data/own/labels.csv)
#
# Each video's output is written as soon as it is done, and existing outputs are skipped, so an
# interrupted Colab run can simply be restarted.

import argparse
import csv
import glob
import json
import os
import re
import time
from collections import Counter, defaultdict

import numpy as np

import config as C
from model.data import label_point_events, label_segments

VIDEO_EXT = ("mp4", "MP4", "webm", "mkv", "avi", "mov")


# ── shared extraction ──────────────────────────────────────────────────────────

def run_video(path: str, perception, max_seconds: float | None = None):
    from modules.features import FeatureExtractor
    from modules.video_source import get_frames
    perception.reset()
    fx = FeatureExtractor()
    feats, times, focus = [], [], []
    for frame in get_frames(path):
        if max_seconds is not None and frame.t > max_seconds:
            break
        feats.append(fx(perception(frame)))
        times.append(frame.t)
        focus.append(fx.focus_obj or "")
    return (np.stack(feats) if feats else np.zeros((0, 105), np.float32),
            np.array(times, np.float32), np.array(focus))


def find_videos(root: str) -> dict[str, str]:
    """video id (file name without extension) → path, searching sub-folders."""
    out = {}
    for ext in VIDEO_EXT:
        for p in glob.glob(os.path.join(root, "**", f"*.{ext}"), recursive=True):
            out[os.path.splitext(os.path.basename(p))[0]] = p
    return out


def safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "-", str(s))


def extract_segments(items: dict, dataset: str, args):
    """
    items: {video_id: {"path": str, "group": str, "segments": [(start, end, action, object)]}}
    Writes <out>/<dataset>/<group>__<video_id>.npz
    """
    from modules.perception import Perception
    out_dir = os.path.join(args.out, dataset)
    os.makedirs(out_dir, exist_ok=True)
    todo = sorted(items)[:args.max_videos] if args.max_videos else sorted(items)
    n = Counter(s[2] for v in todo for s in items[v]["segments"])
    print(f"[{dataset}] {len(todo)} videos to process: {n['pickup']} pick-up and {n['putdown']} put-down segments")
    if not todo:
        raise FileNotFoundError(f"[{dataset}] no annotated video files found under {args.videos}")

    # Public datasets: keep every hand (camera position differs from yours) and any portable object
    # (their objects aren't keys/wallets; HOI-TCN's features don't depend on object identity).
    perception = Perception(any_object=True, own_hands_only=False)
    for k, vid in enumerate(todo, 1):
        it = items[vid]
        out = os.path.join(out_dir, f"{safe(it['group'])}__{safe(vid)}.npz")
        if os.path.exists(out) and not args.overwrite:
            continue
        t0 = time.time()
        max_s = args.max_minutes * 60 if args.max_minutes else None
        feats, times, focus = run_video(it["path"], perception, max_s)
        if len(times) == 0:
            print(f"  [{k}/{len(todo)}] {vid}: could not read video, skipped")
            continue
        segs = [s for s in it["segments"] if s[1] <= times[-1]]
        lab, gt = label_segments(times, segs)
        np.savez_compressed(out, feats=feats, times=times, event=lab,
                            hold=np.zeros(len(times), np.float32), hold_mask=np.zeros(len(times), np.float32),
                            focus=focus, **gt)
        dt = time.time() - t0
        print(f"  [{k}/{len(todo)}] {vid}: {len(times)} frames, {len(segs)} segments, {dt:.0f}s "
              f"({len(times) / max(dt, 1e-6):.1f} frames/s)")


# ── dataset converters ─────────────────────────────────────────────────────────

def _ts(s: str) -> float:
    h, m, sec = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(sec)


def load_epic_segments(args) -> dict[str, list]:
    """EPIC-KITCHENS-100: verb classes 'take' → pickup, 'put' → putdown (by class, so the narrated
    variants 'pick-up', 'grab', 'place', ... are included)."""
    with open(args.verb_classes) as f:
        key_of = {int(r["id"]): r["key"] for r in csv.DictReader(f)}
    wanted = {"take": "pickup", "put": "putdown"}
    found = {k for k in key_of.values() if k in wanted}
    if found != set(wanted):
        raise ValueError(f"Verb classes 'take'/'put' not found in {args.verb_classes}; found {found}")
    segs = defaultdict(list)
    with open(args.annotations) as f:
        for r in csv.DictReader(f):
            if args.participants and r["participant_id"] not in args.participants:
                continue
            key = key_of.get(int(r["verb_class"]))
            if key not in wanted:
                continue
            segs[r["video_id"]].append((_ts(r["start_timestamp"]), _ts(r["stop_timestamp"]),
                                        wanted[key], r["noun"]))
    return segs


def epic_items(args) -> dict:
    segs = load_epic_segments(args)
    videos = find_videos(args.videos)
    return {v: {"path": videos[v], "group": v.split("_")[0], "segments": s}
            for v, s in segs.items() if v in videos}


def ego4d_verb_action(verb: str) -> str | None:
    """Ego4D FHO verb taxonomy, e.g. 'take_(pick,_grab,_get)', 'put_(place,_leave,_drop)'.
    Excludes wearing/removing clothes ('put_on', 'take_off')."""
    v = verb.lower().strip()
    if v.startswith(("put_on", "put-on", "take_off", "take-off")):
        return None
    if v.startswith(("take", "pick", "grab")):
        return "pickup"
    if v.startswith(("put", "place", "drop")):
        return "putdown"
    return None


def _first(d: dict, *keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def load_ego4d_segments(args) -> tuple[dict, dict]:
    """
    Ego4D Forecasting Hands & Objects, long-term-anticipation annotation files (fho_lta_*.json):
    one record per action with clip_uid, video_uid, action_clip_start_sec, action_clip_end_sec,
    verb, noun. Field names are checked at load time; if your annotation version differs,
    the error message prints the keys that were found.
    Returns (segments per clip_uid, group (= source video_uid) per clip_uid).
    """
    segs, groups, verbs = defaultdict(list), {}, Counter()
    for path in args.annotations if isinstance(args.annotations, list) else [args.annotations]:
        with open(path) as f:
            data = json.load(f)
        records = data.get("clips", data) if isinstance(data, dict) else data
        if not records:
            raise ValueError(f"{path}: no records found")
        r0 = records[0]
        need = {"clip": ("clip_uid",), "start": ("action_clip_start_sec", "clip_start_sec", "start_sec"),
                "end": ("action_clip_end_sec", "clip_end_sec", "end_sec"), "verb": ("verb",)}
        missing = [k for k, opts in need.items() if _first(r0, *opts) is None]
        if missing:
            raise ValueError(f"{path}: could not find fields {missing}. First record keys: {sorted(r0)}")
        for r in records:
            verbs[r["verb"]] += 1
            action = ego4d_verb_action(str(r["verb"]))
            if action is None:
                continue
            clip = r["clip_uid"]
            segs[clip].append((float(_first(r, *need["start"])), float(_first(r, *need["end"])),
                               action, str(r.get("noun", ""))))
            groups[clip] = str(r.get("video_uid", clip))
    chosen = {v: c for v, c in verbs.items() if ego4d_verb_action(str(v))}
    print(f"[ego4d] verbs mapped to pick-up/put-down (check this list!): {dict(Counter(chosen).most_common())}")
    return segs, groups


def ego4d_items(args) -> dict:
    segs, groups = load_ego4d_segments(args)
    videos = find_videos(args.videos)
    return {c: {"path": videos[c], "group": groups[c], "segments": s} for c, s in segs.items() if c in videos}


CHARADES_PICK = re.compile(r"^(Taking|Picking up) .+ from somewhere$", re.I)
CHARADES_PUT = re.compile(r"^Putting (?!on ).+ somewhere$", re.I)


def load_charades_classes(path: str) -> dict[str, str]:
    """Charades_v1_classes.txt lines: 'c092 Putting a phone/camera somewhere' → {code: action}."""
    out = {}
    with open(path) as f:
        for line in f:
            code, _, text = line.strip().partition(" ")
            if CHARADES_PICK.match(text):
                out[code] = ("pickup", text)
            elif CHARADES_PUT.match(text):
                out[code] = ("putdown", text)
    return out


def load_charades_segments(args) -> tuple[dict, dict]:
    """Charades-Ego: CSV with id, subject, actions = 'c092 11.90 21.20;c147 0.00 12.60'.
    Only first-person videos (id ending in 'EGO' or egocentric column = Yes)."""
    classes = load_charades_classes(args.classes)
    print(f"[charades] classes mapped (check this list!): {sorted(t for _, t in classes.values())}")
    if not classes:
        raise ValueError("No 'Taking ... from somewhere' / 'Putting ... somewhere' classes found")
    segs, groups = defaultdict(list), {}
    with open(args.annotations) as f:
        for r in csv.DictReader(f):
            vid = r["id"]
            ego = str(r.get("egocentric", "")).lower() in ("yes", "1", "true") or vid.upper().endswith("EGO")
            if not ego:
                continue
            for part in (r.get("actions") or "").split(";"):
                bits = part.split()
                if len(bits) == 3 and bits[0] in classes:
                    action, text = classes[bits[0]]
                    segs[vid].append((float(bits[1]), float(bits[2]), action, text))
            groups[vid] = r.get("subject") or vid
    return segs, groups


def charades_items(args) -> dict:
    segs, groups = load_charades_segments(args)
    videos = find_videos(args.videos)
    return {v: {"path": videos[v], "group": groups[v], "segments": s} for v, s in segs.items() if v in videos}


def generic_items(args) -> dict:
    """CSV: video,start_s,end_s,action,object[,group]  (action = pickup|putdown)."""
    videos = find_videos(args.videos)
    items = {}
    with open(args.annotations) as f:
        for r in csv.DictReader(f):
            vid = os.path.splitext(os.path.basename(r["video"]))[0]
            if vid not in videos:
                continue
            it = items.setdefault(vid, {"path": videos[vid], "group": r.get("group") or vid, "segments": []})
            it["segments"].append((float(r["start_s"]), float(r["end_s"]), r["action"].strip(), r["object"]))
    return items


# ── your own footage (point labels from tools/label_events.py) ────────────────

def extract_own(args):
    from modules.perception import Perception
    labels = defaultdict(list)
    with open(args.labels) as f:
        for row in csv.DictReader(f):
            labels[row["video"]].append((float(row["time_s"]), row["action"].strip(), row["object"].strip()))
    out_dir = os.path.join(args.out, "own")
    os.makedirs(out_dir, exist_ok=True)
    perception = Perception()
    for video, events in sorted(labels.items()):
        path = video if os.path.isabs(video) else os.path.join(args.videos, video)
        name = safe(os.path.splitext(os.path.basename(video))[0])
        out = os.path.join(out_dir, f"{name}__{name}.npz")
        if os.path.exists(out) and not args.overwrite:
            print(f"skip {name} (exists)")
            continue
        t0 = time.time()
        feats, times, focus = run_video(path, perception)
        lab, hold, gt = label_point_events(times, events)
        np.savez_compressed(out, feats=feats, times=times, event=lab, hold=hold,
                            hold_mask=np.ones_like(hold), focus=focus, **gt)
        print(f"{name}: {len(times)} frames, {len(events)} events, "
              f"{(lab > 0).sum()} event frames, {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = {name: sub.add_parser(name) for name in ("epic", "ego4d", "charades", "segments", "own")}
    p["epic"].add_argument("--videos", default="data/epic_kitchens/videos")
    p["epic"].add_argument("--annotations", default="data/epic_kitchens/EPIC_100_train.csv")
    p["epic"].add_argument("--verb-classes", default="data/epic_kitchens/EPIC_100_verb_classes.csv")
    p["epic"].add_argument("--participants", nargs="*", default=None)
    p["ego4d"].add_argument("--videos", required=True, help="folder with the downloaded FHO clips (<clip_uid>.mp4)")
    p["ego4d"].add_argument("--annotations", nargs="+", required=True, help="fho_lta_train.json [fho_lta_val.json]")
    p["charades"].add_argument("--videos", required=True)
    p["charades"].add_argument("--annotations", required=True, help="CharadesEgo_v1_train.csv")
    p["charades"].add_argument("--classes", required=True, help="Charades_v1_classes.txt")
    p["segments"].add_argument("--videos", required=True)
    p["segments"].add_argument("--annotations", required=True)
    p["own"].add_argument("--videos", default=C.OWN_VIDEO_DIR)
    p["own"].add_argument("--labels", default=C.OWN_LABELS_CSV)
    for name, sp in p.items():
        sp.add_argument("--out", default=C.FEATURE_DIR)
        sp.add_argument("--overwrite", action="store_true")
        if name != "own":
            sp.add_argument("--max-videos", type=int, default=None)
            sp.add_argument("--max-minutes", type=float, default=None, help="only the first N minutes of each video")
            sp.add_argument("--name", default=None, help="output sub-folder (default: dataset name)")
    args = ap.parse_args()

    if args.cmd == "own":
        return extract_own(args)
    items = {"epic": epic_items, "ego4d": ego4d_items, "charades": charades_items,
             "segments": generic_items}[args.cmd](args)
    extract_segments(items, args.name or args.cmd, args)


if __name__ == "__main__":
    main()
