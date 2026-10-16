# tools/fetch_epic_windows.py
# Fetch only the BUSIEST short pieces of EPIC-KITCHENS-100 videos instead of whole 6 GB files.
#
# The EPIC server supports HTTP range requests, so ffmpeg can jump to minute 15 of a video and read
# just that part (verified). Each piece is shrunk to 640 px / 10 fps on the fly, so a 60 s piece
# is only a few MB on disk. Pieces are spread across participants so the person-based
# train/val/test split still works.
#
#   python -m tools.fetch_epic_windows --data <folder with the 4 CSVs> --out <clips folder> --windows 5
#
# <data> must contain: EPIC_100_train.csv, EPIC_100_verb_classes.csv, EPIC_100_video_info.csv,
# epic_55_splits.csv (see the notebook cell). Output: <out>/<video>_w<start>.mp4 and
# <out>/segments.csv (video,start_s,end_s,action,object,group), which model.extract_features
# `segments` reads. Finished pieces are skipped, so the command can be re-run to resume or extend.

import argparse
import csv
import os
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from model.extract_features import load_epic_segments

EPIC55_URL = "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d"   # original videos (P01_01)
EPIC100_URL = "https://data.bris.ac.uk/datasets/2g1n6qdydwa9u22shpxqzp0t8m"  # extension videos (P01_101)


def video_url(video_id: str, split55: dict[str, str]) -> str:
    """Same layout as the official epic_downloader.py."""
    participant, num = video_id.split("_")
    if len(num) == 3:                                   # EPIC-100 extension
        return f"{EPIC100_URL}/{participant}/videos/{video_id}.MP4"
    return f"{EPIC55_URL}/videos/{split55[video_id]}/{participant}/{video_id}.MP4"


def best_windows(segs: list[tuple], duration: float, window_s: float, max_windows: int) -> list[tuple]:
    """
    Non-overlapping windows with the most take/put segments ending inside them.
    Returns [(count, start)] best first. Candidate starts: 2 s before each segment start.
    """
    ends = sorted(s[1] for s in segs)
    cands = []
    for s in segs:
        t0 = min(max(0.0, s[0] - 2.0), max(0.0, duration - window_s))
        n = sum(1 for e in ends if t0 <= e <= t0 + window_s)
        cands.append((n, t0))
    cands.sort(key=lambda c: (-c[0], c[1]))
    chosen = []
    for n, t0 in cands:
        if all(abs(t0 - c) >= window_s for _, c in chosen):
            chosen.append((n, t0))
            if len(chosen) == max_windows:
                break
    return chosen


def plan(segs_by_video: dict, durations: dict, n_windows: int, window_s: float) -> list[dict]:
    """Round-robin over participants (busiest window first) so every person contributes."""
    per_p = defaultdict(list)
    for vid, segs in segs_by_video.items():
        if vid not in durations:
            continue
        for n, t0 in best_windows(segs, durations[vid], window_s, max_windows=3):
            per_p[vid.split("_")[0]].append({"video": vid, "start": t0, "count": n})
    for p in per_p:
        per_p[p].sort(key=lambda w: -w["count"])
    order = sorted(per_p, key=lambda p: -per_p[p][0]["count"])
    out, k = [], 0
    while len(out) < n_windows and any(k < len(per_p[p]) for p in order):
        for p in order:
            if k < len(per_p[p]) and len(out) < n_windows:
                out.append(per_p[p][k])
        k += 1
    return out


def plan_random(segs_by_video: dict, durations: dict, n_windows: int, window_s: float, seed: int = 0) -> list[dict]:
    """Ordinary (not busiest) stretches of video: random start times, round-robin over participants.
    Used to measure a trigger fairly, because real days contain idle time too."""
    import random
    rng = random.Random(seed)
    per_p = defaultdict(list)
    for vid in sorted(segs_by_video):
        if vid in durations and durations[vid] >= window_s:
            per_p[vid.split("_")[0]].append(vid)
    out, used = [], defaultdict(list)
    people = sorted(per_p)
    tries = 0
    while len(out) < n_windows and people and tries < n_windows * 50:
        p = people[len(out) % len(people)] if tries < n_windows * 25 else rng.choice(people)
        vid = rng.choice(per_p[p])
        t0 = round(rng.uniform(0, durations[vid] - window_s), 1)
        tries += 1
        if any(abs(t0 - u) < window_s for u in used[vid]):
            continue                                        # no overlapping pieces
        used[vid].append(t0)
        n = sum(1 for s, e, *_ in segs_by_video[vid] if t0 <= e <= t0 + window_s)
        out.append({"video": vid, "start": t0, "count": n})
    return out


def window_segments(segs: list[tuple], t0: float, window_s: float) -> list[tuple]:
    """Segments whose END is inside the window, re-timed relative to the window start."""
    return [(max(0.0, s - t0), e - t0, a, o) for s, e, a, o in segs if t0 <= e <= t0 + window_s]


def fetch(src: str, t0: float, window_s: float, out_path: str, timeout_s: int = 1800) -> bool:
    tmp = out_path + ".part.mp4"
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{t0:.3f}", "-t", f"{window_s:.3f}", "-i", src,
           "-vf", "scale=640:-2,fps=10", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", tmp]
    for attempt in range(2):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
            if r.returncode == 0 and os.path.getsize(tmp) > 0:
                os.replace(tmp, out_path)       # only finished files get the final name (safe resume)
                return True
            print(f"    ffmpeg failed (attempt {attempt + 1}): {r.stderr.strip()[-300:]}")
        except subprocess.TimeoutExpired:
            print(f"    timed out after {timeout_s}s (attempt {attempt + 1})")
    if os.path.exists(tmp):
        os.remove(tmp)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with the 4 EPIC CSV files")
    ap.add_argument("--out", required=True, help="folder for the clips + segments.csv")
    ap.add_argument("--windows", type=int, default=100, help="how many pieces to fetch in total")
    ap.add_argument("--window-s", type=float, default=60.0)
    ap.add_argument("--workers", type=int, default=4, help="pieces fetched in parallel")
    ap.add_argument("--participants", nargs="*", default=None)
    ap.add_argument("--select", choices=["busiest", "random"], default="busiest",
                    help="busiest = most pick/put per minute (training); random = ordinary stretches (trigger test)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ns = argparse.Namespace(annotations=os.path.join(args.data, "EPIC_100_train.csv"),
                            verb_classes=os.path.join(args.data, "EPIC_100_verb_classes.csv"),
                            participants=args.participants)
    segs = load_epic_segments(ns)
    with open(os.path.join(args.data, "EPIC_100_video_info.csv")) as f:
        durations = {r["video_id"]: float(r["duration"]) for r in csv.DictReader(f)}
    with open(os.path.join(args.data, "epic_55_splits.csv")) as f:
        split55 = {r["video_id"]: r["split"] for r in csv.DictReader(f)}

    if args.select == "random":
        windows = plan_random(segs, durations, args.windows, args.window_s, args.seed)
    else:
        windows = plan(segs, durations, args.windows, args.window_s)
    people = sorted({w["video"].split("_")[0] for w in windows})
    total = sum(w["count"] for w in windows)
    print(f"plan: {len(windows)} pieces of {args.window_s:.0f}s from {len(people)} participants, "
          f"~{total} pick-up/put-down events")

    os.makedirs(args.out, exist_ok=True)
    rows = []
    for w in windows:
        w["name"] = f"{w['video']}_w{int(w['start'])}"
        for s, e, a, o in window_segments(segs[w["video"]], w["start"], args.window_s):
            rows.append({"video": w["name"], "start_s": f"{s:.2f}", "end_s": f"{e:.2f}", "action": a,
                         "object": o, "group": w["video"].split("_")[0]})
    with open(os.path.join(args.out, "segments.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["video", "start_s", "end_s", "action", "object", "group"])
        wr.writeheader()
        wr.writerows(rows)

    todo = [w for w in windows if not os.path.exists(os.path.join(args.out, w["name"] + ".mp4"))]
    print(f"{len(windows) - len(todo)} already fetched, {len(todo)} to fetch ({args.workers} at a time)")
    t_start, done, failed = time.time(), 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch, video_url(w["video"], split55), w["start"], args.window_s,
                          os.path.join(args.out, w["name"] + ".mp4")): w for w in todo}
        for fut in as_completed(futs):
            w = futs[fut]
            ok = fut.result()
            done += ok
            failed += not ok
            el = time.time() - t_start
            left = len(todo) - done - failed
            eta = el / max(done + failed, 1) * left
            print(f"  [{done + failed}/{len(todo)}] {w['name']} ({w['count']} events) "
                  f"{'ok' if ok else 'FAILED'} | elapsed {el / 60:.1f} min | ETA {eta / 60:.0f} min", flush=True)
    print(f"done: {done} fetched, {failed} failed. Clips + segments.csv in {args.out}")
    if failed:
        print("Re-run the same command to retry the failed pieces.")


if __name__ == "__main__":
    main()
