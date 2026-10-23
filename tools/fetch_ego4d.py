# tools/fetch_ego4d.py
# Download only the Ego4D NLQ clips we test on (one clip per source video, ~72 MB per 8-min clip)
# with the official `ego4d` CLI. Needs the AWS profile from the licence email in ~/.aws (the Kaggle
# notebook writes it from Kaggle Secrets; keys never go in code or output). Keep --out OUTSIDE the
# results folder: the Ego4D licence does not allow sharing the videos.
#
#   python -m tools.fetch_ego4d --nlq nlq_val.json --out /tmp/ego4d --clips 20 --extra 20 --seed 0 --skip 3
# --skip N leaves out the first N clips of the same selection: the development clips used to fix
# query/locate.py, so the reported test never includes clips we tuned on.
# → <out>/selected.json: [{"clip_uid", "video_uid", "seconds", "questions", "path", "full"}]
#   full=True: through the VLM; full=False: extra clips for the cheap trigger-only test.

import argparse
import glob
import json
import os
import subprocess

from evaluation.ego4d_nlq import load_nlq, select_clips


def find(out: str, uid: str) -> str | None:
    hits = glob.glob(os.path.join(out, "**", f"{uid}*.mp4"), recursive=True)
    return hits[0] if hits else None


def download(uids: list[str], out: str, batch: int = 40):
    todo = [u for u in uids if not find(out, u)]
    for i in range(0, len(todo), batch):
        subprocess.run(["ego4d", "--output_directory", out, "--datasets", "clips",
                        "--video_uids", *todo[i:i + batch], "-y"], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nlq", required=True, help="nlq_val.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--clips", type=int, default=20, help="clips through the VLM")
    ap.add_argument("--extra", type=int, default=20, help="more clips for the trigger-only test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-seconds", type=float, default=600)
    ap.add_argument("--skip", type=int, default=0, help="leave out the first N (development) clips")
    a = ap.parse_args()
    nlq = load_nlq(a.nlq)
    uids = select_clips(nlq, a.skip + a.clips + a.extra, a.seed, a.max_seconds)[a.skip:]
    download(uids, a.out)
    sel = []
    for i, u in enumerate(uids):
        path = find(a.out, u)
        if path is None:
            print(f"missing after download: {u}")
            continue
        c = nlq[u]
        sel.append({"clip_uid": u, "video_uid": c["video_uid"], "seconds": c["seconds"],
                    "questions": len(c["questions"]), "path": path, "full": i < a.clips})
    with open(os.path.join(a.out, "selected.json"), "w") as f:
        json.dump(sel, f, indent=1)
    full = [s for s in sel if s["full"]]
    print(f"{len(sel)} clips ({len(full)} full, {len(sel) - len(full)} trigger-only), "
          f"{sum(s['seconds'] for s in sel) / 3600:.1f} h, {sum(s['questions'] for s in full)} questions on the full clips "
          f"→ {a.out}/selected.json")


if __name__ == "__main__":
    main()
