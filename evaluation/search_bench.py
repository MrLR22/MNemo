# evaluation/search_bench.py
# Search-only benchmark: scores the search part (search/) on diaries already written by a Kaggle run, on a laptop,
# without a GPU and without any video. Input: Ego4D NLQ result folders (evaluation/ego4d_nlq.py output:
# ego4d_nlq.json + vlm_cache/ with the big model's text and, if the run had them, image fingerprints).
#
# For every clip the diary is rebuilt for each system (every window, trigger at 0.6, every-nth with the same calls)
# and every question is answered by the BEST SEARCH MATCH alone (no language model picks a line, so the numbers are
# lower than the Kaggle tables, where the model picks among the top 20). Hit ±2 s = the found window overlaps the
# person's window widened by 2 s.
#
# Clips scored here become development clips: a change chosen by looking at these numbers must be confirmed on
# clips it was not tuned on.
#
#   python -m evaluation.search_bench data/kaggle_results/v4/ego4d [more folders] [--out results.md]

import argparse
import glob
import json
import os

import numpy as np

import config as C
from evaluation.benchmark import _evenly, _pad, _windows_from_json
from evaluation.ego4d_nlq import judge, paired
from modules.hoi_gate import gate_decisions
from modules.vlm_logger import replay
from query.locate import locate
from search.rank import hybrid


def load_clips(folders: list[str]) -> list[dict]:
    """[{"clip", "questions", "windows", "scores", "fps" (or None), "prompt"}] for the clips run through the VLM."""
    clips = []
    for d in folders:
        with open(os.path.join(d, "ego4d_nlq.json")) as f:
            results = json.load(f)
        for r in results:
            if "systems" not in r:                   # trigger-only clip: no diary
                continue
            cache = [p for p in glob.glob(os.path.join(d, "vlm_cache", f"{r['clip']}__*.json"))
                     if not p.endswith("__gate.json")]
            if not cache:
                continue
            with open(cache[0]) as f:
                c = json.load(f)
            fp = os.path.join(d, "vlm_cache", f"{r['clip']}__fingerprints.npy")
            clips.append({"clip": r["clip"], "questions": r["questions"], "windows": _windows_from_json(c["windows"]),
                          "scores": np.array(c["scores"]), "fps": np.load(fp) if os.path.exists(fp) else None,
                          "prompt": cache[0].rsplit("__", 1)[1].removesuffix(".json")})
    return clips


def systems(n: int, scores: np.ndarray, threshold: float = C.GATE_THRESHOLD) -> dict:
    call = _pad(gate_decisions(scores, threshold, hangover=0), n)
    return {"every window": None, f"mnemo@{threshold}": call, f"every-nth = mnemo@{threshold}": _evenly(n, int(call.sum()))}


def searches(text=None, clip=None, fps=None, has_seen: bool = False) -> dict:
    """{name: (keep SEEN objects?, candidates function or None = keyword)}"""
    images = (fps, clip) if clip is not None and fps is not None else None
    s = {"keyword": (False, None)}
    if text is not None:
        s["keyword + meaning"] = (False, hybrid(text=text))
    if images:
        s["keyword + image"] = (False, hybrid(images=images))
    if text is not None and images:
        s["keyword + meaning + image"] = (False, hybrid(text=text, images=images))
    if has_seen:
        s["+ SEEN objects, keyword"] = (True, None)
        if text is not None and images:
            s["+ SEEN objects, keyword + meaning + image"] = (True, hybrid(text=text, images=images))
    return s


def run(clips: list[dict], text=None, clip=None) -> list[dict]:
    """Per clip: {"clip", "systems": {"<system> | <search>": {"answers": [...]}}} (the shape ego4d_nlq.paired reads)."""
    out = []
    for c in clips:
        n = len(c["windows"])
        has_seen = any("SEEN:" in (w["text"] or "") for w in c["windows"])
        res = {"clip": c["clip"], "systems": {}}
        for sys_name, call in systems(n, c["scores"]).items():
            rows = replay(c["windows"], call, track_states=True)
            for s_name, (seen, cand) in searches(text, clip, c["fps"], has_seen).items():
                rr = rows if seen else [dict(r, seen=None) for r in rows]
                answers = []
                for q in c["questions"]:
                    p = locate(q["q"], rr, None, candidates=cand)
                    answers.append(dict(judge((p["t0"], p["t1"]) if p else None, q), none=p is None))
                res["systems"][f"{sys_name} | {s_name}"] = {"answers": answers}
        out.append(res)
    return out


def report(results: list[dict]) -> str:
    names = list(results[0]["systems"]) if results else []
    nq = sum(len(r["systems"][names[0]]["answers"]) for r in results) if names else 0
    md = [f"## Search only ({len(results)} clips, {nq} questions, best search match, no language model)\n",
          "| System | Search | Hit ±2 s | Hit ±10 s | No answer |", "|---|---|---|---|---|"]
    for s in names:
        a = [x for r in results for x in r["systems"][s]["answers"]]
        sys_name, s_name = s.split(" | ")
        md.append(f"| {sys_name} | {s_name} | {sum(x['hit2'] for x in a)} ({sum(x['hit2'] for x in a) / len(a):.0%}) "
                  f"| {sum(x['hit10'] for x in a)} | {sum(x['none'] for x in a)} |")
    main = f"mnemo@{C.GATE_THRESHOLD}"
    pairs = [(s, f"{main} | keyword") for s in names if s.startswith(f"{main} | ") and s != f"{main} | keyword"]
    best = f"{main} | keyword + meaning + image"
    if best in names:
        pairs += [(best, f"every-nth = {main} | keyword + meaning + image"), (best, "every window | keyword + meaning + image")]
    if pairs:
        md += ["\n## Paired differences (hit ±2 s; 95% range from resampling clips)\n",
               "| System | vs | Hits | Difference [95% range] | Clips better / equal / worse |", "|---|---|---|---|---|"]
        for a, b in pairs:
            p = paired(results, a, b)
            show = lambda x: x.replace(" | ", ", ")
            md.append(f"| {show(a)} | {show(b)} | {p['a']} vs {p['b']} | {p['diff']:+d} [{p['low']:+d}, {p['high']:+d}] "
                      f"| {p['better']} / {p['clips'] - p['better'] - p['worse']} / {p['worse']} |")
    return "\n".join(md) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+", help="Ego4D result folders (with ego4d_nlq.json and vlm_cache/)")
    ap.add_argument("--no-models", action="store_true", help="keyword search only (no text / image model)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    clips = load_clips(a.folders)
    text = clip = None
    if not a.no_models:
        from modules.landmarks import ClipEmbedder
        from modules.text_embed import TextEmbedder
        text, clip = TextEmbedder(), ClipEmbedder()
    print(f"{len(clips)} clips, prompts: {sorted({c['prompt'] for c in clips})}, "
          f"with image fingerprints: {sum(c['fps'] is not None for c in clips)}")
    md = report(run(clips, text, clip))
    print(md)
    if a.out:
        with open(a.out, "w") as f:
            f.write(md)


if __name__ == "__main__":
    main()
