# evaluation/ego4d_nlq.py
# Experiment D: Mnemo on Ego4D Episodic Memory, Natural Language Queries (NLQ), validation split.
# Real first-person videos from many places (8-min clips) and questions written by people
# ("Where did I put the drill?"), each with the time window where the answer can be seen.
#
# Per clip (ONE VLM pass over every 4-s window, cached like evaluation/benchmark.py):
#   1. TRIGGER: of the moments people asked about, how many did the trigger send to the VLM,
#      vs every N-th window with the same number of calls? Needs only the trigger, so extra clips can
#      be scored without the VLM (trigger_only_clip).
#   2. DIARY: is there a diary row at the asked-about moment that names the object?
#   3. ANSWER: query/locate.py picks one diary line → its time window, compared with the person's window:
#      hit ±2 s / ±10 s (the windows overlap once the true one is widened), and the official
#      R@1 at IoU 0.3 / 0.5. Each diary is scored with the model picking the line and with the best
#      search match alone ("(search only)"). Baseline "direct": the VLM gets 32 frames of the whole clip
#      and is asked for the time.
# Everything is also split by question type and by how well a diary of hand actions fits it (fit()).
#
#   from evaluation.ego4d_nlq import load_nlq, run_nlq
#   run_nlq(clips, nlq, describer, gate, out_dir, trigger_only=extra_clips)

import json
import os
import re
from collections import defaultdict

import numpy as np

import config as C
from evaluation.benchmark import START, THRESHOLDS, _evenly, _pad, _windows_from_json, _windows_to_json
from evaluation.qa_benchmark import clip_frames
from modules.hoi_gate import gate_decisions
from modules.vlm_logger import ActionLogger, replay
from query.llm_qa import norm_tokens
from query.locate import diary_line, locate

TOL_S, WIDE_S = 2.0, 10.0
DIRECT = "direct VLM (no memory)"
GROUPS = ("good", "partial", "poor")


def fit(template: str | None) -> str:
    """How well a diary of hand actions can answer this question type."""
    t = (template or "").lower()
    if t.startswith("people") or "how many" in t or "what x is y" in t:
        return "poor"                     # people, counts, colours: not in the diary
    if any(k in t for k in ("where did i put", "what did i put", "before / after", "what x did i y", "state of")):
        return "good"                     # hand actions (+ where), object states
    return "partial"                      # things only SEEN ("where is X", "where did I see X")


def load_nlq(path: str) -> dict:
    """{clip_uid: {"clip_uid", "video_uid", "seconds", "questions": [{"q", "template", "slot", "t0", "t1"}]}}"""
    with open(path) as f:
        data = json.load(f)
    clips = {}
    for v in data["videos"]:
        for c in v["clips"]:
            qs = []
            for a in c["annotations"]:
                for q in a["language_queries"]:
                    text = re.sub(r"^\s*query text\s*:\s*", "", q.get("query") or "", flags=re.I).strip()
                    if text and q["clip_end_sec"] >= q["clip_start_sec"]:
                        qs.append({"q": text, "template": q.get("template"), "slot": (q.get("slot_x") or "").strip(),
                                   "t0": float(q["clip_start_sec"]), "t1": float(q["clip_end_sec"])})
            if qs:
                clips[c["clip_uid"]] = {"clip_uid": c["clip_uid"], "video_uid": v["video_uid"],
                                        "seconds": c["clip_end_sec"] - c["clip_start_sec"], "questions": qs}
    return clips


def select_clips(nlq: dict, n: int, seed: int = 0, max_seconds: float = 600) -> list[str]:
    """n clips, at most one per source video (different people and places), clips of ≤ max_seconds."""
    rng = np.random.default_rng(seed)
    by_video = defaultdict(list)
    for c in sorted(nlq.values(), key=lambda c: c["clip_uid"]):
        if c["seconds"] <= max_seconds:
            by_video[c["video_uid"]].append(c["clip_uid"])
    videos = sorted(by_video)
    rng.shuffle(videos)
    return [by_video[v][rng.integers(len(by_video[v]))] for v in videos[:n]]


def _windows(t0: float, t1: float, n: int, w: float = C.VLM_WINDOW_S) -> slice:
    return slice(min(int(t0 // w), n - 1), min(int(max(t0, t1 - 1e-6) // w), n - 1) + 1)


def trigger_coverage(scores: np.ndarray, questions: list[dict], n: int | None = None) -> dict:
    """Per system: VLM calls and, per question, whether a called window overlaps the asked-about moment."""
    n = n or len(scores)
    systems = {"every window": np.ones(n, bool)}
    for th in THRESHOLDS:
        call = _pad(gate_decisions(scores, th, hangover=0), n)
        systems[f"mnemo@{th}"] = call
        systems[f"every-nth = mnemo@{th}"] = _evenly(n, int(call.sum()))
    return {s: {"calls": int(c.sum()), "windows": n, "covered": [bool(c[_windows(q["t0"], q["t1"], n)].any())
                                                                  for q in questions]}
            for s, c in systems.items()}


def judge(pred: tuple | None, q: dict) -> dict:
    if pred is None:
        return {"hit2": False, "hit10": False, "iou": 0.0}
    a0, a1 = pred
    inter = max(0.0, min(a1, q["t1"]) - max(a0, q["t0"]))
    union = max(a1, q["t1"]) - min(a0, q["t0"])
    near = lambda tol: a0 <= q["t1"] + tol and a1 >= q["t0"] - tol
    return {"hit2": near(TOL_S), "hit10": near(WIDE_S), "iou": inter / union if union > 0 else 0.0}


def logged(rows: list[dict], q: dict) -> bool:
    """A diary row near the asked-about moment that names the asked-about object."""
    want = norm_tokens(q["slot"] or q["q"])
    return any(r["verb"] and r["t"] <= q["t1"] + TOL_S and r["t_end"] >= q["t0"] - TOL_S
               and want & norm_tokens(" ".join(str(r.get(f) or "") for f in ("verb", "object", "where")))
               for r in rows)


def parse_seconds(text: str) -> float | None:
    m = re.search(r"(\d+):(\d{1,2}(?:\.\d+)?)", text or "")
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    m = re.search(r"\d+(?:\.\d+)?", text or "")
    return float(m.group()) if m else None


def _answers(questions, preds, rows=None):
    out = []
    for q, p in zip(questions, preds):
        a = dict(q, pred=list(p) if p else None, fit=fit(q["template"]), **judge(p, q))
        a["logged"] = logged(rows, q) if rows is not None else None
        out.append(a)
    return out


def run_clip(video: str, clip: dict, describer, gate, cache_dir: str, direct: bool = True,
             verbose: bool = True) -> dict:
    uid, qs = clip["clip_uid"], clip["questions"]
    cache = os.path.join(cache_dir, f"{uid}__{C.VLM_PROMPT_VERSION}.json")
    if os.path.exists(cache):
        with open(cache) as f:
            c = json.load(f)
        windows, stats, scores = _windows_from_json(c["windows"]), c["stats"], np.array(c["scores"])
        print(f"  (VLM answers loaded from cache: {cache})")
    else:
        sig, gstats = gate.signals(video)
        scores = sig["hoi"]
        logger = ActionLogger(describer, store=None, track_states=True)
        _, stats = logger.run(video, start_time=START, verbose=verbose, scores=scores)
        windows = logger.windows
        with open(cache, "w") as f:
            json.dump({"windows": _windows_to_json(windows), "stats": stats, "scores": scores.tolist(),
                       "gate": gstats}, f)
    n = len(windows)
    main = f"mnemo@{C.GATE_THRESHOLD}"
    call = _pad(gate_decisions(scores, C.GATE_THRESHOLD, hangover=0), n)
    systems = {"every window": None, main: call, f"every-nth = {main}": _evenly(n, int(call.sum()))}
    res = {"clip": uid, "video_uid": clip["video_uid"], "seconds": stats["video_seconds"], "questions": qs,
           "trigger": trigger_coverage(scores, qs, n), "systems": {}}
    for name, c in systems.items():
        rows = replay(windows, c, track_states=True)
        calls = n if c is None else int(c.sum())
        for sname, llm in ((name, describer), (f"{name} (search only)", None)):
            preds = []
            for q in qs:
                p = locate(q["q"], rows, llm)
                preds.append((p["t0"], p["t1"]) if p else None)
            res["systems"][sname] = {"calls": calls, "windows": n, "answers": _answers(qs, preds, rows)}
        if name == main:
            res["log"] = [diary_line(r) for r in rows if r["verb"]]
    if direct:
        frames, sec = clip_frames(video, 32)
        preds = []
        for q in qs:
            t = parse_seconds(describer.ask_time(frames, sec, q["q"]))
            w = C.VLM_WINDOW_S / 2
            preds.append((max(0.0, t - w), t + w) if t is not None else None)
        res["systems"][DIRECT] = {"calls": None, "windows": n, "answers": _answers(qs, preds)}
    if verbose:
        a = res["systems"][main]["answers"]
        print(f"  {main}: VLM on {systems[main].sum()}/{n} windows | {len(a)} questions | "
              f"diary has the moment {sum(x['logged'] for x in a)} | hit ±2 s {sum(x['hit2'] for x in a)}")
    return res


def trigger_only_clip(video: str, clip: dict, gate, cache_dir: str) -> dict:
    """Trigger coverage alone (no VLM): cheap, so it can run on many more clips."""
    cache = os.path.join(cache_dir, f"{clip['clip_uid']}__gate.json")
    if os.path.exists(cache):
        with open(cache) as f:
            scores = np.array(json.load(f)["scores"])
    else:
        sig, gstats = gate.signals(video)
        scores = sig["hoi"]
        with open(cache, "w") as f:
            json.dump({"scores": scores.tolist(), "gate": gstats}, f)
    return {"clip": clip["clip_uid"], "video_uid": clip["video_uid"], "seconds": clip["seconds"],
            "questions": clip["questions"], "trigger": trigger_coverage(scores, clip["questions"])}


def _pct(a, b):
    return f"{a}/{b} ({a / b:.0%})" if b else "-"


def summarize(results: list[dict]) -> str:
    main = f"mnemo@{C.GATE_THRESHOLD}"
    full = [r for r in results if "systems" in r]
    hours = sum(r["seconds"] for r in results) / 3600
    md = [f"# Experiment D: Ego4D NLQ (validation split)\n",
          f"{len(results)} clips from {len({r['video_uid'] for r in results})} different videos, {hours:.1f} h; "
          f"{len(full)} clips also through the VLM. Hit ±2 s / ±10 s = the answer's window overlaps the "
          f"person's window widened by 2 / 10 s. IoU = the official NLQ overlap measure (R@1).\n"]

    md += ["## 1. Trigger: asked-about moments sent to the VLM\n",
           "| System | VLM calls | Moments covered | Good-fit questions covered |", "|---|---|---|---|"]
    for s in results[0]["trigger"]:
        calls = sum(r["trigger"][s]["calls"] for r in results)
        wins = sum(r["trigger"][s]["windows"] for r in results)
        cov = [(c, fit(q["template"])) for r in results for c, q in zip(r["trigger"][s]["covered"], r["questions"])]
        good = [c for c, g in cov if g == "good"]
        md.append(f"| {s} | {calls / wins:.0%} | {_pct(sum(c for c, _ in cov), len(cov))} | {_pct(sum(good), len(good))} |")

    if full:
        md += ["\n## 2. Answers: does Mnemo point to the right moment?\n",
               "| System | Questions | Diary has the moment | Hit ±2 s | Hit ±10 s | IoU ≥ 0.3 | IoU ≥ 0.5 | No answer |",
               "|---|---|---|---|---|---|---|---|"]
        for s in full[0]["systems"]:
            a = [x for r in full for x in r["systems"][s]["answers"]]
            n = len(a)
            lg = _pct(sum(x["logged"] for x in a), n) if a and a[0]["logged"] is not None else "-"
            md.append(f"| {s} | {n} | {lg} | {_pct(sum(x['hit2'] for x in a), n)} | {_pct(sum(x['hit10'] for x in a), n)} | "
                      f"{_pct(sum(x['iou'] >= 0.3 for x in a), n)} | {_pct(sum(x['iou'] >= 0.5 for x in a), n)} | "
                      f"{_pct(sum(x['pred'] is None for x in a), n)} |")

        md += ["\n## 3. By how well a hand-action diary fits the question\n",
               "good = put / put in / before-after an action / object state; partial = things only seen; "
               "poor = people, counts, colours.\n",
               f"| Fit | Questions | Diary has the moment | Hit ±2 s ({main}) | Hit ±2 s (direct) |", "|---|---|---|---|---|"]
        for g in GROUPS:
            a = [x for r in full for x in r["systems"][main]["answers"] if x["fit"] == g]
            d = [x for r in full for x in r["systems"].get(DIRECT, {"answers": []})["answers"] if x["fit"] == g]
            md.append(f"| {g} | {len(a)} | {_pct(sum(x['logged'] for x in a), len(a))} | "
                      f"{_pct(sum(x['hit2'] for x in a), len(a))} | {_pct(sum(x['hit2'] for x in d), len(d))} |")

        md += [f"\n## 4. By question type ({main})\n",
               "| Question type | Fit | Questions | Diary has the moment | Hit ±2 s | IoU ≥ 0.3 |", "|---|---|---|---|---|---|"]
        by_t = defaultdict(list)
        for r in full:
            for x in r["systems"][main]["answers"]:
                by_t[x["template"]].append(x)
        for t, a in sorted(by_t.items(), key=lambda kv: -len(kv[1])):
            md.append(f"| {t} | {fit(t)} | {len(a)} | {_pct(sum(x['logged'] for x in a), len(a))} | "
                      f"{_pct(sum(x['hit2'] for x in a), len(a))} | {_pct(sum(x['iou'] >= 0.3 for x in a), len(a))} |")
    return "\n".join(md) + "\n"


def run_nlq(clips: list[dict], nlq: dict, describer, gate, out_dir: str, trigger_only: list[dict] = (),
            direct: bool = True) -> tuple[list[dict], str]:
    """clips / trigger_only: [{"clip_uid", "path"}]. Writes ego4d_nlq.json + ego4d_nlq.md to out_dir."""
    cache_dir = os.path.join(out_dir, "vlm_cache")
    os.makedirs(cache_dir, exist_ok=True)
    results = []

    def save():
        with open(os.path.join(out_dir, "ego4d_nlq.json"), "w") as f:
            json.dump(results, f)

    for i, c in enumerate(clips, 1):
        clip = nlq[c["clip_uid"]]
        print(f"\n[{i}/{len(clips)}] {c['clip_uid']} ({clip['seconds'] / 60:.1f} min, {len(clip['questions'])} questions)")
        res = run_clip(c["path"], clip, describer, gate, cache_dir, direct=direct)
        results.append(res)
        save()
    for i, c in enumerate(trigger_only, 1):
        clip = nlq[c["clip_uid"]]
        print(f"[trigger only {i}/{len(trigger_only)}] {c['clip_uid']}")
        res = trigger_only_clip(c["path"], clip, gate, cache_dir)
        results.append(res)
        save()
    md = summarize(results)
    with open(os.path.join(out_dir, "ego4d_nlq.md"), "w") as f:
        f.write(md)
    return results, md
