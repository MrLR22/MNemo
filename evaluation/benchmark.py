# evaluation/benchmark.py
# The full Mnemo benchmark on EPIC-KITCHENS clips of people the trigger model never trained on.
#
# Per clip: ONE pass of the VLM over every 4-s window (raw answers cached on disk), the HOI trigger
# scores, then the log is rebuilt for each system from the same answers, so every system is compared
# on identical VLM output:
#   captions     VLM every window, actions only                 (= plain "caption + LLM" memory)
#   + states     VLM every window + object-state tracking
#   mnemo@T      HOI trigger at threshold T + object-state tracking (VLM only on triggered windows)
#   mnemo v2     trigger with the settings chosen on VALIDATION people (evaluation/tune_gate.py):
#                HOI threshold + keep-going + hand motion
#   mnemo@T+everyK  trigger, plus at least one described window every K windows (hybrid)
#   every-nth    for each trigger system: the SAME number of VLM calls per clip, evenly spaced, no model
#                (the fair baseline: does the trigger pick better windows than a fixed rate?)
# Stage 1 scores: evaluation/log_accuracy.py. Stage 2 (questions): evaluation/qa_benchmark.py,
# including the no-memory baseline "direct" (ask the VLM about the whole clip).
#
# Used from notebooks/mnemo_kaggle.ipynb:
#   from evaluation.benchmark import run_benchmark
#   run_benchmark(clips, describer, gate, ann_csv, classes_dir, out_dir)

import json
import os
import time
from datetime import datetime, timedelta

import numpy as np

import config as C
from evaluation.gating_experiment import coverage
from evaluation.log_accuracy import (EpicVocab, clip_name_parts, load_truth, merge_scores, print_details,
                                     score_log)
from evaluation.qa_benchmark import ask_direct, ask_mnemo, make_questions, score_answers
from modules.hoi_gate import gate_decisions
from modules.storage import EventStore
from modules.vlm_logger import ActionLogger, replay

START = datetime(2026, 10, 16, 9, 0)
THRESHOLDS = (0.5, 0.6, 0.7, 0.8)
HYBRIDS = ((0.7, 3), (0.8, 3), (0.8, 2))     # (threshold, describe at least every n windows)


def _pad(call: np.ndarray, n: int) -> np.ndarray:
    return np.concatenate([call, np.ones(max(0, n - len(call)), bool)])[:n] if n else call


def _evenly(n: int, k: int) -> np.ndarray:
    """k of n windows, evenly spaced."""
    m = np.zeros(n, bool)
    if k > 0:
        m[np.minimum(((np.arange(k) + 0.5) * n / k).astype(int), n - 1)] = True
    return m


def _windows_to_json(ws):
    return [dict(w, ts=w["ts"].isoformat()) for w in ws]


def _windows_from_json(ws):
    return [dict(w, ts=datetime.fromisoformat(w["ts"])) for w in ws]


def run_clip(video, describer, gate, truth, vocab, cache_dir, thresholds=THRESHOLDS, qa=True, verbose=True,
             gate_config: dict | None = None):
    name = os.path.splitext(os.path.basename(video))[0]
    cache = os.path.join(cache_dir, f"{name}__{C.VLM_PROMPT_VERSION}.json")      # new prompt → new VLM pass
    if os.path.exists(cache):
        with open(cache) as f:
            c = json.load(f)
        windows, stats, gstats = _windows_from_json(c["windows"]), c["stats"], c["gate"]
        scores, motion = np.array(c["scores"]), np.array(c.get("motion", np.zeros(len(c["scores"]))))
        print(f"  (VLM answers loaded from cache: {cache})")
    else:
        sig, gstats = gate.signals(video)
        scores, motion = sig["hoi"], sig["motion"]
        if verbose:
            print(f"  trigger: {gstats['frames']} frames in {gstats['gate_seconds']}s | window scores "
                  + " ".join(f"{s:.2f}" for s in scores))
        logger = ActionLogger(describer, store=None, track_states=True)
        _, stats = logger.run(video, start_time=START, verbose=verbose, scores=scores)
        windows = logger.windows
        with open(cache, "w") as f:
            json.dump({"windows": _windows_to_json(windows), "stats": stats, "scores": scores.tolist(),
                       "motion": motion.tolist(), "gate": gstats}, f)
    n = len(windows)
    video_s = stats["video_seconds"]
    systems = {"captions": (None, False), "+ states": (None, True)}
    for th in thresholds:
        systems[f"mnemo@{th}"] = (_pad(gate_decisions(scores, th, hangover=0), n), True)
    for th, hb in HYBRIDS:              # trigger + a look at least every hb windows (keeps state checks regular)
        systems[f"mnemo@{th}+every{hb}"] = (_pad(gate_decisions(scores, th, heartbeat_windows=hb, hangover=0), n), True)
    if gate_config:
        g = gate_config
        systems["mnemo v2"] = (_pad(gate_decisions(scores, g["threshold"], motion=motion,
                                                   motion_threshold=g["motion_threshold"],
                                                   hangover=g["hangover"]), n), True)
    qa_gate = "mnemo v2" if gate_config else f"mnemo@{C.GATE_THRESHOLD}"
    ordered = {}
    for sname, sys_ in systems.items():                  # each trigger row followed by its fixed-rate twin
        ordered[sname] = sys_
        if sname.startswith("mnemo"):
            ordered[f"every-nth = {sname}"] = (_evenly(n, int(sys_[0].sum())), True)
    systems = ordered

    res = {"clip": name, "video_seconds": video_s, "true_actions": len(truth), "systems": {}}
    for sname, (call, states) in systems.items():
        rows = replay(windows, call, track_states=states)
        sc = score_log(rows, truth, vocab)
        called = [w for w in windows if w["text"] is not None and (call is None or call[w["k"]])]
        mask = np.zeros(n, bool)
        mask[[w["k"] for w in called]] = True
        seen = coverage([mask], [[(g["start"], g["stop"], "") for g in truth]], C.VLM_WINDOW_S)
        res["systems"][sname] = {
            "seen": round(seen * len(truth)),
            "score": {k: v for k, v in sc.items() if k not in ("per_truth", "unmatched")},
            "vlm_calls": len(called), "windows": n,
            "vlm_rtf": sum(w["dt"] for w in called) / video_s if video_s else None,
            "gate_rtf": gstats["gate_real_time_factor"] if call is not None else 0.0,
            "log": [f"{r['t']:5.1f}-{r['t_end']:5.1f}s | {r['verb'] or '-'} {r['object'] or ''}"
                    + (" [state]" if r.get("source") == "state" else "") for r in rows],
        }
        res["systems"][sname]["_rows"] = rows
        if sname == qa_gate and verbose:
            print(f"  {sname}: VLM on {len(called)}/{n} windows")
            print_details(sc)

    if qa:
        qs = make_questions(truth, vocab)
        res["questions"] = {}
        qa_systems = ["captions", "+ states", qa_gate]
        for sname in qa_systems:
            store = EventStore(":memory:")
            replay(windows, systems[sname][0], track_states=systems[sname][1], store=store)
            res["questions"][sname] = ask_mnemo(qs, store, describer, START + timedelta(hours=1))
        res["questions"]["direct VLM (no memory)"] = ask_direct(qs, describer, video)
        if verbose:
            for i, q in enumerate(qs):
                marks = " ".join(("Y" if res["questions"][s][i]["yes"] else "n") for s in res["questions"])
                print(f"   {'should be YES' if q['expect'] else 'should be NO '} | {marks} | {q['q']}")
            print(f"   (columns: {' | '.join(res['questions'])})")
    for s in res["systems"].values():
        s.pop("_rows")
    return res


def summarize(results: list[dict], thresholds=THRESHOLDS) -> str:
    pct = lambda x: f"{x:.0%}" if x is not None else "-"
    names = list(results[0]["systems"]) if results else []
    minutes = sum(r["video_seconds"] for r in results) / 60
    n_true = sum(r["true_actions"] for r in results)
    lines = [f"## Stage 1: the action log ({len(results)} clips, {minutes:.0f} min, {n_true} true EPIC actions, "
             f"people the trigger never trained on)", "",
             "| System | VLM calls | VLM time / video time | Actions the VLM got to see | Actions found (recall) "
             "| Open/close/on/off found | Log rows correct (precision) |", "|---|---|---|---|---|---|---|"]
    for s in names:
        sc = merge_scores([r["systems"][s]["score"] for r in results])
        calls = sum(r["systems"][s]["vlm_calls"] for r in results)
        wins = sum(r["systems"][s]["windows"] for r in results)
        vlm = sum(r["systems"][s]["vlm_rtf"] * r["video_seconds"] for r in results) / (minutes * 60)
        seen = sum(r["systems"][s].get("seen", 0) for r in results)
        lines.append(f"| {s} | {pct(calls / wins)} | {vlm:.2f} | {pct(seen / n_true if n_true else None)} "
                     f"| {pct(sc['recall'])} ({sc['found']}/{sc['true_actions']}) "
                     f"| {pct(sc['state_recall'])} ({sc['state_found']}/{sc['state_true']}) | {pct(sc['precision'])} |")
    g = [r["systems"][names[-1]]["gate_rtf"] for r in results if names]
    if g:
        lines += ["", f"Trigger (hands + objects + HOI-TCN) costs {np.mean(g):.2f} s per video second on this "
                      "machine; it can run on the CPU in parallel with the VLM on the GPU."]
    qa = [r for r in results if r.get("questions")]
    if qa:
        lines += ["", "## Stage 2: memory questions (auto-generated from the ground truth)", "",
                  "| System | 'Yes' questions answered yes | 'No' questions answered yes (hallucination) | Accuracy |",
                  "|---|---|---|---|"]
        for s in qa[0]["questions"]:
            sc = score_answers([x for r in qa for x in r["questions"][s]])
            lines.append(f"| {s} | {pct(sc['recall'])} ({sc['yes_answered_yes']}/{sc['yes_questions']}) "
                         f"| {pct(sc['hallucination'])} ({sc['no_answered_yes']}/{sc['no_questions']}) "
                         f"| {pct(sc['accuracy'])} |")
    return "\n".join(lines)


def _duration(video: str, cache_dir: str | None = None) -> float:
    name = os.path.splitext(os.path.basename(video))[0]
    cached = os.path.join(cache_dir or "", f"{name}__{C.VLM_PROMPT_VERSION}.json")
    if not os.path.exists(video) and cache_dir and os.path.exists(cached):   # re-scoring saved answers
        with open(cached) as f:
            return json.load(f)["stats"]["video_seconds"]
    import cv2
    cap = cv2.VideoCapture(video)
    n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return n / fps


def run_benchmark(clips, describer, gate, ann_csv, classes_dir, out_dir, thresholds=THRESHOLDS, qa=True,
                  gate_config: dict | None = None):
    """gate_config: the 'chosen' dict from evaluation/tune_gate.py (adds the 'mnemo v2' system)."""
    thresholds = tuple(sorted(set(thresholds) | {C.GATE_THRESHOLD}))
    vocab = EpicVocab(os.path.join(classes_dir, "EPIC_100_verb_classes.csv"),
                      os.path.join(classes_dir, "EPIC_100_noun_classes.csv"))
    cache_dir = os.path.join(out_dir, "vlm_cache")
    os.makedirs(cache_dir, exist_ok=True)
    results = []
    for i, video in enumerate(clips, 1):
        name = os.path.splitext(os.path.basename(video))[0]
        vid, t0 = clip_name_parts(name)
        truth = load_truth(ann_csv, vid, t0, _duration(video, cache_dir))
        print(f"\n[{i}/{len(clips)}] {name}: {len(truth)} true actions")
        t = time.time()
        try:
            results.append(run_clip(video, describer, gate, truth, vocab, cache_dir, thresholds, qa,
                                    gate_config=gate_config))
        except Exception as e:                                   # one broken clip must not stop the run
            print(f"  FAILED: {type(e).__name__}: {e}")
            continue
        print(f"  done in {time.time() - t:.0f}s")
        with open(os.path.join(out_dir, "benchmark.json"), "w") as f:
            json.dump(results, f, indent=1, default=str)
    md = summarize(results, thresholds)
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(md + "\n")
    print("\n" + md)
    return results, md
