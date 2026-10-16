# evaluation/long_qa.py
# EXPERIMENT C: the real use of a memory aid. Questions about a whole MORNING, not about one minute.
#
# The benchmark's test clips are put into one morning: clip i starts at 09:00 + i * GAP_S (default
# 7.5 min), so 16 one-minute clips span 2 hours. Questions are generated from the ground truth of the
# whole morning (as in qa_benchmark.py), plus "When did I ...?" questions for things that happened.
#
#   mnemo    the morning's log (trigger at GATE_THRESHOLD + object states), answers only from it
#   direct   the VLM gets 32 frames spread over all the clips (generous: the idle gaps between clips
#            are not even shown) plus the question; it has no clock, so it cannot answer "when"
#
# Uses the VLM answers cached by evaluation/benchmark.py, so run the benchmark first.

import json
import os
import re
from datetime import timedelta

import numpy as np

import config as C
from evaluation.benchmark import START, _duration, _pad, _windows_from_json
from evaluation.log_accuracy import EpicVocab, clip_name_parts, load_truth
from evaluation.qa_benchmark import YES_VERBS, ask_mnemo, clip_frames, make_questions, says_yes, score_answers
from modules.hoi_gate import gate_decisions
from modules.storage import EventStore
from modules.vlm_logger import replay

GAP_S = 450.0
TIME_TOL_S = 90.0
_TIME = re.compile(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\s*([ap]\.?m\.?)?", re.I)


def build_morning(clips, cache_dir, ann_csv, vocab, gap_s=GAP_S, threshold=None, db_path=":memory:"):
    """→ (store with the whole morning's log, truth with times in seconds after START, VLM call share).
    db_path: also keep the morning as a database file (the app's demo data)."""
    threshold = C.GATE_THRESHOLD if threshold is None else threshold
    if db_path != ":memory:" and os.path.exists(db_path):
        os.remove(db_path)
    store = EventStore(db_path)
    truth_all, calls, wins = [], 0, 0
    for i, video in enumerate(clips):
        name = os.path.splitext(os.path.basename(video))[0]
        path = os.path.join(cache_dir, f"{name}__{C.VLM_PROMPT_VERSION}.json")
        if not os.path.exists(path):
            print(f"  {name}: no cached VLM answers (run the benchmark first), skipped")
            continue
        with open(path) as f:
            c = json.load(f)
        off = i * gap_s
        ws = [dict(w, ts=w["ts"] + timedelta(seconds=off), t=w["t"] + off) for w in _windows_from_json(c["windows"])]
        call = _pad(gate_decisions(np.array(c["scores"]), threshold, hangover=0), len(ws))
        calls, wins = calls + int(call.sum()), wins + len(ws)
        replay(ws, call, track_states=True, store=store)
        vid, t0 = clip_name_parts(name)
        for g in load_truth(ann_csv, vid, t0, _duration(video, cache_dir)):
            truth_all.append(dict(g, start=g["start"] + off, stop=g["stop"] + off, clip=name))
    return store, truth_all, (calls / wins if wins else 0.0)


def when_questions(truth, vocab, yes_questions):
    """'When did I open the fridge?' for each yes-question, with every true time it happened."""
    out = []
    for q in yes_questions:
        m = re.match(r"Did I (.+) the (.+)\?", q["q"])
        if not m:
            continue
        verb, noun = m.groups()
        times = [g["start"] for g in truth if YES_VERBS.get(vocab.verb_key[g["verb_class"]]) == verb
                 and vocab.noun_text(g["noun_class"]) == noun]
        if times:
            out.append({"q": f"When did I {verb} the {noun}?", "times": times})
    return out


def time_correct(answer: str, true_secs: list[float], start=START, tol=TIME_TOL_S) -> bool:
    """Does the answer name a clock time within tol of one of the true times? (minutes-only times get
    one extra minute of tolerance for rounding)"""
    for h, mi, s, ap in _TIME.findall(answer or ""):
        h = int(h)
        if ap and ap.lower().startswith("p") and h < 12:
            h += 12
        t = start.replace(hour=h % 24, minute=int(mi), second=int(s or 0))
        d = (t - start).total_seconds()
        if any(abs(d - x) <= tol + (0 if s else 60) for x in true_secs):
            return True
    return False


def morning_frames(clips, n_total=32):
    frames, secs = [], 0.0
    per = max(2, n_total // max(1, len(clips)))
    for v in clips:
        f, s = clip_frames(v, per)
        frames += f
        secs += s
    return frames, secs


def run_long_qa(clips, describer, ann_csv, classes_dir, out_dir, gap_s=GAP_S, n_yes=24, n_no=16, seed=0):
    vocab = EpicVocab(os.path.join(classes_dir, "EPIC_100_verb_classes.csv"),
                      os.path.join(classes_dir, "EPIC_100_noun_classes.csv"))
    store, truth, call_share = build_morning(clips, os.path.join(out_dir, "vlm_cache"), ann_csv, vocab, gap_s,
                                             db_path=os.path.join(out_dir, "morning.db"))
    span_min = (len(clips) - 1) * gap_s / 60 + 1
    print(f"one morning: {len(clips)} clips over {span_min:.0f} min, {len(truth)} true actions, "
          f"{len(store.activities())} log rows (VLM on {call_share:.0%} of windows)")
    qs = make_questions(truth, vocab, n_yes=n_yes, n_no=n_no, seed=seed)
    now = START + timedelta(seconds=len(clips) * gap_s + 600)
    res = {"mnemo": ask_mnemo(qs, store, describer, now)}
    frames, secs = morning_frames(clips)
    res["direct VLM (32 frames of the morning)"] = [
        dict(q, answer=a, yes=a.strip().lower().startswith("yes"))
        for q, a in ((q, describer.ask_video(frames, secs, q["q"])) for q in qs)]
    wq = when_questions(truth, vocab, [q for q in qs if q["expect"]])
    when = ask_mnemo([{"q": w["q"], "expect": True} for w in wq], store, describer, now)
    for w, a in zip(wq, when):
        w["answer"], w["correct"] = a["answer"], time_correct(a["answer"], w["times"])
        w["said_yes"] = says_yes(a["answer"])

    pct = lambda x: f"{x:.0%}" if x is not None else "-"
    lines = [f"## Experiment C: questions about a whole morning ({len(clips)} clips over {span_min:.0f} min, "
             f"{len(truth)} true actions)", "",
             "| System | 'Yes' questions answered yes | 'No' questions answered yes (hallucination) | Accuracy "
             "| 'When did I…?' right time (±90 s) |", "|---|---|---|---|---|"]
    for name, items in res.items():
        sc = score_answers(items)
        tcol = (f"{pct(sum(w['correct'] for w in wq) / len(wq) if wq else None)} "
                f"({sum(w['correct'] for w in wq)}/{len(wq)})") if name == "mnemo" else "cannot tell time"
        lines.append(f"| {name} | {pct(sc['recall'])} ({sc['yes_answered_yes']}/{sc['yes_questions']}) "
                     f"| {pct(sc['hallucination'])} ({sc['no_answered_yes']}/{sc['no_questions']}) "
                     f"| {pct(sc['accuracy'])} | {tcol} |")
    md = "\n".join(lines)
    with open(os.path.join(out_dir, "long_qa.json"), "w") as f:
        json.dump({"questions": res, "when": wq, "gap_s": gap_s, "clips": clips}, f, indent=1, default=str)
    with open(os.path.join(out_dir, "long_qa.md"), "w") as f:
        f.write(md + "\n")
    for i, q in enumerate(qs):
        marks = " ".join("Y" if res[s][i]["yes"] else "n" for s in res)
        print(f"   {'should be YES' if q['expect'] else 'should be NO '} | {marks} | {q['q']}")
    print("   (columns: " + " | ".join(res) + ")")
    for w in wq[:8]:
        print(f"   {'✓' if w['correct'] else '✗'} {w['q']} → {w['answer'][:90]}")
    print("\n" + md)
    return res, wq, md
