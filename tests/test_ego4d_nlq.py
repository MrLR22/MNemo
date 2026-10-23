# Ego4D NLQ test (evaluation/ego4d_nlq.py) and moment finding (query/locate.py).
# The VLM and the trigger are replaced by scripted answers; the video is a tiny synthetic file.

import json
from datetime import timedelta

import cv2
import numpy as np

from evaluation.ego4d_nlq import (fit, judge, load_nlq, logged, parse_seconds, run_nlq, select_clips,
                                  trigger_coverage)
from evaluation.benchmark import START
from query.locate import locate, search


def _nlq(path, clips):
    """clips: [(video_uid, clip_uid, seconds, [(query, template, slot, t0, t1)])]"""
    vids = {}
    for v, c, sec, qs in clips:
        vids.setdefault(v, []).append({
            "clip_uid": c, "clip_start_sec": 0, "clip_end_sec": sec,
            "annotations": [{"language_queries": [{"query": q, "template": t, "slot_x": s,
                                                   "clip_start_sec": t0, "clip_end_sec": t1} for q, t, s, t0, t1 in qs]}]})
    with open(path, "w") as f:
        json.dump({"videos": [{"video_uid": v, "clips": cs} for v, cs in vids.items()]}, f)
    return path


PUT = "Place: Where did I put X?"


def test_load_cleans_text_and_select_one_clip_per_video(tmp_path):
    p = _nlq(tmp_path / "nlq.json", [
        ("v1", "c1", 480, [("Query Text:Where did I put the drill?", PUT, "drill ", 10, 12), ("", PUT, "", 1, 2)]),
        ("v1", "c2", 480, [("Where is the cup?", "Objects: Where is object X?", "cup", 5, 6)]),
        ("v2", "c3", 1200, [("Who did I talk to?", "People: Who did I talk to in location X?", None, 5, 9)]),
        ("v3", "c4", 300, [("How many cups?", "Objects: How many X’s? (quantity question)", "cups", 1, 3)])])
    nlq = load_nlq(p)
    assert nlq["c1"]["questions"] == [{"q": "Where did I put the drill?", "template": PUT, "slot": "drill",
                                       "t0": 10.0, "t1": 12.0}]                      # empty question dropped
    sel = select_clips(nlq, 10, seed=0, max_seconds=600)
    assert "c3" not in sel and "c4" in sel and len(sel) == 2                       # 20-min clip skipped
    assert len({nlq[c]["video_uid"] for c in sel}) == len(sel)                      # one per video


def test_fit_groups():
    assert fit(PUT) == fit("Objects: What did I put in X?") == fit("Objects: State of an object") == "good"
    assert fit("Objects: Where is object X before / after event Y?") == "good"
    assert fit("Objects: Where is object X?") == fit("Objects: In what location did I see object X ?") == "partial"
    assert fit("Objects: How many X’s? (quantity question)") == fit("People: Who did I talk to in location X?") == "poor"
    assert fit("Objects: What X is Y?") == "poor" and fit(None) == "partial"


def _row(verb, obj, where, t):          # same fields as modules.vlm_logger.replay rows
    return {"verb": verb, "object": obj, "where": where, "place": "garage", "t": t, "t_end": t + 4,
            "ts_start": START + timedelta(seconds=t), "ts_end": START + timedelta(seconds=t + 4)}


ROWS = [_row("picks up", "drill", "table", 0.0), _row("puts", "drill", "shelf", 8.0),
        _row("opens", "door", None, 16.0), _row(None, None, None, 20.0)]


class Chat:
    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def chat(self, prompt, max_new_tokens=300):
        self.prompts.append(prompt)
        return self.reply


def test_locate_picks_line_or_falls_back():
    llm = Chat("2")
    p = locate("Where did I put the drill?", ROWS, llm)
    assert (p["t0"], p["t1"]) == (8.0, 12.0) and p["candidates"] == 2 and p["by"] == "model"
    assert "[1] 00:00.0-00:04.0 picks up drill" in llm.prompts[0]                    # candidates in time order
    p = locate("Where did I put the drill?", ROWS, Chat("0"))                        # not a line: best match
    assert p["by"] == "search" and p["t0"] == 8.0
    assert locate("Where is my passport?", ROWS, Chat("1")) is None                  # no candidate: no answer
    assert locate("Did I open the door?", ROWS)["t0"] == 16.0                        # no model: best match


def test_search_rare_words_count_more():
    rows = [_row("puts down", "cup", "table", t) for t in (0.0, 4.0, 8.0)] + [_row("picks up", "sieve", "sink", 12.0)]
    assert search(rows, "Where did I put the sieve?")[0]["t"] == 12.0                # "sieve" beats "put"
    assert [r["t"] for r in search(rows, "Where did I put the cup?")][:3] == [0.0, 4.0, 8.0]   # ties: earlier first


def test_judge_logged_and_seconds():
    q = {"q": "Where did I put the drill?", "slot": "drill", "template": PUT, "t0": 13.0, "t1": 14.0}
    j = judge((8.0, 12.0), q)
    assert j["hit2"] and j["hit10"] and j["iou"] == 0.0                               # 1 s away: hit, no overlap
    assert not judge((30.0, 34.0), q)["hit10"] and not judge(None, q)["hit2"]
    assert abs(judge((12.0, 14.0), q)["iou"] - 0.5) < 1e-9
    assert logged(ROWS, q) and not logged(ROWS, dict(q, slot="cup"))
    assert parse_seconds("At 1:05.") == 65 and parse_seconds("around 42.5 seconds") == 42.5
    assert parse_seconds("I can't tell") is None


def test_trigger_coverage_counts_moments():
    scores = np.array([0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1])
    qs = [{"t0": 1.0, "t1": 2.0}, {"t0": 13.0, "t1": 15.0}, {"t0": 25.0, "t1": 26.0}]
    cov = trigger_coverage(scores, qs)
    assert cov["every window"]["covered"] == [True, True, True]
    m = cov["mnemo@0.6"]
    assert m["calls"] == 3 and m["covered"] == [True, False, True]                   # window 0, heartbeat 5, 6
    assert cov["every-nth = mnemo@0.6"]["calls"] == 3


class Scripted:
    def __init__(self, answers, chat_reply="2", time_reply="9"):
        self.answers, self.chat_reply, self.time_reply = list(answers), chat_reply, time_reply

    def describe(self, frames, seconds, check=None):
        return self.answers.pop(0) if self.answers else "PLACE: garage"

    def chat(self, prompt, max_new_tokens=300):
        return self.chat_reply

    def ask_time(self, frames, seconds, question):
        return self.time_reply


class Gate:
    def signals(self, video):
        return {"hoi": np.array([0.9, 0.1, 0.9, 0.1, 0.1, 0.1]), "motion": np.zeros(6)}, \
               {"gate_seconds": 0.1, "frames": 240, "gate_real_time_factor": 0.01}


def _video(path, seconds=24, fps=10):
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (160, 120))
    for i in range(seconds * fps):
        w.write(np.full((120, 160, 3), i % 255, np.uint8))
    w.release()
    return path


def test_run_nlq_end_to_end(tmp_path):
    p = _nlq(tmp_path / "nlq.json", [
        ("v1", "c1", 24, [("Where did I put the drill?", PUT, "drill", 9.0, 10.0),
                          ("How many cups did I see?", "Objects: How many X’s? (quantity question)", "cups", 2.0, 3.0)]),
        ("v2", "c2", 24, [("Where did I put the drill?", PUT, "drill", 9.0, 10.0)])])
    nlq = load_nlq(p)
    vid = _video(str(tmp_path / "c1.avi"))
    answers = ["PLACE: garage\nDID: pick up | drill | table", "PLACE: garage\nDID: put | drill | shelf"] + \
              ["PLACE: garage"] * 4
    res, md = run_nlq([{"clip_uid": "c1", "path": vid}], nlq, Scripted(answers), Gate(), str(tmp_path / "out"),
                      trigger_only=[{"clip_uid": "c2", "path": vid}])
    r = res[0]
    every = r["systems"]["every window"]["answers"]
    assert every[0]["logged"] and every[0]["hit2"] and every[0]["pred"] == [4.0, 8.0]
    assert every[1]["pred"] is None and every[1]["fit"] == "poor"                    # nothing about cups
    trig = r["systems"]["mnemo@0.6"]
    assert trig["calls"] == 2                                                        # windows 0 and 2
    assert not trig["answers"][0]["logged"]                                          # window 1 (the put) skipped
    assert r["systems"]["direct VLM (no memory)"]["answers"][0]["hit2"]              # "9" → 7-11 s
    assert "systems" not in res[1] and res[1]["trigger"]["mnemo@0.6"]["covered"] == [True]
    assert "## 1. Trigger" in md and "| good | 1 |" in md and "Place: Where did I put X?" in md
    saved = json.load(open(tmp_path / "out" / "ego4d_nlq.json"))
    assert len(saved) == 2 and saved[0]["questions"][0]["slot"] == "drill"
    # second run reads the cache: same result without any VLM answers left
    res2, _ = run_nlq([{"clip_uid": "c1", "path": vid}], nlq, Scripted([]), Gate(), str(tmp_path / "out"))
    assert res2[0]["systems"]["every window"]["answers"][0]["pred"] == [4.0, 8.0]
