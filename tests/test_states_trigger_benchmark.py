# Object-state tracking, the interaction trigger in stage 1, log scoring and question generation.
# The VLM is replaced by scripted answers; the scoring uses tiny EPIC-style class files.

import json
from datetime import datetime

import cv2
import numpy as np

from evaluation.log_accuracy import EpicVocab, score_log
from evaluation.qa_benchmark import make_questions, says_yes, score_answers
from modules.hoi_gate import gate_decisions, window_hoi_scores
from modules.storage import EventStore
from modules.vlm_logger import ActionLogger, StateTracker, canon_state, parse_output, replay

T = datetime(2026, 10, 16, 9, 0)


def test_canonical_states():
    assert canon_state("Fridge door", "Opened") == ("fridge", "open")
    assert canon_state("refrigerator", "shut") == ("fridge", "closed")
    assert canon_state("freezer drawer", "open") == ("freezer", "open")        # most specific first
    assert canon_state("kitchen tap", "running") == ("tap", "on")
    assert canon_state("door", "on") is None                                    # wrong kind of state
    assert canon_state("banana", "open") is None


def test_state_tracker_derives_changes_only():
    tr = StateTracker()
    assert tr.update([{"object": "fridge", "state": "open"}], T) == []          # first sighting: unknown before
    assert tr.update([{"object": "fridge door", "state": "open"}], T) == []
    d = tr.update([{"object": "Refrigerator", "state": "closed"}, {"object": "light", "state": "off"}], T)
    assert [(x["verb"], x["object"]) for x in d] == [("closes", "fridge")]
    assert tr.current()["fridge"][0] == "closed"


def test_parse_states_and_cut_off_answer():
    p = parse_output('{"place": "kitchen", "actions": [], "states": [{"object": "Tap", "state": "On"}]}')
    assert p["states"] == [{"object": "tap", "state": "on"}]
    cut = ('{"place": "kitchen", "actions": [{"verb": "opens", "object": "fridge", "where": null}, '
           '{"verb": "picks up", "object": "milk", "where": null}], "states": [{"object": "fridge", "sta')
    p = parse_output(cut)
    assert p["place"] == "kitchen" and [a["verb"] for a in p["actions"]] == ["opens", "picks up"]


class Scripted:
    def __init__(self, answers):
        self.answers, self.calls = list(answers), 0

    def describe(self, frames, seconds, check=None):
        self.calls += 1
        self.checks = getattr(self, "checks", []) + [check]
        return self.answers.pop(0)


def _video(path, seconds=24, fps=20):
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (160, 120))
    for i in range(seconds * fps):
        w.write(np.full((120, 160, 3), i % 255, np.uint8))
    w.release()
    return path


def _answers():
    st = lambda s: json.dumps({"place": "kitchen", "actions": [], "states": [{"object": "fridge door", "state": s}]})
    return [st("closed"),
            json.dumps({"place": "kitchen", "actions": [{"verb": "opens", "object": "fridge door", "where": None}],
                        "states": [{"object": "fridge", "state": "open"}]}),   # VLM saw it: no duplicate row
            st("open"), st("open"), st("closed"),                               # closed, never written as action
            json.dumps({"place": "kitchen", "actions": [{"verb": "picks up", "object": "milk", "where": None}]})]


def test_logger_states_no_duplicates_and_gated_equals_replay(tmp_path):
    vid = _video(str(tmp_path / "v.avi"))
    store = EventStore(":memory:")
    lg = ActionLogger(Scripted(_answers()), store, track_states=True)
    rows, stats = lg.run(vid, start_time=T, verbose=False)
    got = [(r["verb"], r["object"], r["source"]) for r in store.activities() if r["verb"]]
    assert got == [("opens", "fridge door", "vlm"), ("closes", "fridge", "state"), ("picks up", "milk", "vlm")]
    assert stats["vlm_calls"] == 6 == stats["windows"]
    # once the fridge is believed open, every next prompt asks for its state (state check)
    assert lg.describer.checks == [None, None, ["fridge"], ["fridge"], ["fridge"], None]

    # without the tracker the close is lost
    assert ("closes", "fridge") not in [(r["verb"], r["object"]) for r in replay(lg.windows, track_states=False)]

    # gated run == replay of the full run with the same mask (so the benchmark can replay)
    call = np.array([1, 1, 0, 0, 1, 1], bool)
    lg2 = ActionLogger(Scripted([a for a, c in zip(_answers(), call) if c]), None, track_states=True)
    rows2, st2 = lg2.run(vid, start_time=T, verbose=False, call=call)
    assert st2["vlm_calls"] == 4 and st2["windows"] == 6
    rep = replay(lg.windows, call, track_states=True)
    key = lambda rs: [(r["verb"], r["object"], r["t"], r["t_end"]) for r in rs]
    assert key(rows2) == key(rep)
    assert ("closes", "fridge") in [(r["verb"], r["object"]) for r in rep]   # open seen at 4 s, closed at 16 s


def test_trigger_scores_and_heartbeat():
    times = np.arange(0, 12, 0.1)
    probs = np.tile([0.95, 0.03, 0.02], (len(times), 1))
    probs[45] = [0.2, 0.7, 0.1]                                  # pick-up at 4.5 s → window 1
    s = window_hoi_scores(times, probs, 4.0)
    assert len(s) == 3 and s[1] > 0.7 and s[0] < 0.1
    sc = np.array([0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1])
    assert gate_decisions(sc, 0.5, heartbeat_windows=3, hangover=0).tolist() == \
        [True, False, False, True, False, False, True, False]
    # keep-going: the window after a triggered one is described too (heartbeat counts from there)
    assert gate_decisions(sc, 0.5, heartbeat_windows=3, hangover=1).tolist() == \
        [True, True, False, False, True, False, True, True]
    # hand motion triggers a window HOI-TCN scored low (e.g. stirring)
    motion = np.array([0, 0, 0, 0, 0, 2.0, 0, 0])
    assert gate_decisions(sc, 0.5, heartbeat_windows=99, hangover=0, motion=motion, motion_threshold=1.0).tolist() == \
        [True, False, False, False, False, True, True, False]


def test_tune_gate_prefers_fewest_calls_at_target():
    from evaluation.tune_gate import tune
    # 20 windows; actions only where the HOI score is high or where hands move (window 12)
    hoi = np.full(20, 0.05)
    hoi[[3, 4, 9]] = 0.9
    motion = np.zeros(20)
    motion[12] = 3.0
    acts = [(12.5, 14.0, "pick"), (16.5, 18.0, "put"), (36.5, 37.0, "open"), (49.0, 50.0, "stir")]
    best, rows = tune([{"hoi": hoi, "motion": motion}], [acts], target=1.0, window_s=4.0)
    assert best["coverage"] == 1.0 and best["motion_pct"] is not None      # stirring needs the motion rule
    assert best["calls"] == min(r["calls"] for r in rows if r["coverage"] >= 1.0)


def _vocab(tmp_path):
    (tmp_path / "v.csv").write_text('id,key,instances,category\n0,take,"[\'pick-up\', \'take\', \'grab\']",x\n'
                                    '1,put,"[\'put-down\', \'put\', \'place\']",x\n3,open,"[\'open\']",x\n'
                                    '4,close,"[\'close\', \'shut\']",x\n')
    (tmp_path / "n.csv").write_text('id,key,instances,category\n0,fridge,"[\'fridge\', \'refrigerator\', \'door:fridge\']",x\n'
                                    '1,bag,"[\'bag\', \'bag:plastic\']",x\n2,oven,"[\'oven\']",x\n'
                                    '3,phone,"[\'phone\']",x\n4,orange,"[\'orange\']",x\n')
    return EpicVocab(str(tmp_path / "v.csv"), str(tmp_path / "n.csv"))


def test_log_scoring(tmp_path):
    v = _vocab(tmp_path)
    assert v.verb_class("picks up") == 0 and v.verb_class("closes") == 4 and v.verb_class("shuts") == 4
    assert v.noun_classes("refrigerator door") == {0} and v.noun_classes("orange plastic bag") == {1}
    truth = [{"start": 1.0, "stop": 2.0, "narration": "open fridge", "verb_class": 3, "noun_class": 0, "nouns": {0}},
             {"start": 9.0, "stop": 10.0, "narration": "close fridge", "verb_class": 4, "noun_class": 0, "nouns": {0}},
             {"start": 20.0, "stop": 21.0, "narration": "take bag", "verb_class": 0, "noun_class": 1, "nouns": {1}}]
    rows = [{"t": 0.0, "t_end": 4.0, "verb": "opens", "object": "fridge door"},
            {"t": 8.0, "t_end": 12.0, "verb": "closes", "object": "refrigerator"},
            {"t": 40.0, "t_end": 44.0, "verb": "picks up", "object": "plastic bag"}]    # too late
    sc = score_log(rows, truth, v)
    assert sc["found"] == 2 and abs(sc["recall"] - 2 / 3) < 1e-9 and sc["state_found"] == 2 == sc["state_true"]
    assert abs(sc["precision"] - 2 / 3) < 1e-9 and [r["object"] for r in sc["unmatched"]] == ["plastic bag"]


def test_questions_and_answer_reading(tmp_path):
    v = _vocab(tmp_path)
    truth = [{"start": 1.0, "stop": 2.0, "narration": "open fridge", "verb_class": 3, "noun_class": 0, "nouns": {0}},
             {"start": 5.0, "stop": 6.0, "narration": "pick up phone", "verb_class": 0, "noun_class": 3, "nouns": {3}}]
    qs = make_questions(truth, v, n_yes=4, n_no=100)                       # all eligible no-questions
    yes = [q["q"] for q in qs if q["expect"]]
    no = [q["q"] for q in qs if not q["expect"]]
    assert yes == ["Did I open the fridge?", "Did I pick up the phone?"]
    assert not any("fridge" in q or "phone" in q for q in no) and "Did I open the oven?" in no
    assert says_yes("Yes, at 9:00 AM in the kitchen.") and not says_yes("I didn't see that in my records.")
    assert not says_yes("No, the log shows nothing about that.")
    sc = score_answers([{"expect": True, "yes": True}, {"expect": False, "yes": True}, {"expect": False, "yes": False}])
    assert sc["recall"] == 1.0 and sc["hallucination"] == 0.5


def test_v2_line_format():
    p = parse_output("PLACE: kitchen counter\nDID: picks up | plate | cupboard\nDID: puts down | Plate | -\n"
                     "STATE: oven = open\nSTATE: Fridge door: closed\nsome extra words")
    assert p["place"] == "kitchen counter"
    assert p["actions"] == [{"verb": "picks up", "object": "plate", "where": "cupboard"},
                            {"verb": "puts down", "object": "plate", "where": None}]
    assert p["states"] == [{"object": "oven", "state": "open"}, {"object": "fridge door", "state": "closed"}]
    assert parse_output("PLACE: hallway")["actions"] == []                       # nothing done
    arrows = parse_output("DID: opens | oven | ->\nDID: picks up | plate | -> cupboard")["actions"]
    assert [a["where"] for a in arrows] == [None, "cupboard"]
    cut = parse_output("PLACE: kitchen\nDID: opens | drawer | kitch")            # cut off mid-line: kept
    assert cut["actions"][0]["verb"] == "opens"
    assert parse_output('{"place": "desk", "actions": [{"verb": "types", "object": "laptop"}]}')["actions"][0]["verb"] == "types"


def test_v2_prompt_lists_fixed_verbs_and_states():
    from modules.vlm_logger import PROMPT_LINES, PROMPT_LINES_STATES
    p = PROMPT_LINES_STATES.format(n=8, sec=4)
    assert "STATE: <object>" in p and "picks up, puts down, opens, closes" in p and "Ignore what other people" in p
    assert "STATE" not in PROMPT_LINES.format(n=8, sec=4)


def test_loop_stop_dedupe_and_state_check_prompt():
    from modules.vlm_logger import build_prompt, loop_or_full
    loop = "PLACE: k\nDID: picks up | cup | -\nDID: puts down | cup | -\nDID: picks up | cup | -\n"
    assert loop_or_full(loop) and not loop_or_full(loop.rsplit("DID", 1)[0])
    assert loop_or_full("".join(f"DID: picks up | item{i} | -\n" for i in range(8)))
    p = parse_output(loop + "DID: puts down | cup | -")
    assert [(a["verb"], a["object"]) for a in p["actions"]] == [("picks up", "cup"), ("puts down", "cup")]
    pr = build_prompt(8, 4, True, ["fridge", "oven (on or off)"])
    assert "CHECK" in pr and "fridge, oven (on or off)" in pr and pr.index("STATE:") < pr.index("DID:")
    assert "CHECK" not in build_prompt(8, 4, True, None)


def test_state_check_list_and_appliance_power():
    tr = StateTracker()
    tr.update([{"object": "fridge", "state": "open"}, {"object": "oven", "state": "on"},
               {"object": "tap", "state": "off"}], T)
    assert tr.to_check(T) == ["fridge", "oven (on or off)"]                 # only open/on things
    d = tr.update([{"object": "oven", "state": "off"}, {"object": "fridge", "state": "closed"}], T)
    assert sorted((x["verb"], x["object"]) for x in d) == [("closes", "fridge"), ("turns off", "oven")]
    assert tr.to_check(T) == []
    from datetime import timedelta
    tr.update([{"object": "drawer", "state": "open"}], T)
    assert tr.to_check(T + timedelta(minutes=30)) == []                     # too old: stop asking


def test_long_qa_time_reading():
    from evaluation.benchmark import START
    from evaluation.long_qa import time_correct
    assert time_correct("You opened the fridge at 9:07 AM.", [7 * 60 + 20])           # minutes only
    assert time_correct("at 09:30:05", [30 * 60]) and not time_correct("at 10:00", [5])
    assert time_correct("at 1:05 PM", [(13 - 9) * 3600 + 5 * 60]) and not time_correct("I didn't see that", [0])
    assert START.hour == 9


def test_exact_clip_start_from_stored_events():
    from model.train_interaction import act_labels, exact_start
    v = {"name": "P01__P01_01_w10", "gt_end": np.array([2.0, 7.5]), "times": np.arange(0, 20, 0.1)}
    acts = [(11.0, 12.6), (16.0, 18.1), (25.0, 26.0)]                # true start = 10.6
    t0 = exact_start(v, acts)
    assert abs(t0 - 10.6) < 1e-6
    y = act_labels(v, acts, t0)
    assert y[int(1.0 / 0.1)] == 1 and y[int(4.0 / 0.1)] == 0
