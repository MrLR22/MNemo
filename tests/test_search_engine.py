# The search part (search/): vectors stored in the diary file, the Searcher over a time range, fingerprints written
# while a video is processed, and the rule that an image match alone never makes a "yes".
# The text, image and language models are replaced by tiny fakes.

from datetime import datetime, timedelta

import numpy as np

from modules.storage import EventStore
from modules.vlm_logger import ActionLogger
from query.llm_qa import answer
from search.engine import Searcher
from tests.test_ego4d_nlq import _video

T = datetime(2026, 11, 1, 9, 0)


class FakeText:
    model_id = "fake-text"
    GROUPS = [("cooker", "stove", "hob"), ("bike", "bicycle"), ("keys", "key")]

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += len(texts)
        v = np.array([[float(any(w in t.lower() for w in g)) for g in self.GROUPS] + [0.1] for t in texts])
        return v / np.linalg.norm(v, axis=1, keepdims=True)


class FakeClip:
    """Image vector = [brightness, 0.2]; every question asks for 'bright'."""
    model_id = "fake-clip"

    def embed(self, images):
        return np.array([[im.mean() / 255.0, 0.2] for im in images])

    def embed_text(self, texts):
        return np.array([[1.0, 0.0]])


def _act(store, sec, verb, obj, dur=4):
    store.add_activity({"ts_start": T + timedelta(seconds=sec), "ts_end": T + timedelta(seconds=sec + dur),
                        "verb": verb, "object": obj, "place": "kitchen"})


def _diary(tmp_path):
    store = EventStore(str(tmp_path / "diary.db"))
    _act(store, 0, "opens", "fridge")
    _act(store, 8, "turns on", "stove")
    _act(store, 16, "picks up", "cup")
    for k, bright in enumerate([0.1, 0.2, 0.3, 0.95, 0.2, 0.1]):         # window 3 (12-16 s) has no diary row
        store.add_fingerprint(T + timedelta(seconds=4 * k), T + timedelta(seconds=4 * k + 4), [bright, 0.2], "fake-clip")
    return store


def test_vectors_are_stored_and_cleared(tmp_path):
    store = _diary(tmp_path)
    starts, ends, fps = store.fingerprints("fake-clip", T + timedelta(seconds=5), T + timedelta(seconds=13))
    assert [s.second for s in starts] == [4, 8, 12] and fps.shape == (3, 2)
    assert store.fingerprints("other-model")[2].shape[0] == 0                   # models never mix
    store.add_text_vectors({"opens fridge": np.array([0.6, 0.8])}, "fake-text")
    assert np.allclose(store.text_vectors(["opens fridge", "x"], "fake-text")["opens fridge"], [0.6, 0.8], atol=1e-3)
    store.clear_activities()
    assert store.activities() == [] and store.fingerprints("fake-clip")[0] == []
    assert store.text_vectors(["opens fridge"], "fake-text") == {}


def test_meaning_vectors_computed_once_per_line(tmp_path):
    store, text = _diary(tmp_path), FakeText()
    s = Searcher(store, text=text)
    assert s.search("did I use the cooker?")[0]["object"] == "stove"           # keyword alone finds nothing
    first = text.calls
    s.search("did I use the hob?")
    assert text.calls == first + 1                                            # only the new question
    assert Searcher(store, text=FakeText()).text.embed(["turns on stove (place: kitchen)"]).shape == (1, 4)


def test_image_moment_with_clock_time_but_never_a_yes(tmp_path):
    store = _diary(tmp_path)
    s = Searcher(store, clip=FakeClip())
    found = s.search("where is the bright thing?")
    assert found[0]["image_only"] and found[0]["ts_start"] == T + timedelta(seconds=12)
    assert not any(r.get("image_only") for r in s.search("where is the bright thing?", diary_only=True))
    m = s.moment("where is the bright thing?")
    assert (m["ts_start"], m["ts_end"]) == (T + timedelta(seconds=12), T + timedelta(seconds=16))
    later = s.moment("where is the bright thing?", start=T + timedelta(seconds=16))   # time range respected
    assert later is None or later["ts_start"] >= T + timedelta(seconds=16)


class FakeLLM:
    def __init__(self):
        self.prompts = []

    def chat(self, prompt, max_new_tokens=300):
        self.prompts.append(prompt)
        return "Yes, at 9:00."


def test_answer_uses_the_search_but_only_diary_lines(tmp_path):
    store, llm = _diary(tmp_path), FakeLLM()
    s = Searcher(store, text=FakeText(), clip=FakeClip())
    r = answer("did I use the cooker?", store, llm=llm, now=T + timedelta(minutes=5), searcher=s)
    assert "stove" in llm.prompts[-1] and "not described" not in llm.prompts[-1]
    assert all(x.get("verb") for x in r["lines"])


class Scripted:
    def describe(self, frames, seconds, check=None):
        return "PLACE: garage\nDID: picks up | drill | -"


def test_logger_writes_fingerprints_into_the_diary(tmp_path):
    store = EventStore(str(tmp_path / "d.db"))
    vid = _video(str(tmp_path / "v.avi"), seconds=12)
    ActionLogger(Scripted(), store, embedder=FakeClip()).run(vid, start_time=T, verbose=False,
                                                              call=np.array([True, False, True]))
    starts, ends, fps = store.fingerprints("fake-clip")
    assert len(starts) == 3 and starts[1] == T + timedelta(seconds=4)            # skipped window too
    assert len(store.activities()) == 2


def test_search_bench_on_a_saved_run(tmp_path):
    from evaluation.search_bench import load_clips, report, run
    from tests.test_ego4d_nlq import PUT, Gate, Scripted as Script, _nlq
    from evaluation.ego4d_nlq import load_nlq, run_nlq
    p = _nlq(tmp_path / "nlq.json", [("v1", "c1", 24, [("Where did I put the drill?", PUT, "drill", 9.0, 10.0)])])
    vid = _video(str(tmp_path / "c1.avi"))
    answers = ["PLACE: garage\nDID: pick up | drill | table", "PLACE: garage\nDID: put | drill | shelf"] + ["PLACE: garage"] * 4
    run_nlq([{"clip_uid": "c1", "path": vid}], load_nlq(p), Script(answers), Gate(), str(tmp_path / "out"),
            embedder=FakeClip(), direct=False)
    clips = load_clips([str(tmp_path / "out")])
    assert len(clips) == 1 and clips[0]["fps"].shape == (6, 2) and clips[0]["prompt"] == "lines-v3"
    md = report(run(clips, text=FakeText(), clip=FakeClip()))
    assert "| every window | keyword | 1 (100%)" in md and "mnemo@0.6, keyword + meaning + image" in md
