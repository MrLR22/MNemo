# The SEEN line (prompt lines-v4), its storage, and the hybrid diary search (keyword + meaning + image).
# The text and image models are replaced by tiny fake embedders.

import sqlite3
from datetime import datetime, timedelta

import numpy as np

from evaluation.ego4d_nlq import FULL, load_nlq, run_nlq
from modules.storage import EventStore
from modules.vlm_logger import build_prompt, loop_or_full, parse_output, replay
from query.locate import diary_line, hybrid, locate
from tests.test_ego4d_nlq import PUT, Gate, Scripted, _nlq, _video

T = datetime(2026, 10, 30, 9, 0)


def test_seen_line_parsed_cleaned_and_capped():
    p = parse_output("PLACE: garage\nSEEN: Bicycle, wrench; red toolbox., none, bicycle, a, b, c, d\nDID: picks up | wrench | -")
    assert p["seen"] == ["bicycle", "wrench", "red toolbox", "a", "b"]               # at most VLM_MAX_SEEN (5)
    assert p["actions"] == [{"verb": "picks up", "object": "wrench", "where": None}]
    assert parse_output("SEEN: -")["seen"] == []


def test_did_lines_without_separators():
    acts = lambda t: [(a["verb"], a["object"], a["where"]) for a in parse_output(t)["actions"]]
    assert acts("DID: picks up frying pan -> kitchen sink") == [("picks up", "frying pan", "kitchen sink")]
    assert acts("DID: picks up bottle -> fridge, pours bottle -> pot") == [("picks up", "bottle", "fridge"),
                                                                          ("pours", "bottle", "pot")]
    assert acts("DID: picks up plates, cups -> sink") == [("picks up", "plates, cups", "sink")]
    assert acts("DID: holds phone -> table") == [("holds", "phone", "table")]
    assert acts("DID: opens fridge") == [("opens", "fridge", None)]
    assert acts("DID: picks up | cup | ->") == [("picks up", "cup", None)]


def test_seen_is_last_and_stops_generation():
    p = build_prompt(8, 4.0, True, ["fridge"])
    lines = p.splitlines()
    assert lines.index("SEEN: <objects in view, comma-separated>") > lines.index("DID: <verb> | <object> | <where the object is or ends up, or ->")
    assert p.index("CHECK:") < p.index("SEEN line (always the last line)")       # answered before the stop
    assert loop_or_full("PLACE: kitchen\nDID: opens | fridge | -\nSEEN: cup\n")
    assert not loop_or_full("PLACE: kitchen\nDID: opens | fridge | -\nSEEN: cup")    # line not finished yet


def _win(k, text):
    return {"k": k, "t": 4.0 * k, "ts": T + timedelta(seconds=4 * k), "seconds": 4.0, "text": text, "called": True}


def test_seen_rows_merge_and_are_stored(tmp_path):
    ws = [_win(0, "PLACE: garage\nSEEN: bicycle, helmet"),
          _win(1, "PLACE: garage\nSEEN: wrench\nDID: picks up | wrench | -"),
          _win(2, "PLACE: garage\nSEEN: wrench, toolbox\nDID: picks up | wrench | -")]
    store = EventStore(str(tmp_path / "d.db"))
    rows = replay(ws, store=store)
    assert rows[0]["verb"] is None and rows[0]["seen"] == "bicycle, helmet"         # nothing done, still findable
    assert len(rows) == 2 and rows[1]["seen"] == "wrench, toolbox" and rows[1]["t_end"] == 12.0
    assert [r["seen"] for r in store.activities()] == ["bicycle, helmet", "wrench, toolbox"]
    assert "saw: bicycle, helmet" in diary_line(rows[0])


def test_old_diary_gets_the_seen_column(tmp_path):
    db = str(tmp_path / "old.db")
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE activities (id INTEGER PRIMARY KEY AUTOINCREMENT, ts_start TEXT NOT NULL, "
                "ts_end TEXT NOT NULL, verb TEXT, object TEXT, place TEXT, where_ TEXT, raw TEXT, source TEXT)")
    con.execute("INSERT INTO activities (ts_start, ts_end, verb) VALUES ('2026-10-01T09:00:00', '2026-10-01T09:00:04', 'opens')")
    con.commit(); con.close()
    store = EventStore(db)
    assert store.activities()[0]["seen"] is None
    store.add_activity({"ts_start": T, "ts_end": T, "verb": None, "seen": "bicycle"})
    assert store.activities()[1]["seen"] == "bicycle"


class FakeText:
    """Meaning = shared letters of a fixed vocabulary: 'cooker' is close to 'stove'."""
    GROUPS = [("cooker", "stove", "hob"), ("bike", "bicycle"), ("keys", "key")]

    def embed(self, texts):
        v = np.array([[float(any(w in t.lower() for w in g)) for g in self.GROUPS] + [0.1] for t in texts])
        return v / np.linalg.norm(v, axis=1, keepdims=True)


class FakeClip:
    def embed_text(self, texts):
        return np.array([[1.0, 0.0]])


def _row(t, verb, obj, seen=None):
    return {"t": t, "t_end": t + 4, "verb": verb, "object": obj, "where": None, "place": None, "seen": seen}


def test_meaning_search_finds_synonyms_keyword_cannot():
    rows = [_row(0, "opens", "fridge"), _row(4, "turns on", "stove"), _row(8, "picks up", "cup")]
    assert locate("Did I use the cooker?", rows) is None                             # no shared word
    hit = locate("Did I use the cooker?", rows, candidates=hybrid(text=FakeText()))
    assert hit["row"]["object"] == "stove"


def test_image_search_finds_a_window_the_trigger_skipped():
    rows = [_row(0, "picks up", "cup")]                                              # windows 1-3 not described
    fps = np.array([[0.1, 1.0], [0.2, 1.0], [0.99, 0.1], [0.3, 1.0]])
    fps /= np.linalg.norm(fps, axis=1, keepdims=True)
    found = hybrid(images=(fps, FakeClip()))(rows, "where is my bicycle?", 3)
    assert found[0]["image_only"] and found[0]["t"] == 8.0                           # window 2: best image match
    hit = locate("where is my bicycle?", rows, candidates=hybrid(images=(fps, FakeClip())))
    assert (hit["t0"], hit["t1"]) == (8.0, 12.0)
    assert "not described" in diary_line(found[0])


class FakeEmbedder(FakeClip):
    def embed(self, images):
        return np.array([[im.mean() / 255.0, 0.5] for im in images])


def test_run_nlq_with_all_search_parts(tmp_path):
    p = _nlq(tmp_path / "nlq.json", [("v1", "c1", 24, [("Where did I put the drill?", PUT, "drill", 9.0, 10.0),
                                                       ("Where is the bicycle?", "Objects: Where is object X?", "bicycle", 1.0, 2.0)])])
    vid = _video(str(tmp_path / "c1.avi"))
    answers = ["PLACE: garage\nSEEN: bicycle\nDID: pick up | drill | table", "PLACE: garage\nDID: put | drill | shelf"] + \
              ["PLACE: garage"] * 4
    res, md = run_nlq([{"clip_uid": "c1", "path": vid}], load_nlq(p), Scripted(answers), Gate(), str(tmp_path / "out"),
                      embedder=FakeEmbedder(), text=FakeText())
    s = res[0]["systems"]
    assert {"mnemo@0.6", "mnemo@0.6 +seen", "mnemo@0.6 +seen +meaning", "mnemo@0.6 +seen +image", "mnemo@0.6 +meaning +image", "mnemo@0.6" + FULL,
            "every window", "every window" + FULL, "every-nth = mnemo@0.6" + FULL} <= set(s)
    assert "every window +seen" not in s
    assert not s["mnemo@0.6"]["answers"][1]["logged"] and s["mnemo@0.6 +seen"]["answers"][1]["logged"]
    assert (tmp_path / "out" / "vlm_cache" / "c1__fingerprints.npy").exists()
    assert "## 5. What each part adds" in md and "Image fingerprints" in md
    assert "Hit ±2 s (mnemo@0.6 +seen +meaning +image)" in md
