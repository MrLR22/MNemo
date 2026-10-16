# Stage 1 (VLM action log) and stage 2 (answers from the log).
# The real 4B model only runs on a GPU (Colab); here the logic is tested with a fake describer,
# and the real VLMDescriber code path with a tiny randomly-initialised Qwen3-VL.

import os
from datetime import datetime, timedelta

import cv2
import numpy as np
import pytest

from modules.storage import EventStore
from modules.vlm_logger import ActionLogger, parse_output
from query.llm_qa import answer, norm_tokens, retrieve, time_range

NOW = datetime(2026, 10, 17, 21, 0)


def test_parse_output_variants():
    good = parse_output('```json\n{"place": "kitchen", "actions": [{"verb": "Closes", "object": "Fridge door", "where": ""}]}\n```')
    assert good["place"] == "kitchen" and good["actions"] == [{"verb": "closes", "object": "fridge door", "where": None}]
    assert parse_output('{"place": "", "actions": []}') == {"place": None, "actions": [], "raw": '{"place": "", "actions": []}'}
    assert parse_output("I cannot tell.")["actions"] == []
    assert parse_output('{"place": "desk", "actions": [{"object": "pen"}]}')["actions"] == []   # no verb → dropped
    assert parse_output('{"place": "desk", "actions": [ broken json')["actions"] == []


class FakeDescriber:
    """Returns scripted answers, one per window."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def describe(self, frames, seconds):
        self.calls.append((len(frames), round(seconds, 2)))
        return self.answers.pop(0) if self.answers else '{"place": null, "actions": []}'


def _video(path, seconds=16, fps=30):
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (320, 240))
    for i in range(seconds * fps):
        w.write(np.full((240, 320, 3), i % 255, np.uint8))
    w.release()
    return path


def test_logger_windows_rows_and_merging(tmp_path):
    vid = _video(str(tmp_path / "v.avi"))
    fridge = '{"place": "kitchen", "actions": [{"verb": "opens", "object": "fridge door", "where": null}]}'
    d = FakeDescriber([
        fridge,
        fridge,                                                                     # continues → merged
        '{"place": "kitchen", "actions": [{"verb": "closes", "object": "fridge door"}, {"verb": "puts down", "object": "milk", "where": "counter"}]}',
        '{"place": "kitchen", "actions": []}',
    ])
    store = EventStore(":memory:")
    rows, stats = ActionLogger(d, store).run(vid, start_time=NOW - timedelta(minutes=5), verbose=False)
    assert d.calls == [(8, 4.0)] * 4                                   # 16 s → 4 windows of 8 frames
    got = [(r["verb"], r["object"]) for r in store.activities()]
    assert got == [("opens", "fridge door"), ("closes", "fridge door"), ("puts down", "milk"), (None, None)]
    first = store.activities()[0]
    assert (first["ts_end"] - first["ts_start"]).total_seconds() == 8.0   # two windows merged
    assert stats["windows"] == 4 and stats["video_seconds"] == 16.0 and stats["real_time_factor"] is not None


def _store_with_log():
    s = EventStore(":memory:")
    t = NOW.replace(hour=9, minute=14)
    for verb, obj, where, place, dt in [
        ("opens", "fridge door", None, "kitchen", 0),
        ("takes", "milk bottle", None, "kitchen", 4),
        ("closes", "fridge door", None, "kitchen", 8),
        ("puts down", "keys", "on the shoe rack", "hallway", 3600),
        ("swallows", "tablet", None, "bedroom", 3 * 3600),
    ]:
        s.add_activity({"ts_start": t + timedelta(seconds=dt), "ts_end": t + timedelta(seconds=dt + 4),
                        "verb": verb, "object": obj, "where": where, "place": place, "raw": ""})
    return s


def test_synonyms_and_retrieval():
    assert {"close", "door", "refrigerator"} <= norm_tokens("Did I shut the door of the fridge?")
    rows = _store_with_log().activities()
    hits = retrieve(rows, "did I close the fridge door?")
    assert hits[0]["verb"] == "closes" and {r["object"] for r in hits[:2]} == {"fridge door"}
    assert retrieve(rows, "where did I keep my keys")[0]["object"] == "keys"
    assert retrieve(rows, "did I take my medicine")[0]["object"] == "tablet"


def test_time_ranges():
    assert time_range("what did I do yesterday", NOW)[2] == "yesterday"
    s, e, p = time_range("what did I do in the last 2 hours", NOW)
    assert p == "in the last 2 hours" and e - s == timedelta(hours=2)


def test_answer_without_llm_lists_records_and_never_invents():
    s = _store_with_log()
    r = answer("did I close the fridge?", s, now=NOW)
    assert r["mode"] == "search" and "close" in r["answer"] and "9:14 AM" in r["answer"]
    r = answer("did I water the plants?", s, now=NOW)
    assert r["answer"].startswith("I didn't see that")


def test_answer_with_llm_gets_only_matching_lines():
    class FakeLLM:
        prompt = None

        def chat(self, prompt):
            FakeLLM.prompt = prompt
            return "Yes, you closed the fridge door at 9:14 AM in the kitchen."

    r = answer("did I shut the fridge?", _store_with_log(), llm=FakeLLM(), now=NOW)
    assert r["mode"] == "llm" and "closed the fridge" in r["answer"]
    assert "closes fridge door" in FakeLLM.prompt and "keys" not in FakeLLM.prompt
    assert "Do not guess" in FakeLLM.prompt


def _tiny_qwen3vl():
    """Tiny random-weight Qwen3-VL from the cached real config + processor (no network)."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
        proc = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
        cfg = AutoConfig.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
    except Exception as e:                                         # not cached / not installed
        pytest.skip(f"Qwen3-VL processor not cached: {e}")
    tc, vc = cfg.text_config, cfg.vision_config
    tc.num_hidden_layers, tc.hidden_size, tc.intermediate_size = 2, 128, 256
    tc.num_attention_heads, tc.num_key_value_heads, tc.head_dim = 4, 2, 32
    for attr in ("rope_scaling", "rope_parameters"):
        rp = getattr(tc, attr, None)
        if rp and "mrope_section" in rp:
            rp["mrope_section"] = [4, 6, 6]
    vc.depth, vc.hidden_size, vc.intermediate_size, vc.num_heads, vc.out_hidden_size = 2, 64, 128, 2, 128
    vc.deepstack_visual_indexes = [0]
    return AutoModelForImageTextToText.from_config(cfg), proc


def test_real_vlm_code_path_with_tiny_qwen3vl(tmp_path):
    from modules.vlm_logger import VLMDescriber
    model, proc = _tiny_qwen3vl()
    d = VLMDescriber(model=model, processor=proc)
    frames = [np.full((360, 640, 3), 40 * i, np.uint8) for i in range(7)]       # odd count → padded to 8
    assert isinstance(d.describe(frames, seconds=3.5), str)
    assert isinstance(d.chat("Say hello."), str)
    rows, stats = ActionLogger(d, EventStore(":memory:")).run(_video(str(tmp_path / "v.avi"), seconds=8),
                                                              verbose=False)
    assert stats["windows"] == 2                     # random weights → no parsable actions, but it runs
