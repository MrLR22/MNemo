from datetime import datetime

import config as C
from modules.storage import EventStore
from modules.types import Event
from query.query_engine import QueryEngine
from tools.demo_video import load_labels

EPIC_HEADER = ("narration_id,participant_id,video_id,narration_timestamp,start_timestamp,stop_timestamp,"
               "start_frame,stop_frame,narration,verb,verb_class,noun,noun_class,all_nouns,all_noun_classes\n")


def test_query_answers_about_any_tracked_object():
    s = EventStore(":memory:")
    s.add(Event(datetime(2026, 10, 17, 9, 0), "plate", "putdown", "kitchen_counter", 0.8))
    ans = QueryEngine(s).ask("where is the plate?", now=datetime(2026, 10, 17, 9, 5))
    assert "kitchen counter" in ans and "plate" in ans


def test_set_objects_overrides_targets_and_prompts():
    old = (list(C.TARGET_OBJECTS), dict(C.YOLOWORLD_PROMPTS), dict(C.OBJECT_SYNONYMS))
    try:
        C.set_objects(["cup", "pill_bottle"])
        assert C.TARGET_OBJECTS == ["cup", "pill_bottle"]
        assert C.YOLOWORLD_PROMPTS == {"cup": "cup", "pill_bottle": "pill bottle"}   # keeps known wording
        assert "cups" in C.OBJECT_SYNONYMS["cup"]
    finally:
        C.TARGET_OBJECTS, C.YOLOWORLD_PROMPTS, C.OBJECT_SYNONYMS = old


def test_demo_labels_from_epic_annotations(tmp_path):
    p = tmp_path / "EPIC_100_train.csv"
    p.write_text(EPIC_HEADER +
                 "a,P26,P26_110,0,00:03:00.00,00:03:01.50,0,0,take lid,take,0,lid,1,[],[]\n"
                 "b,P26,P26_110,0,00:03:05.00,00:03:06.00,0,0,wash cup,wash,2,cup,2,[],[]\n"
                 "c,P26,P26_110,0,00:03:10.00,00:03:12.00,0,0,put lid,put,1,lid,1,[],[]\n"
                 "d,P26,P26_110,0,00:05:10.00,00:05:12.00,0,0,put lid,put,1,lid,1,[],[]\n")
    labels = load_labels(str(p), "P26_110_w178.mp4", duration=60)
    assert [(round(t, 1), a, o) for t, a, o in labels] == [(3.5, "pickup", "lid"), (14.0, "putdown", "lid")]
