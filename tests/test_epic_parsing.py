import argparse
import json

import numpy as np
import pytest

from model.data import group_of, label_segments, split_videos
from model.extract_features import (ego4d_verb_action, load_charades_segments, load_ego4d_segments,
                                    load_epic_segments)

# ── EPIC-KITCHENS-100 ──
ANN = """narration_id,participant_id,video_id,narration_timestamp,start_timestamp,stop_timestamp,start_frame,stop_frame,narration,verb,verb_class,noun,noun_class,all_nouns,all_noun_classes
P01_01_0,P01,P01_01,00:00:01.00,00:00:02.00,00:00:03.50,120,210,take plate,take,0,plate,2,['plate'],[2]
P01_01_1,P01,P01_01,00:00:05.00,00:00:05.20,00:00:06.00,312,360,put plate,put,1,plate,2,['plate'],[2]
P01_01_2,P01,P01_01,00:00:07.00,00:00:07.00,00:00:09.00,420,540,wash cup,wash,2,cup,13,['cup'],[13]
P02_01_0,P02,P02_01,00:00:01.00,00:00:01.00,00:00:02.00,60,120,pick up knife,pick-up,0,knife,4,['knife'],[4]
"""
VERBS = """id,key,instances,category
0,take,"['take', 'pick-up', 'grab']",retrieve
1,put,"['put', 'place']",leave
2,wash,"['wash']",clean
"""


def _args(tmp_path, participants=None):
    (tmp_path / "ann.csv").write_text(ANN)
    (tmp_path / "verbs.csv").write_text(VERBS)
    return argparse.Namespace(annotations=str(tmp_path / "ann.csv"), verb_classes=str(tmp_path / "verbs.csv"),
                              participants=participants)


def test_take_put_by_verb_class_not_raw_verb(tmp_path):
    segs = load_epic_segments(_args(tmp_path))
    # "pick-up" narration maps to class 0 = take → pickup; "wash" is dropped
    assert segs["P02_01"] == [(1.0, 2.0, "pickup", "knife")]
    assert [s[2] for s in segs["P01_01"]] == ["pickup", "putdown"]


def test_participant_filter(tmp_path):
    assert set(load_epic_segments(_args(tmp_path, ["P01"]))) == {"P01_01"}


def test_missing_take_put_classes_is_an_error(tmp_path):
    a = _args(tmp_path)
    (tmp_path / "verbs.csv").write_text("id,key,instances,category\n0,wash,[],clean\n")
    with pytest.raises(ValueError):
        load_epic_segments(a)


def test_segment_labels_cover_end_of_action():
    times = np.arange(0, 10, 0.1)
    lab, gt = label_segments(times, [(2.0, 3.5, "pickup", "plate"), (5.2, 6.0, "putdown", "plate")])
    assert set(lab[(times > 2.55) & (times < 4.15)]) == {1}       # last 1 s of segment + 0.7 s after
    assert set(lab[(times > 0.0) & (times < 2.45)]) == {0}
    assert set(lab[(times > 5.25) & (times < 6.65)]) == {2}
    assert list(gt["gt_action"]) == ["pickup", "putdown"]


# ── Ego4D FHO ──
def test_ego4d_verb_mapping():
    assert ego4d_verb_action("take_(pick,_grab,_get)") == "pickup"
    assert ego4d_verb_action("put_(place,_leave,_drop)") == "putdown"
    assert ego4d_verb_action("put_on_(wear)") is None
    assert ego4d_verb_action("take_off_(remove)") is None
    assert ego4d_verb_action("cut_(trim,_slice)") is None


def test_ego4d_loader(tmp_path):
    recs = [
        {"clip_uid": "c1", "video_uid": "v1", "action_clip_start_sec": 1.0, "action_clip_end_sec": 2.5,
         "verb": "take_(pick,_grab,_get)", "noun": "cup"},
        {"clip_uid": "c1", "video_uid": "v1", "action_clip_start_sec": 4.0, "action_clip_end_sec": 5.0,
         "verb": "put_(place,_leave,_drop)", "noun": "cup"},
        {"clip_uid": "c2", "video_uid": "v2", "action_clip_start_sec": 0.0, "action_clip_end_sec": 3.0,
         "verb": "cut_(trim,_slice)", "noun": "onion"},
    ]
    p = tmp_path / "fho_lta_train.json"
    p.write_text(json.dumps({"clips": recs}))
    segs, groups = load_ego4d_segments(argparse.Namespace(annotations=[str(p)]))
    assert segs["c1"] == [(1.0, 2.5, "pickup", "cup"), (4.0, 5.0, "putdown", "cup")]
    assert "c2" not in segs and groups["c1"] == "v1"


def test_ego4d_unknown_format_gives_clear_error(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"clips": [{"uid": "c1", "t0": 1}]}))
    with pytest.raises(ValueError, match="First record keys"):
        load_ego4d_segments(argparse.Namespace(annotations=[str(p)]))


# ── Charades-Ego ──
def test_charades_loader(tmp_path):
    (tmp_path / "classes.txt").write_text(
        "c015 Taking a phone/camera from somewhere\nc016 Putting a phone/camera somewhere\n"
        "c017 Taking a picture of something\nc018 Putting on shoe/shoes\nc019 Holding some medicine\n")
    (tmp_path / "ann.csv").write_text(
        "id,subject,scene,actions,length\n"
        "AB12EGO,S1,Bedroom,c015 1.0 4.0;c017 2.0 3.0;c016 6.0 8.5,30\n"
        "AB12,S1,Bedroom,c015 1.0 4.0,30\n")              # third-person version: skipped
    segs, groups = load_charades_segments(argparse.Namespace(classes=str(tmp_path / "classes.txt"),
                                                             annotations=str(tmp_path / "ann.csv")))
    assert [s[2] for s in segs["AB12EGO"]] == ["pickup", "putdown"]
    assert "AB12" not in segs and groups["AB12EGO"] == "S1"


# ── split ──
def _v(ds, name):
    return {"name": name, "path": f"/x/{ds}/{name}.npz"}


def test_split_by_person_no_leakage_and_every_dataset_in_test():
    vids = [_v("epic", f"P{p:02d}__P{p:02d}_{k}") for p in range(1, 9) for k in range(3)] + \
           [_v("ego4d", f"vid{u}__clip{u}_{k}") for u in range(10) for k in range(2)]
    train, val, test = split_videos(vids, 0.15, 0.15, seed=0)
    g = lambda vs: {group_of(v) for v in vs}
    assert not (g(train) & g(val)) and not (g(train) & g(test)) and not (g(val) & g(test))
    assert len(train) + len(val) + len(test) == len(vids)
    for ds in ("epic", "ego4d"):
        assert any(ds in v["path"] for v in test) and any(ds in v["path"] for v in val)
