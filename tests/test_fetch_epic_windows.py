import cv2
import numpy as np

from tools.fetch_epic_windows import best_windows, fetch, plan, video_url, window_segments


def test_video_urls_match_official_layout():
    split = {"P01_01": "train"}
    assert video_url("P01_01", split) == \
        "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/train/P01/P01_01.MP4"
    assert video_url("P01_102", split) == \
        "https://data.bris.ac.uk/datasets/2g1n6qdydwa9u22shpxqzp0t8m/P01/videos/P01_102.MP4"


def test_best_windows_picks_busiest_and_no_overlap():
    segs = [(t, t + 1, "pickup", "x") for t in (10, 12, 14, 16, 18)] + [(200, 201, "putdown", "x")]
    w = best_windows(segs, duration=300, window_s=60, max_windows=3)
    assert w[0] == (5, 8.0)                       # 5 events, starting 2 s before the first
    assert all(abs(a[1] - b[1]) >= 60 for a in w for b in w if a is not b)


def test_window_segments_retimed_and_clipped():
    segs = [(5, 8, "pickup", "cup"), (58, 63, "putdown", "cup"), (70, 75, "pickup", "cup")]
    out = window_segments(segs, t0=6, window_s=60)
    assert out == [(0.0, 2, "pickup", "cup"), (52, 57, "putdown", "cup")]   # third ends after window


def test_plan_spreads_across_participants():
    segs = {"P01_01": [(t, t + 1, "pickup", "x") for t in range(0, 600, 3)],     # very busy person
            "P02_01": [(10, 11, "pickup", "x")], "P03_01": [(20, 21, "putdown", "x")]}
    dur = {"P01_01": 600, "P02_01": 100, "P03_01": 100}
    w = plan(segs, dur, n_windows=3, window_s=60)
    assert {x["video"][:3] for x in w} == {"P01", "P02", "P03"}


def test_fetch_cuts_and_shrinks_a_piece(tmp_path):
    src = str(tmp_path / "src.avi")
    vw = cv2.VideoWriter(src, cv2.VideoWriter_fourcc(*"MJPG"), 50, (1280, 720))
    for i in range(50 * 6):
        vw.write(np.full((720, 1280, 3), i % 255, np.uint8))
    vw.release()
    out = str(tmp_path / "piece.mp4")
    assert fetch(src, t0=2.0, window_s=3.0, out_path=out)
    cap = cv2.VideoCapture(out)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    assert (w, h) == (640, 360) and n == 30          # 3 s at 10 fps, 640 px wide
