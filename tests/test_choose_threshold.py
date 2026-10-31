# The validation rule that picks the trigger threshold.

from evaluation.choose_threshold import choose, report


def _clip(found: dict, calls: dict):
    """found / calls: {threshold: (trigger, every-nth)}."""
    systems = {}
    for th, (a, b) in found.items():
        systems[f"mnemo@{th}"] = {"score": {"found": a}, "vlm_calls": calls[th], "windows": 15}
        systems[f"every-nth = mnemo@{th}"] = {"score": {"found": b}, "vlm_calls": calls[th], "windows": 15}
    systems["mnemo@0.7+every3"] = {"score": {"found": 99}, "vlm_calls": 15, "windows": 15}   # hybrids ignored
    return {"true_actions": 10, "systems": systems}


def test_largest_gain_wins_and_ties_go_to_fewer_calls():
    calls = {0.5: 13, 0.6: 11, 0.7: 8}
    res = [_clip({0.5: (5, 5), 0.6: (6, 4), 0.7: (5, 3)}, calls),
           _clip({0.5: (4, 3), 0.6: (5, 5), 0.7: (4, 4)}, calls)]
    best, rows = choose(res)
    assert [r["threshold"] for r in rows] == [0.5, 0.6, 0.7]
    assert [r["gain"] for r in rows] == [1, 2, 2]
    assert best["threshold"] == 0.7                       # same gain as 0.6, fewer calls
    assert rows[1]["better"] == 1 and rows[1]["worse"] == 0
    assert "**← chosen**" in report(res)
