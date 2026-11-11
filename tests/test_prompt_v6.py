# lines-v6: the any-place prompt (any action verb, no kitchen list, general names when unsure).
from modules.vlm_logger import build_prompt, canon_verb, parse_output


def test_v6_prompt_has_no_fixed_verb_list_and_no_kitchen_list():
    p = build_prompt(8, 4.0, True, ["fridge"], "lines-v6")
    assert "MUST be one of" not in p and "Use the closest one" not in p
    assert "for each of these" not in p.lower() and "No STATE line for anything that is not in view" in p
    assert "general name" in p and "CHECK:" in p and p.endswith("not in view.")
    assert "MUST be one of" in build_prompt(8, 4.0, True, None, "lines-v3")      # v3 unchanged


def test_free_verbs_are_kept_in_one_form():
    assert [canon_verb(v) for v in ("press", "presses", "rake", "carry", "wash", "picks up", "type on")] == \
        ["presses", "presses", "rakes", "carries", "washes", "picks up", "types on"]
    out = parse_output("PLACE: garden\nDID: rake | leaves | bucket\nDID: press | phone button | -")
    assert [(a["verb"], a["object"]) for a in out["actions"]] == [("rakes", "leaves"), ("presses", "phone button")]
