# evaluation/export_demo.py
# Collect the real outputs of a finished Kaggle run into one JSON file for the demo page:
# the whole-morning log, end-of-morning alerts, every morning question with Mnemo's and the direct
# model's answers, one clip's log next to its EPIC ground truth, and the headline numbers.
#
#   python -m evaluation.export_demo --results data/kaggle_results/v3_final --annotations EPIC_100_train.csv \
#       --classes-dir <folder with the EPIC class CSVs> --out demo_data.json

import argparse
import json
import os
import sqlite3
from datetime import timedelta

import numpy as np

import config as C
from evaluation.benchmark import START, _pad, _windows_from_json
from evaluation.log_accuracy import EpicVocab, clip_name_parts, load_truth, merge_scores, score_log
from evaluation.long_qa import GAP_S
from evaluation.qa_benchmark import score_answers
from modules.hoi_gate import gate_decisions
from modules.vlm_logger import StateTracker, parse_output, replay


def morning_log(db: str) -> list[dict]:
    con = sqlite3.connect(db)
    rows = con.execute("SELECT ts_start, ts_end, verb, object, place, where_, source FROM activities "
                       "WHERE verb IS NOT NULL ORDER BY ts_start, id").fetchall()
    clean = lambda x: (x or "").removeprefix("->").removesuffix("->").strip().strip("-").strip()   # older parser kept "->"
    return [{"start": a[11:19], "end": b[11:19], "verb": v, "object": o or "", "place": clean(p),
             "where": clean(w), "state": s == "state"} for a, b, v, o, p, w, s in rows]


def morning_alerts(cache_dir: str, clips: list[str], gap_s: float) -> list[dict]:
    tr = StateTracker()
    for i, name in enumerate(clips):
        with open(os.path.join(cache_dir, f"{name}__{C.VLM_PROMPT_VERSION}.json")) as f:
            c = json.load(f)
        ws = _windows_from_json(c["windows"])
        call = _pad(gate_decisions(np.array(c["scores"])), len(ws))
        for w in ws:
            if w["text"] is not None and call[w["k"]]:
                end = w["ts"] + timedelta(seconds=i * gap_s + w["seconds"])
                tr.update(parse_output(w["text"])["states"], end)
    return [{"object": k.removesuffix(" power"), "state": st, "since": ts.strftime("%H:%M:%S")}
            for k, (st, ts) in sorted(tr.current().items(), key=lambda x: x[1][1]) if st in ("open", "on")]


def clip_example(cache_dir, name, ann, vocab):
    with open(os.path.join(cache_dir, f"{name}__{C.VLM_PROMPT_VERSION}.json")) as f:
        c = json.load(f)
    ws = _windows_from_json(c["windows"])
    call = _pad(gate_decisions(np.array(c["scores"])), len(ws))
    rows = replay(ws, call, track_states=True)
    vid, t0 = clip_name_parts(name)
    truth = load_truth(ann, vid, t0, c["stats"]["video_seconds"])
    sc = score_log(rows, truth, vocab)
    return {"clip": name, "found": sc["found"], "true": sc["true_actions"],
            "windows": [{"t": round(w["t"]), "called": bool(call[w["k"]]), "score": round(float(s), 2)}
                        for w, s in zip(ws, c["scores"])],
            "log": [{"t": round(r["t"], 1), "verb": r["verb"], "object": r["object"] or "", "state": r.get("source") == "state"}
                    for r in rows if r["verb"]],
            "truth": [{"t": round(g["start"], 1), "narration": g["narration"], "found": bool(f)} for g, f in sc["per_truth"]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--classes-dir", required=True)
    ap.add_argument("--clip", default="P02_08_w3", help="clip shown next to its ground truth")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    vocab = EpicVocab(os.path.join(a.classes_dir, "EPIC_100_verb_classes.csv"),
                      os.path.join(a.classes_dir, "EPIC_100_noun_classes.csv"))
    results = json.load(open(os.path.join(a.results, "benchmark.json")))
    lq = json.load(open(os.path.join(a.results, "long_qa.json")))
    clips = [os.path.splitext(os.path.basename(c))[0] for c in lq["clips"]]
    cache = os.path.join(a.results, "vlm_cache")
    direct = [k for k in lq["questions"] if k != "mnemo"][0]
    qs = [{"q": m["q"], "expect": m["expect"], "mnemo": m["answer"], "mnemo_yes": m["yes"],
           "direct": d["answer"], "direct_yes": d["yes"]} for m, d in zip(lq["questions"]["mnemo"], lq["questions"][direct])]
    full = merge_scores([r["systems"]["+ states"]["score"] for r in results])
    plain = merge_scores([r["systems"]["captions"]["score"] for r in results])
    m_long, d_long = score_answers(lq["questions"]["mnemo"]), score_answers(lq["questions"][direct])
    data = {
        "morning": {"start": START.strftime("%H:%M"), "clips": len(clips), "gap_min": lq["gap_s"] / 60,
                    "log": morning_log(os.path.join(a.results, "morning.db")),
                    "alerts": morning_alerts(cache, clips, lq.get("gap_s", GAP_S))},
        "questions": qs,
        "when": [{"q": w["q"], "answer": w["answer"], "correct": w["correct"]} for w in lq["when"]],
        "example": clip_example(cache, a.clip, a.annotations, vocab),
        "numbers": {"clips": len(results), "true_actions": full["true_actions"],
                    "state_plain": plain["state_found"], "state_ours": full["state_found"], "state_total": full["state_true"],
                    "morning_acc_mnemo": m_long["accuracy"], "morning_acc_direct": d_long["accuracy"],
                    "morning_invented_mnemo": m_long["no_answered_yes"], "morning_invented_direct": d_long["no_answered_yes"],
                    "morning_no_q": m_long["no_questions"],
                    "when_ok": sum(w["correct"] for w in lq["when"]), "when_total": len(lq["when"])},
    }
    with open(a.out, "w") as f:
        json.dump(data, f, indent=1)
    print(f"{len(data['morning']['log'])} log rows, {len(data['morning']['alerts'])} alerts, {len(qs)} questions, "
          f"example {a.clip}: {data['example']['found']}/{data['example']['true']} found → {a.out}")


if __name__ == "__main__":
    main()
