# evaluation/log_accuracy.py
# How good is the STAGE 1 log? Compare it with the EPIC-KITCHENS ground truth of the same clip.
#
# Matching uses EPIC's own vocabulary (EPIC_100_verb_classes.csv / EPIC_100_noun_classes.csv), so
# "picks up" = take, "refrigerator door" = fridge, "plastic container" = container, without a
# hand-made synonym list. A true action counts as FOUND when a log row overlaps it in time
# (± TOL_S seconds) with the same verb class AND one of its noun classes.
#
#   recall      found true actions / all true actions          ("how much of my day is in the log")
#   precision   log rows matching some true action / log rows    (the rest are wrong OR unannotated)
#   open/close  recall on open, close, turn-on, turn-off only   (what the object-state tracker targets)
#
# EPIC annotators skip some small actions, so a log row without a match is not always invented:
# the unmatched rows are listed so they can be checked by eye.
#
#   python -m evaluation.log_accuracy --db mnemo_log.db --video P02_102_w38 --start "2026-10-16 09:00" \
#       --annotations EPIC_100_train.csv --classes-dir <folder with the two class CSVs>

import argparse
import ast
import csv
import os
import re
from datetime import datetime

TOL_S = 2.0
STATE_VERBS = {"open", "close", "turn-on", "turn-off"}
_ADJECTIVES = set("""red orange yellow green blue purple pink brown black white grey gray silver golden
small big large little tiny empty full open closed new old clear transparent dark light""".split()) - {"light"}


def secs(ts: str) -> float:
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _lemmas(w: str) -> list[str]:
    """picks → pick, washes → wash, empties → empty, closes → close (try all, vocabulary decides)."""
    out = [w]
    if w.endswith("ies"):
        out.append(w[:-3] + "y")
    if w.endswith("es"):
        out.append(w[:-2])
    if w.endswith("s"):
        out.append(w[:-1])
    if w.endswith("ing"):
        out += [w[:-3], w[:-3] + "e"]
    if w.endswith("ed"):
        out += [w[:-2], w[:-1]]
    return out


class EpicVocab:
    def __init__(self, verb_csv: str, noun_csv: str):
        self.verb, self.noun, self.verb_key, self.noun_key = {}, {}, {}, {}
        for path, inst, keys in ((verb_csv, self.verb, self.verb_key), (noun_csv, self.noun, self.noun_key)):
            with open(path) as f:
                for r in csv.DictReader(f):
                    c = int(r["id"])
                    keys[c] = r["key"]
                    for i in [r["key"]] + ast.literal_eval(r["instances"]):
                        inst.setdefault(i.lower(), c)

    def verb_class(self, text: str | None) -> int | None:
        words = re.findall(r"[a-z]+", (text or "").lower())
        if not words:
            return None
        for first in _lemmas(words[0]):
            if len(words) > 1:                              # "picks up" → pick-up, "turns off" → turn-off
                c = self.verb.get(f"{first}-{words[1]}")
                if c is not None:
                    return c
            c = self.verb.get(first)
            if c is not None:
                return c
        return None

    def noun_classes(self, text: str | None) -> set[int]:
        """'plastic bag of onions' → {bag, onion}; 'fridge door' → {fridge}; 'light switch' → {light, switch}."""
        words = re.findall(r"[a-z]+", (text or "").lower())
        words = [w for w in words if w not in ("a", "an", "the", "of", "some", "with", "and", "in", "on")]
        if len(words) > 1:                                  # "orange bag" is a bag, not an orange
            words = [w for w in words if w not in _ADJECTIVES] or words
        out = set()
        if len(words) >= 2:                                 # EPIC writes "noun:modifier" ("door:fridge")
            for a in _lemmas(words[-1]):
                c = self.noun.get(f"{a}:{':'.join(reversed(words[:-1]))}") or self.noun.get(f"{a}:{words[-2]}")
                if c is not None:
                    return {c}
        for w in words:
            for a in _lemmas(w):
                c = self.noun.get(a)
                if c is not None:
                    out.add(c)
                    break
        return out

    def noun_text(self, c: int) -> str:
        """'machine:washing' → 'washing machine'."""
        return " ".join(reversed(self.noun_key[c].split(":")))


def load_truth(ann_csv: str, video: str, t0: float, duration: float) -> list[dict]:
    """True actions of EPIC video `video` inside [t0, t0+duration], times relative to t0."""
    out = []
    with open(ann_csv) as f:
        for r in csv.DictReader(f):
            if r["video_id"] != video:
                continue
            s, e = secs(r["start_timestamp"]) - t0, secs(r["stop_timestamp"]) - t0
            if e <= 0 or s >= duration:
                continue
            nouns = {int(x) for x in ast.literal_eval(r["all_noun_classes"])} if r.get("all_noun_classes") else set()
            out.append({"start": max(0.0, s), "stop": min(duration, e), "narration": r["narration"],
                        "verb_class": int(r["verb_class"]), "noun_class": int(r["noun_class"]),
                        "nouns": nouns | {int(r["noun_class"])}})
    return sorted(out, key=lambda x: x["start"])


def clip_name_parts(name: str) -> tuple[str, float]:
    """'P02_102_w38' → ('P02_102', 38.0)."""
    video, _, t0 = os.path.splitext(os.path.basename(name))[0].rpartition("_w")
    return video, float(t0)


def score_log(rows: list[dict], truth: list[dict], vocab: EpicVocab, tol: float = TOL_S) -> dict:
    """rows: log rows with 't' and 't_end' (seconds from clip start), 'verb', 'object'."""
    acts = [r for r in rows if r.get("verb")]
    for r in acts:
        r["_v"], r["_n"] = vocab.verb_class(r["verb"]), vocab.noun_classes(r["object"])

    def near(r, g):
        return r["t"] - tol <= g["stop"] and r["t_end"] + tol >= g["start"]

    def full(r, g):
        return r["_v"] == g["verb_class"] and bool(r["_n"] & g["nouns"])

    found = {"full": 0, "verb": 0, "noun": 0}
    state_total = state_found = 0
    per_truth = []
    for g in truth:
        cand = [r for r in acts if near(r, g)]
        f = any(full(r, g) for r in cand)
        found["full"] += f
        found["verb"] += any(r["_v"] == g["verb_class"] for r in cand)
        found["noun"] += any(r["_n"] & g["nouns"] for r in cand)
        is_state = vocab.verb_key[g["verb_class"]] in STATE_VERBS
        state_total += is_state
        state_found += is_state and f
        per_truth.append((g, f))
    correct = [r for r in acts if any(near(r, g) and full(r, g) for g in truth)]
    noun_ok = [r for r in acts if any(near(r, g) and r["_n"] & g["nouns"] for g in truth)]
    unmatched = [r for r in acts if r not in noun_ok]
    n = len(truth)
    return {
        "true_actions": n, "log_rows": len(acts),
        "found": found["full"], "recall": found["full"] / n if n else None,
        "recall_verb_only": found["verb"] / n if n else None,
        "recall_object_only": found["noun"] / n if n else None,
        "precision": len(correct) / len(acts) if acts else None,
        "precision_object_only": len(noun_ok) / len(acts) if acts else None,
        "state_true": state_total, "state_found": state_found,
        "state_recall": state_found / state_total if state_total else None,
        "per_truth": per_truth, "unmatched": unmatched,
    }


def merge_scores(scores: list[dict]) -> dict:
    """Micro-average over clips (every true action / log row counts once)."""
    tot = lambda k: sum(s[k] for s in scores)
    n, rows = tot("true_actions"), tot("log_rows")
    out = {"clips": len(scores), "true_actions": n, "log_rows": rows, "found": tot("found"),
           "state_true": tot("state_true"), "state_found": tot("state_found")}
    out["recall"] = out["found"] / n if n else None
    out["state_recall"] = out["state_found"] / out["state_true"] if out["state_true"] else None
    for k in ("recall_verb_only", "recall_object_only"):
        out[k] = sum(s[k] * s["true_actions"] for s in scores if s[k] is not None) / n if n else None
    for k in ("precision", "precision_object_only"):
        out[k] = sum(s[k] * s["log_rows"] for s in scores if s[k] is not None) / rows if rows else None
    return out


def print_report(sc: dict, title: str = ""):
    pct = lambda x: f"{x:.0%}" if x is not None else "-"
    if title:
        print(title)
    print(f"  recall {pct(sc['recall'])} ({sc['found']}/{sc['true_actions']} true actions found)"
          f" | verb only {pct(sc['recall_verb_only'])} | object only {pct(sc['recall_object_only'])}")
    print(f"  precision {pct(sc['precision'])} of {sc['log_rows']} log rows"
          f" (object right: {pct(sc['precision_object_only'])})")
    print(f"  open/close/on/off: {sc['state_found']}/{sc['state_true']} found ({pct(sc['state_recall'])})")


def print_details(sc: dict):
    for g, f in sc["per_truth"]:
        print(f"   {'✓' if f else '✗'} {g['start']:5.1f}s  {g['narration']}")
    if sc["unmatched"]:
        print("  log rows with no matching true object nearby (wrong, or not annotated by EPIC: check by eye):")
        for r in sc["unmatched"]:
            print(f"     {r['t']:5.1f}s  {r['verb']} {r['object'] or ''}")


def rows_from_store(store, start: datetime) -> list[dict]:
    rows = store.activities()
    for r in rows:
        r["t"] = (r["ts_start"] - start).total_seconds()
        r["t_end"] = (r["ts_end"] - start).total_seconds()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--video", required=True, help="clip name <video>_w<start>, e.g. P02_102_w38")
    ap.add_argument("--start", required=True, help='the start time given to the logger, e.g. "2026-10-16 09:00"')
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--classes-dir", required=True)
    args = ap.parse_args()
    from modules.storage import EventStore
    vocab = EpicVocab(os.path.join(args.classes_dir, "EPIC_100_verb_classes.csv"),
                      os.path.join(args.classes_dir, "EPIC_100_noun_classes.csv"))
    video, t0 = clip_name_parts(args.video)
    truth = load_truth(args.annotations, video, t0, args.duration)
    sc = score_log(rows_from_store(EventStore(args.db), datetime.fromisoformat(args.start)), truth, vocab)
    print_report(sc, args.video)
    print_details(sc)


if __name__ == "__main__":
    main()
