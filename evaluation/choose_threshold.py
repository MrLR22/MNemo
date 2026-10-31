# evaluation/choose_threshold.py
# Choose the trigger threshold on the VALIDATION people of HOI-TCN (P04 P21 P22 P27 P31), so the
# test-people benchmark (P02 P06 P08 P14 P26) is not used for any choice.
#
# The rule is fixed BEFORE the run and uses the same metric as the paper's test table:
#   gain(T) = actions found by mnemo@T  -  actions found by every-nth with the same number of VLM calls
# summed over the validation clips. The threshold with the largest gain wins; ties go to the one with
# fewer VLM calls. A clip-level bootstrap gives a 95% range for each gain.
#
# Input: the results list (or benchmark.json) written by evaluation/benchmark.run_benchmark.
#   python -m evaluation.choose_threshold results_val/benchmark.json

import json
import random
import sys


def per_clip(results: list[dict], th: float) -> list[tuple[int, int]]:
    """[(found by mnemo@th, found by its every-nth twin)] per clip."""
    a, b = f"mnemo@{th}", f"every-nth = mnemo@{th}"
    return [(r["systems"][a]["score"]["found"], r["systems"][b]["score"]["found"]) for r in results]


def bootstrap(pairs: list[tuple[int, int]], n: int = 10000, seed: int = 0) -> tuple[int, int]:
    """95% range of the summed difference, resampling clips."""
    if not pairs:
        return 0, 0
    rng = random.Random(seed)
    sums = sorted(sum(p - q for p, q in (pairs[rng.randrange(len(pairs))] for _ in pairs)) for _ in range(n))
    return sums[int(0.025 * n)], sums[int(0.975 * n) - 1]


def thresholds_in(results: list[dict]) -> list[float]:
    names = results[0]["systems"] if results else {}
    return sorted(float(s.split("@")[1]) for s in names if s.startswith("mnemo@") and "+" not in s)


def choose(results: list[dict]) -> tuple[dict, list[dict]]:
    rows = []
    for th in thresholds_in(results):
        pairs = per_clip(results, th)
        s = f"mnemo@{th}"
        calls = sum(r["systems"][s]["vlm_calls"] for r in results) / max(1, sum(r["systems"][s]["windows"] for r in results))
        lo, hi = bootstrap(pairs)
        rows.append({"threshold": th, "calls": calls, "found": sum(p for p, _ in pairs),
                     "found_every_nth": sum(q for _, q in pairs), "gain": sum(p - q for p, q in pairs),
                     "low": lo, "high": hi, "better": sum(p > q for p, q in pairs),
                     "worse": sum(p < q for p, q in pairs), "clips": len(pairs)})
    best = max(rows, key=lambda r: (r["gain"], -r["calls"])) if rows else None
    return best, rows


def report(results: list[dict]) -> str:
    best, rows = choose(results)
    n_true = sum(r["true_actions"] for r in results)
    lines = [f"## Trigger threshold chosen on validation people ({len(results)} clips, {n_true} true actions)", "",
             "| Threshold | VLM calls | Found: trigger | Found: every-nth | Gain [95% range] | Clips better / equal / worse |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        mark = " **← chosen**" if best and r["threshold"] == best["threshold"] else ""
        lines.append(f"| {r['threshold']}{mark} | {r['calls']:.0%} | {r['found']} | {r['found_every_nth']} "
                     f"| {r['gain']:+d} [{r['low']:+d}, {r['high']:+d}] "
                     f"| {r['better']} / {r['clips'] - r['better'] - r['worse']} / {r['worse']} |")
    if best:
        lines += ["", f"Chosen threshold: **{best['threshold']}** (largest gain over every-nth; ties go to fewer calls)."]
    return "\n".join(lines)


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        print(report(json.load(f)))
