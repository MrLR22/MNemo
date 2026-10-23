# query/locate.py
# "Where did I put the drill?" → the MOMENT in the diary that answers it (start/end, seconds into the video).
# Used by the Ego4D NLQ test (evaluation/ego4d_nlq.py), where a person marked the answer's time window.
#
# 1. Candidates: diary rows that share words with the question. Rare words count more than common ones
#    (in a diary of 300 lines "put" is everywhere, "sieve" is not), ties go to the earlier row.
#    `candidates=` swaps in another search (e.g. keyword + vector) without touching the rest.
# 2. The language model sees the candidates in time order and picks the ONE line that shows the answer.
#    If its reply is not a valid line number, the best search match is used. Without a model: the best match.
# No candidate → no answer (counted as a miss, never a guess).
# Measured on 3 development clips: the old rule (equal word weights, model allowed to say "none")
# found the moment for 1 of 37 questions; the best match alone found 7.

import math
import re
from collections import Counter

from query.llm_qa import norm_tokens

PROMPT = """You help a person find a moment in their diary. Each line is something they did, written \
automatically from a camera they wear (it may contain mistakes). Times are minutes:seconds into the video.

{lines}

QUESTION: {question}

Which ONE line shows the moment that answers the question? For "where was X before / after I did Y", pick \
the line where they did Y with X. Reply with the line number only."""

_FIELDS = ("verb", "object", "where", "place")


def clock(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


def diary_line(r: dict) -> str:
    what = " ".join(x for x in (r["verb"], r["object"]) if x)
    extra = "; ".join(x for x in (f"at: {r['where']}" if r.get("where") else "",
                                  f"place: {r['place']}" if r.get("place") else "") if x)
    return f"{clock(r['t'])}-{clock(r['t_end'])} {what}" + (f" ({extra})" if extra else "")


def search(rows: list[dict], question: str, k: int = 20) -> list[dict]:
    """Rows sharing words with the question, best first; each shared word weighs log(N / rows with it)."""
    toks = [norm_tokens(" ".join(str(r.get(f) or "") for f in _FIELDS)) for r in rows]
    df = Counter(t for ts in toks for t in ts)
    q, n = norm_tokens(question), len(rows)
    scored = [(sum(math.log((n + 1) / (df[t] + 0.5)) for t in q & ts), r["t"], i)
              for i, (r, ts) in enumerate(zip(rows, toks)) if q & ts]
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [rows[i] for _, _, i in scored[:k]]


def locate(question: str, rows: list[dict], llm=None, k: int = 20, candidates=None) -> dict | None:
    """rows: diary rows with t / t_end in seconds (modules.vlm_logger.replay output).
    Returns {"t0", "t1", "row", "candidates", "by"} or None (no answer); by = "model" or "search"."""
    rows = [r for r in rows if r.get("verb")]
    ranked = (candidates or search)(rows, question, k)
    if not ranked:
        return None
    pick, by = ranked[0], "search"
    if llm is not None:
        cands = sorted(ranked, key=lambda r: r["t"])
        lines = "\n".join(f"[{i}] {diary_line(r)}" for i, r in enumerate(cands, 1))
        m = re.search(r"\d+", llm.chat(PROMPT.format(lines=lines, question=question), max_new_tokens=8))
        if m and 1 <= int(m.group()) <= len(cands):
            pick, by = cands[int(m.group()) - 1], "model"
    return {"t0": pick["t"], "t1": pick["t_end"], "row": pick, "candidates": len(ranked), "by": by}
