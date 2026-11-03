# query/locate.py
# "Where did I put the drill?" → the MOMENT in the diary that answers it (start/end, seconds into the video).
# Used by the Ego4D NLQ test (evaluation/ego4d_nlq.py), where a person marked the answer's time window.
#
# 1. Candidates: search/rank.py (keyword search by default; `candidates=` swaps in another one, e.g. hybrid()).
# 2. The language model sees the candidates in time order and picks the ONE line that shows the answer.
#    If its reply is not a valid line number, the best search match is used. Without a model: the best match.
# No candidate → no answer (counted as a miss, never a guess).
# Measured on 3 development clips: the old rule (equal word weights, model allowed to say "none")
# found the moment for 1 of 37 questions; the best match alone found 7.

import re

from search.rank import diary_text, hybrid, keyword_search as search   # noqa: F401  (re-exported)

PROMPT = """You help a person find a moment in their diary. Each line is something they did or saw, written \
automatically from a camera they wear (it may contain mistakes). Times are minutes:seconds into the video.

{lines}

QUESTION: {question}

Which ONE line shows the moment that answers the question? For "where was X before / after I did Y", pick \
the line where they did Y with X. Reply with the line number only."""


def clock(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


def diary_line(r: dict) -> str:
    return f"{clock(r['t'])}-{clock(r['t_end'])} {diary_text(r)}"


def locate(question: str, rows: list[dict], llm=None, k: int = 20, candidates=None) -> dict | None:
    """rows: diary rows with t / t_end in seconds (modules.vlm_logger.replay output).
    Returns {"t0", "t1", "row", "candidates", "by"} or None (no answer); by = "model" or "search"."""
    rows = [r for r in rows if r.get("verb") or r.get("seen")]
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
