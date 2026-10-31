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
#
# hybrid(): three searches fused by rank (reciprocal rank fusion, Cormack et al. 2009), no weights to tune:
#   keyword  search() above, over what was done and what was SEEN (prompt lines-v4)
#   meaning  text embeddings of the diary lines (modules/text_embed.py)
#   image    CLIP fingerprints of EVERY window (modules/landmarks.ClipEmbedder), also windows the trigger
#            skipped; such a window enters the list as "not described, the camera image matches".

import math
import re
from collections import Counter

import numpy as np

import config as C
from query.llm_qa import STOP, norm_tokens

PROMPT = """You help a person find a moment in their diary. Each line is something they did or saw, written \
automatically from a camera they wear (it may contain mistakes). Times are minutes:seconds into the video.

{lines}

QUESTION: {question}

Which ONE line shows the moment that answers the question? For "where was X before / after I did Y", pick \
the line where they did Y with X. Reply with the line number only."""

_FIELDS = ("verb", "object", "where", "place", "seen")


def clock(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:04.1f}"


def diary_text(r: dict) -> str:
    """The line without its time (what the meaning search embeds)."""
    if r.get("image_only"):
        return "not described; the camera image matches the question"
    what = " ".join(x for x in (r["verb"], r["object"]) if x)
    extra = "; ".join(x for x in (f"at: {r['where']}" if r.get("where") else "",
                                  f"place: {r['place']}" if r.get("place") else "",
                                  f"saw: {r['seen']}" if r.get("seen") else "") if x)
    return " ".join(x for x in (what, f"({extra})" if extra else "") if x)


def diary_line(r: dict) -> str:
    return f"{clock(r['t'])}-{clock(r['t_end'])} {diary_text(r)}"


def search(rows: list[dict], question: str, k: int = 20) -> list[dict]:
    """Rows sharing words with the question, best first; each shared word weighs log(N / rows with it)."""
    toks = [norm_tokens(" ".join(str(r.get(f) or "") for f in _FIELDS)) for r in rows]
    df = Counter(t for ts in toks for t in ts)
    q, n = norm_tokens(question), len(rows)
    scored = [(sum(math.log((n + 1) / (df[t] + 0.5)) for t in q & ts), r["t"], i)
              for i, (r, ts) in enumerate(zip(rows, toks)) if q & ts]
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [rows[i] for _, _, i in scored[:k]]


def _image_query(question: str) -> str:
    words = [w for w in re.findall(r"[a-z]+", question.lower()) if w not in STOP and len(w) > 1]
    return "a photo of " + " ".join(words or [question])


def hybrid(text=None, images=None, window_s: float = C.VLM_WINDOW_S, pool: int = 50, fusion_k: int = C.FUSION_K):
    """A `candidates=` function for locate(): keyword + meaning + image search fused by rank.
    text: modules.text_embed.TextEmbedder or None. images: (fingerprints [windows, d], ClipEmbedder) or None."""
    def run(rows: list[dict], question: str, k: int = 20) -> list[dict]:
        rows = list(rows)
        pos = {id(r): i for i, r in enumerate(rows)}
        rankings = [[pos[id(r)] for r in search(rows, question, pool)]]
        if text is not None and rows:
            sims = text.embed([diary_text(r) for r in rows]) @ text.embed([question])[0]
            rankings.append([int(i) for i in np.argsort(-sims, kind="stable")[:pool]])
        if images is not None:
            fps, clip = images
            sims = fps.astype(np.float32) @ clip.embed_text([_image_query(question)])[0]
            ranked = []
            for w in np.argsort(-sims, kind="stable")[:pool]:
                t0 = float(w) * window_s
                hit = [i for i, r in enumerate(rows) if r["t"] <= t0 < r["t_end"] or t0 <= r["t"] < t0 + window_s]
                if not hit:                          # a skipped (or empty) window: findable by its image alone
                    rows.append({"t": t0, "t_end": t0 + window_s, "verb": None, "object": None, "where": None,
                                 "place": None, "seen": None, "image_only": True})
                    hit = [len(rows) - 1]
                ranked += [i for i in hit if i not in ranked]
            rankings.append(ranked)
        score = Counter()
        for ranking in rankings:
            for rank, i in enumerate(ranking, 1):
                score[i] += 1.0 / (fusion_k + rank)
        order = sorted(score, key=lambda i: (-score[i], rows[i]["t"]))
        return [rows[i] for i in order[:k]]
    return run


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
