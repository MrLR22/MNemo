# search/rank.py
# Three searches over diary rows, each giving its own ranking, combined by reciprocal rank fusion
# (Cormack et al. 2009): score = sum over searches of 1 / (FUSION_K + rank). No weights to tune.
#
#   keyword  rows sharing words with the question; rare words count more (in a diary of 300 lines "put" is
#            everywhere, "sieve" is not), ties go to the earlier row
#   meaning  text embeddings of the diary lines (modules/text_embed.py): "cooker" finds "stove"
#   image    CLIP fingerprints of EVERY window (modules/landmarks.ClipEmbedder), also windows the trigger skipped;
#            such a window enters the list as "not described, the camera image matches". An image match can
#            point to a MOMENT, it never shows that the person DID something (query/llm_qa.py leaves them out).
#
# Rows are dicts with t / t_end in seconds (from any reference time) and the diary fields
# (verb, object, where, place, seen).
#
# Measured on Ego4D NLQ (20 clips never used to build the search, prompt with SEEN line, line picked by the LLM):
# keyword only 40 right moments, all three 68 (+28, 95% range +16 to +40).

import math
import re
from collections import Counter

import numpy as np

import config as C
from query.llm_qa import STOP, norm_tokens

_FIELDS = ("verb", "object", "where", "place", "seen")


def diary_text(r: dict) -> str:
    """The line without its time (what the meaning search embeds)."""
    if r.get("image_only"):
        return "not described; the camera image matches the question"
    what = " ".join(x for x in (r["verb"], r["object"]) if x)
    extra = "; ".join(x for x in (f"at: {r['where']}" if r.get("where") else "",
                                  f"place: {r['place']}" if r.get("place") else "",
                                  f"saw: {r['seen']}" if r.get("seen") else "") if x)
    return " ".join(x for x in (what, f"({extra})" if extra else "") if x)


def keyword_search(rows: list[dict], question: str, k: int = 20) -> list[dict]:
    """Rows sharing words with the question, best first; each shared word weighs log(N / rows with it)."""
    toks = [norm_tokens(" ".join(str(r.get(f) or "") for f in _FIELDS)) for r in rows]
    df = Counter(t for ts in toks for t in ts)
    q, n = norm_tokens(question), len(rows)
    scored = [(sum(math.log((n + 1) / (df[t] + 0.5)) for t in q & ts), r["t"], i)
              for i, (r, ts) in enumerate(zip(rows, toks)) if q & ts]
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [rows[i] for _, _, i in scored[:k]]


def image_query(question: str) -> str:
    words = [w for w in re.findall(r"[a-z]+", question.lower()) if w not in STOP and len(w) > 1]
    return "a photo of " + " ".join(words or [question])


def hybrid(text=None, images=None, window_s: float = C.VLM_WINDOW_S, pool: int = 50, fusion_k: int = C.FUSION_K):
    """A `candidates=` function for query.locate.locate(): keyword + meaning + image search fused by rank.
    text: an object with .embed(texts) → unit vectors (TextEmbedder), or None.
    images: (fingerprints [windows, d], clip) or (fingerprints, clip, window starts in seconds), or None;
    clip has .embed_text(texts). Without starts, window w starts at w * window_s."""
    def run(rows: list[dict], question: str, k: int = 20) -> list[dict]:
        rows = list(rows)
        pos = {id(r): i for i, r in enumerate(rows)}
        rankings = [[pos[id(r)] for r in keyword_search(rows, question, pool)]]
        if text is not None and rows:
            sims = text.embed([diary_text(r) for r in rows]) @ text.embed([question])[0]
            rankings.append([int(i) for i in np.argsort(-sims, kind="stable")[:pool]])
        if images is not None and len(images[0]):
            fps, clip, *starts = images
            starts = np.asarray(starts[0], float) if starts else np.arange(len(fps)) * window_s
            sims = np.asarray(fps, np.float32) @ clip.embed_text([image_query(question)])[0]
            ranked = []
            for w in np.argsort(-sims, kind="stable")[:pool]:
                t0 = float(starts[w])
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
