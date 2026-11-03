# search/engine.py
# Searcher: the search part as the demo and the app use it. Reads one diary file (modules/storage.EventStore):
# the diary rows, the image fingerprints and the stored meaning vectors of a time range, runs the three searches
# (search/rank.py) and returns the best moments with real clock times.
#
#   s = Searcher(store, text=TextEmbedder(), clip=ClipEmbedder())     # either model may be None
#   s.search("where is my bicycle?", start, end)                      # rows, best first
#   s.search(..., diary_only=True)                                    # only rows the diary writer wrote
#   s.moment("where did I put the keys?", start, end, llm)            # ONE moment, picked by the LLM
#
# Meaning vectors are computed once per distinct diary line and kept in the diary file (text_vectors), so a
# question over a whole day only embeds the question. Image fingerprints are written while the video is
# processed (modules/vlm_logger.ActionLogger with store + embedder).

from datetime import datetime, timedelta

import numpy as np

from query.locate import locate
from search.rank import hybrid


class StoredText:
    """Meaning vectors kept in the diary file: only lines never seen before go through the text model."""

    def __init__(self, store, text):
        self.store, self.text = store, text

    def embed(self, texts: list[str]) -> np.ndarray:
        model = self.text.model_id
        have = self.store.text_vectors(list(dict.fromkeys(texts)), model)
        todo = [t for t in dict.fromkeys(texts) if t not in have]
        if todo:
            new = dict(zip(todo, self.text.embed(todo)))
            self.store.add_text_vectors(new, model)
            have.update(new)
        return np.stack([have[t] for t in texts]) if texts else np.zeros((0, 1), np.float32)


class Searcher:
    def __init__(self, store, text=None, clip=None):
        self.store, self.clip = store, clip
        self.text = StoredText(store, text) if text is not None else None

    def _rows(self, start: datetime | None, end: datetime | None) -> tuple[list[dict], datetime | None, tuple | None]:
        rows = [dict(r) for r in self.store.activities(start, end) if r.get("verb") or r.get("seen")]
        starts, _, fps = self.store.fingerprints(self.clip.model_id, start, end) if self.clip else ([], [], [])
        times = [r["ts_start"] for r in rows] + list(starts)
        ref = start or (min(times) if times else None)
        for r in rows:                                  # seconds since ref: what search/rank.py works with
            r["t"] = (r["ts_start"] - ref).total_seconds()
            r["t_end"] = (r["ts_end"] - ref).total_seconds()
        images = (fps, self.clip, [(t - ref).total_seconds() for t in starts]) if len(starts) else None
        return rows, ref, images

    def _candidates(self, images):
        return hybrid(text=self.text, images=images)

    def search(self, question: str, start: datetime | None = None, end: datetime | None = None, k: int = 20,
               diary_only: bool = False) -> list[dict]:
        """The best rows for the question, best first. Image-only moments carry image_only=True and no diary
        text; diary_only=True leaves them out (for yes/no answers, which need something the diary writer saw)."""
        rows, ref, images = self._rows(start, end)
        if ref is None:
            return []
        found = self._candidates(None if diary_only else images)(rows, question, k)
        for r in found:
            if r.get("image_only"):
                r["ts_start"] = ref + timedelta(seconds=r["t"])
                r["ts_end"] = ref + timedelta(seconds=r["t_end"])
        return found

    def moment(self, question: str, start: datetime | None = None, end: datetime | None = None, llm=None,
               k: int = 20) -> dict | None:
        """The ONE moment that answers the question (query/locate.py), with clock times; None = not found."""
        rows, ref, images = self._rows(start, end)
        if ref is None:
            return None
        hit = locate(question, rows, llm, k, candidates=self._candidates(images))
        if hit:
            hit["ts_start"] = ref + timedelta(seconds=hit["t0"])
            hit["ts_end"] = ref + timedelta(seconds=hit["t1"])
        return hit
