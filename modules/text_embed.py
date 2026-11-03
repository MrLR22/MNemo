# modules/text_embed.py
# Meaning-based search over the reference diary: "cooker" finds "pressure cooker", "tidied the desk" finds
# "puts down pen | desk". A small sentence-embedding model (all-MiniLM-L6-v2, 22M parameters, runs on a CPU)
# turns each diary line and the question into a vector; similar meaning = close vectors.
# Embeddings are cached per text, so a diary line is embedded once however many questions are asked.

import numpy as np

import config as C


class TextEmbedder:
    def __init__(self, model_id: str = C.TEXT_EMBED_MODEL, device: str | None = None):
        from sentence_transformers import SentenceTransformer
        self.model_id = model_id
        self.model = SentenceTransformer(model_id, device=device)
        self.cache: dict[str, np.ndarray] = {}

    def embed(self, texts: list[str]) -> np.ndarray:
        """Unit-length vectors, one row per text."""
        todo = list(dict.fromkeys(t for t in texts if t not in self.cache))
        if todo:
            for t, v in zip(todo, self.model.encode(todo, normalize_embeddings=True, batch_size=64)):
                self.cache[t] = np.asarray(v, np.float32)
        return np.stack([self.cache[t] for t in texts]) if texts else np.zeros((0, 1), np.float32)
