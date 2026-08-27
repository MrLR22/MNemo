# modules/action_classifier.py
# Uses CLIP to score action prompts against a video frame.
# Returns ranked similarity scores for all defined actions.

import torch
import clip
import numpy as np
from PIL import Image

from config import CLIP_MODEL, ACTION_PROMPTS


class ActionClassifier:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(CLIP_MODEL, device=self.device)

        tokens = clip.tokenize(ACTION_PROMPTS).to(self.device)
        with torch.no_grad():
            self.text_features = self.model.encode_text(tokens)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

        print(f"[ActionClassifier] CLIP ({CLIP_MODEL}) loaded on {self.device}")

    # ── public API ──────────────────────────────────────────────────────────

    def classify(self, frame: np.ndarray) -> list[dict]:
        """Return action scores for *frame*, sorted highest-first."""
        image_features = self._encode_frame(frame)
        scores = (image_features @ self.text_features.T).softmax(dim=-1)[0].cpu().numpy()

        return sorted(
            [{"action": p, "score": round(float(s), 4)} for p, s in zip(ACTION_PROMPTS, scores)],
            key=lambda x: x["score"],
            reverse=True,
        )

    def get_embedding(self, frame: np.ndarray) -> np.ndarray:
        """Return the L2-normalised image embedding for *frame*."""
        return self._encode_frame(frame).cpu().numpy()[0]

    def embed_text(self, text: str) -> np.ndarray:
        """Return the L2-normalised text embedding for *text*."""
        tokens = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(tokens)
            features /= features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()[0]

    # ── internals ───────────────────────────────────────────────────────────

    def _encode_frame(self, frame: np.ndarray) -> torch.Tensor:
        """BGR ndarray → normalised CLIP image feature tensor."""
        pil = Image.fromarray(frame[:, :, ::-1])
        image = self.preprocess(pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(image)
            features /= features.norm(dim=-1, keepdim=True)
        return features
