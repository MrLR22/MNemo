# modules/location_detector.py
# Detects room type and surface location using CLIP.
# Shares the CLIP model with ActionClassifier to avoid double-loading.

import torch
import clip
import numpy as np
from PIL import Image

from config import CLIP_MODEL, ROOM_PROMPTS, SURFACE_PROMPTS


class LocationDetector:
    def __init__(self, model=None, preprocess=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if model is not None and preprocess is not None:
            self.model = model
            self.preprocess = preprocess
        else:
            self.model, self.preprocess = clip.load(CLIP_MODEL, device=self.device)

        self.room_features = self._encode_prompts(ROOM_PROMPTS)
        self.surface_features = self._encode_prompts(SURFACE_PROMPTS)

        print(f"[LocationDetector] Ready on {self.device}")

    # ── public API ──────────────────────────────────────────────────────────

    def detect_room(self, frame: np.ndarray) -> dict:
        """Classify the room type from a full frame."""
        label, score = self._score(frame, self.room_features, ROOM_PROMPTS)
        return {"room": label, "confidence": round(score, 4)}

    def detect_surface(self, frame: np.ndarray, object_box: tuple = None) -> dict:
        """Classify the surface near an object. Uses a padded crop if box is given."""
        region = self._crop_region(frame, object_box) if object_box else frame
        label, score = self._score(region, self.surface_features, SURFACE_PROMPTS)

        vertical = "middle"
        if object_box:
            h = frame.shape[0]
            cy = (object_box[1] + object_box[3]) / 2
            if cy < h * 0.33:
                vertical = "top"
            elif cy > h * 0.66:
                vertical = "bottom"

        return {"surface": label, "confidence": round(score, 4), "position": vertical}

    # ── internals ───────────────────────────────────────────────────────────

    def _encode_prompts(self, prompts: list[str]) -> torch.Tensor:
        """Tokenize and encode a prompt list into normalised text features."""
        tokens = clip.tokenize(prompts).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(tokens)
            features /= features.norm(dim=-1, keepdim=True)
        return features

    def _score(self, frame: np.ndarray, text_features: torch.Tensor, prompts: list[str]) -> tuple:
        """Return (best_prompt, best_score) for a frame against text features."""
        pil = Image.fromarray(frame[:, :, ::-1])
        image = self.preprocess(pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            img_features = self.model.encode_image(image)
            img_features /= img_features.norm(dim=-1, keepdim=True)

            logit_scale = self.model.logit_scale.exp()
            logits = logit_scale * img_features @ text_features.T
            scores = logits.softmax(dim=-1)[0]

        idx = scores.argmax().item()
        return prompts[idx], float(scores[idx])

    @staticmethod
    def _crop_region(frame: np.ndarray, box: tuple, pad: int = 40) -> np.ndarray:
        """Crop a padded region around a bounding box. Falls back to full frame."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        region = frame[y1:y2, x1:x2]
        return region if region.size > 0 else frame
