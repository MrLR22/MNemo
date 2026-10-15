# modules/landmarks.py
# M4: home-tour enrollment → CLIP landmark index.
# M5: resolve a post-put-down frame to the nearest landmark, with an "unknown" fallback.
#
# Enrollment input (pick either, or both):
#   data/enroll/<landmark_name>/   short videos and/or photos of that landmark
#                                  e.g. data/enroll/entryway_table/clip1.mp4
#   data/enroll/tour.csv           columns: video,start_s,end_s,landmark
#                                  (time ranges inside one home-tour video)
# The index stores only embeddings + names (no images).

import csv
import os
from collections import defaultdict

import cv2
import numpy as np

import config as C

_VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ClipEmbedder:
    def __init__(self):
        import open_clip
        import torch
        self.torch = torch
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            C.CLIP_ARCH, pretrained=C.CLIP_PRETRAINED)
        self.tokenizer = open_clip.get_tokenizer(C.CLIP_ARCH)
        self.model.eval()

    def embed_text(self, texts: list[str]) -> np.ndarray:
        with self.torch.no_grad():
            f = self.model.encode_text(self.tokenizer(texts))
            f = f / f.norm(dim=-1, keepdim=True)
        return f.cpu().numpy().astype(np.float32)

    @property
    def logit_scale(self) -> float:
        return float(self.model.logit_scale.exp())

    def embed(self, images_bgr: list[np.ndarray]) -> np.ndarray:
        from PIL import Image
        if not images_bgr:
            return np.zeros((0, 512), np.float32)
        batch = self.torch.stack([
            self.preprocess(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))) for im in images_bgr])
        with self.torch.no_grad():
            f = self.model.encode_image(batch)
            f = f / f.norm(dim=-1, keepdim=True)
        return f.cpu().numpy().astype(np.float32)


def _video_frames(path, fps, start_s=None, end_s=None):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(src_fps / fps))
    i = 0
    frames = []
    while cap.grab():
        t = i / src_fps
        if end_s is not None and t > end_s:
            break
        if i % step == 0 and (start_s is None or t >= start_s):
            ok, im = cap.retrieve()
            if ok:
                frames.append(im)
        i += 1
    cap.release()
    return frames


def collect_enrollment(enroll_dir: str = C.ENROLL_DIR, fps: float = C.ENROLL_SAMPLE_FPS):
    """Return {source_id: (landmark, [frames])}. A source is one video/photo/tour segment."""
    sources = {}
    if not os.path.isdir(enroll_dir):
        raise FileNotFoundError(f"No enrollment folder at {enroll_dir}")
    for name in sorted(os.listdir(enroll_dir)):
        d = os.path.join(enroll_dir, name)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            path = os.path.join(d, fn)
            ext = os.path.splitext(fn)[1].lower()
            if ext in _VIDEO_EXT:
                sources[path] = (name, _video_frames(path, fps))
            elif ext in _IMAGE_EXT:
                im = cv2.imread(path)
                if im is not None:
                    sources[path] = (name, [im])
    tour = os.path.join(enroll_dir, "tour.csv")
    if os.path.exists(tour):
        with open(tour) as f:
            for row in csv.DictReader(f):
                video = row["video"] if os.path.isabs(row["video"]) else os.path.join(enroll_dir, row["video"])
                sid = f"{video}@{row['start_s']}-{row['end_s']}"
                sources[sid] = (row["landmark"].strip(),
                                _video_frames(video, fps, float(row["start_s"]), float(row["end_s"])))
    return {k: v for k, v in sources.items() if v[1]}


class LandmarkIndex:
    def __init__(self, embeddings: np.ndarray, labels: list[str], sources: list[str] | None = None):
        self.embeddings = embeddings.astype(np.float32)
        self.labels = np.array(labels)
        self.sources = np.array(sources if sources is not None else [""] * len(labels))

    @property
    def names(self) -> list[str]:
        return sorted(set(self.labels.tolist()))

    def save(self, path: str = C.LANDMARK_INDEX):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, embeddings=self.embeddings, labels=self.labels, sources=self.sources)

    @classmethod
    def load(cls, path: str = C.LANDMARK_INDEX) -> "LandmarkIndex":
        d = np.load(path, allow_pickle=False)
        return cls(d["embeddings"], d["labels"].tolist(), d["sources"].tolist())

    def scores(self, query: np.ndarray, exclude_source: str | None = None) -> dict[str, float]:
        """Per-landmark score = mean of its top-k cosine similarities to the query."""
        sims = self.embeddings @ query
        out = {}
        for name in self.names:
            m = self.labels == name
            if exclude_source is not None:
                m &= self.sources != exclude_source
            if m.any():
                s = np.sort(sims[m])[::-1][:C.LANDMARK_TOPK]
                out[name] = float(s.mean())
        return out

    def resolve(self, query: np.ndarray, exclude_source: str | None = None) -> tuple[str | None, float]:
        s = self.scores(query, exclude_source)
        if not s:
            return None, 0.0
        ranked = sorted(s.items(), key=lambda kv: kv[1], reverse=True)
        best, best_s = ranked[0]
        margin = best_s - ranked[1][1] if len(ranked) > 1 else 1.0
        if best_s < C.LANDMARK_MIN_SIM or margin < C.LANDMARK_MIN_MARGIN:
            return None, best_s
        return best, best_s


class ZeroShotPlaces:
    """
    Fallback when no home-tour enrollment exists: CLIP picks the best-matching generic place
    description (config.PLACE_PROMPTS). Needs no data, but only knows generic surfaces
    ("desk"), not "my desk vs. my roommate's desk". Enrollment is more accurate.
    """

    def __init__(self, embedder: ClipEmbedder, prompts: dict[str, str] = C.PLACE_PROMPTS):
        self.names = list(prompts)          # includes "_..." rejection options
        self.text = embedder.embed_text([prompts[n] for n in self.names])
        self.scale = embedder.logit_scale

    def probs(self, query: np.ndarray) -> dict[str, float]:
        logits = self.scale * (self.text @ query)
        p = np.exp(logits - logits.max())
        p /= p.sum()
        return dict(zip(self.names, p.tolist()))

    def resolve(self, query: np.ndarray) -> tuple[str | None, float]:
        p = self.probs(query)
        best = max(p, key=p.get)
        if best.startswith("_") or p[best] < C.PLACE_MIN_PROB:     # rejection option won / unsure
            return None, p[best]
        return best, p[best]


def build_index(embedder: ClipEmbedder, enroll_dir: str = C.ENROLL_DIR) -> LandmarkIndex:
    sources = collect_enrollment(enroll_dir)
    embs, labels, srcs = [], [], []
    for sid, (name, frames) in sources.items():
        e = embedder.embed(frames)
        embs.append(e)
        labels += [name] * len(e)
        srcs += [sid] * len(e)
    if not embs:
        raise ValueError(f"No enrollment videos/photos found under {enroll_dir}")
    return LandmarkIndex(np.concatenate(embs), labels, srcs)


def check_index(index: LandmarkIndex) -> dict:
    """
    Leave-one-source-out check: each enrollment video/photo is resolved against the index
    built from all OTHER sources. Tells you whether landmarks are separable and how to set
    LANDMARK_MIN_SIM, without needing extra test footage.
    """
    per_source = defaultdict(list)
    for i, sid in enumerate(index.sources.tolist()):
        per_source[sid].append(i)
    correct = unknown = total = 0
    same_sims, best_sims = [], []
    for sid, idxs in per_source.items():
        true = index.labels[idxs[0]]
        for i in idxs:
            sc = index.scores(index.embeddings[i], exclude_source=sid)
            if true in sc:
                same_sims.append(sc[true])
            pred, s = index.resolve(index.embeddings[i], exclude_source=sid)
            best_sims.append(s)
            total += 1
            if pred is None:
                unknown += 1
            elif pred == true:
                correct += 1
    return {
        "frames": total,
        "top1_accuracy": correct / total if total else 0.0,
        "unknown_rate": unknown / total if total else 0.0,
        "true_landmark_sim_p10": float(np.percentile(same_sims, 10)) if same_sims else None,
        "true_landmark_sim_median": float(np.median(same_sims)) if same_sims else None,
        "landmarks": index.names,
        "sources_per_landmark": {n: len({s for s, l in zip(index.sources, index.labels) if l == n})
                                 for n in index.names},
    }
