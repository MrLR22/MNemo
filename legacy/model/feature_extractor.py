# model/feature_extractor.py
# Extracts multi-modal features from EPIC-KITCHENS-100 video clips
# using the existing MNemo pipeline modules (CLIP, YOLO, MediaPipe, Optical Flow).
#
# Input:  EPIC-KITCHENS videos + annotation CSV
# Output: .pt file containing feature tensors + labels ready for training
#
# This is a ONE-TIME offline process. Run it once, save features, then train.

import os
import csv
import cv2
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass

# ── MNemo modules (already built) ──────────────────────────────────────────────
from modules.object_detector import ObjectDetector
from modules.hand_detector import HandDetector
from modules.optical_flow import OpticalFlowAnalyzer
from modules.action_classifier import ActionClassifier


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class ExtractionConfig:
    # Paths
    video_dir: str = "data/epic_kitchens/videos"
    annotation_csv: str = "data/epic_kitchens/EPIC_100_train.csv"
    output_path: str = "data/extracted_features.pt"

    # Frame sampling
    frames_per_clip: int = 8          # T: number of frames to sample per action clip
    min_clip_duration_sec: float = 0.5  # skip clips shorter than this

    # Class filtering — use top-N most frequent verbs/nouns
    top_n_verbs: int = 20
    top_n_nouns: int = 30

    # Feature dimensions (must match model/architecture.py)
    clip_dim: int = 512
    obj_dim: int = 80    # COCO classes
    hand_dim: int = 42   # 21 landmarks × 2 (x, y)
    flow_dim: int = 8    # vx, vy, mag, dir_encoded, each as mean + std


# ── Annotation Parser ─────────────────────────────────────────────────────────

@dataclass
class ActionSegment:
    video_id: str
    start_sec: float
    stop_sec: float
    verb: str
    verb_class: int
    noun: str
    noun_class: int


def parse_timestamp(ts: str) -> float:
    """Convert HH:MM:SS.mmm to seconds."""
    parts = ts.split(":")
    h, m = int(parts[0]), int(parts[1])
    s = float(parts[2])
    return h * 3600 + m * 60 + s


def load_annotations(csv_path: str, participants: list[str] | None = None) -> list[ActionSegment]:
    """Load EPIC-KITCHENS-100 annotations from CSV."""
    segments = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Filter by participant if specified
            if participants and row["participant_id"] not in participants:
                continue

            segments.append(ActionSegment(
                video_id=row["video_id"],
                start_sec=parse_timestamp(row["start_timestamp"]),
                stop_sec=parse_timestamp(row["stop_timestamp"]),
                verb=row["verb"],
                verb_class=int(row["verb_class"]),
                noun=row["noun"],
                noun_class=int(row["noun_class"]),
            ))

    print(f"[Annotations] Loaded {len(segments)} action segments")
    return segments


def filter_top_classes(
    segments: list[ActionSegment],
    top_n_verbs: int,
    top_n_nouns: int,
) -> tuple[list[ActionSegment], dict[int, int], dict[int, int]]:
    """
    Keep only segments whose verb and noun are in the top-N most frequent.
    Returns filtered segments + remapping dicts (original_class → new_index).
    """
    from collections import Counter

    verb_counts = Counter(s.verb_class for s in segments)
    noun_counts = Counter(s.noun_class for s in segments)

    top_verbs = {cls for cls, _ in verb_counts.most_common(top_n_verbs)}
    top_nouns = {cls for cls, _ in noun_counts.most_common(top_n_nouns)}

    # Remap to contiguous 0..N-1 indices
    verb_remap = {orig: idx for idx, (orig, _) in enumerate(verb_counts.most_common(top_n_verbs))}
    noun_remap = {orig: idx for idx, (orig, _) in enumerate(noun_counts.most_common(top_n_nouns))}

    filtered = [s for s in segments if s.verb_class in top_verbs and s.noun_class in top_nouns]

    print(f"[Filter] {len(filtered)}/{len(segments)} segments kept "
          f"(top {top_n_verbs} verbs, top {top_n_nouns} nouns)")

    # Log the class names
    verb_names = {}
    noun_names = {}
    for s in segments:
        if s.verb_class in verb_remap:
            verb_names[verb_remap[s.verb_class]] = s.verb
        if s.noun_class in noun_remap:
            noun_names[noun_remap[s.noun_class]] = s.noun
    print(f"  Verbs: {dict(sorted(verb_names.items()))}")
    print(f"  Nouns: {dict(sorted(noun_names.items()))}")

    return filtered, verb_remap, noun_remap


# ── Per-Frame Feature Extraction ───────────────────────────────────────────────

class MultiModalExtractor:
    """Uses existing MNemo modules to extract features from a single frame."""

    def __init__(self, config: ExtractionConfig):
        self.config = config

        # Load models (these are from the existing pipeline)
        print("[Extractor] Loading models...")
        self.action_classifier = ActionClassifier()  # loads CLIP
        self.object_detector = ObjectDetector()       # loads YOLOv8n
        self.hand_detector = HandDetector()            # loads MediaPipe
        self.flow_analyzer = OpticalFlowAnalyzer()
        print("[Extractor] All models loaded.")

    def extract_clip_feature(self, frame: np.ndarray) -> np.ndarray:
        """CLIP ViT-B/32 image embedding. Returns (512,)."""
        return self.action_classifier.get_embedding(frame)

    def extract_object_feature(self, frame: np.ndarray) -> np.ndarray:
        """
        YOLO detections → 80-dim COCO class presence vector.
        Each element = max confidence for that class in this frame.
        """
        vec = np.zeros(self.config.obj_dim, dtype=np.float32)
        detections = self.object_detector.detect(frame)
        for det in detections:
            class_idx = list(self.object_detector.names.values()).index(det["label"]) \
                if det["label"] in self.object_detector.names.values() else -1
            if 0 <= class_idx < self.config.obj_dim:
                vec[class_idx] = max(vec[class_idx], det["confidence"])
        return vec

    def extract_hand_feature(self, frame: np.ndarray) -> np.ndarray:
        """
        MediaPipe hand landmarks → 42-dim vector.
        21 landmarks × 2 (x, y) normalized to [0, 1] by frame dimensions.
        If no hand detected, returns zeros.
        """
        h, w = frame.shape[:2]
        vec = np.zeros(self.config.hand_dim, dtype=np.float32)
        hands = self.hand_detector.detect(frame, draw=False)
        if hands:
            # Use the first detected hand's box to estimate landmark positions
            # For richer features, we re-detect and use the raw landmarks
            hand = hands[0]
            box = hand["box"]
            # Normalize box coordinates
            vec[0] = box[0] / w   # x1
            vec[1] = box[1] / h   # y1
            vec[2] = box[2] / w   # x2
            vec[3] = box[3] / h   # y2
            # Wrist
            vec[4] = hand["wrist"][0] / w
            vec[5] = hand["wrist"][1] / h
            # Index tip
            vec[6] = hand["index_tip"][0] / w
            vec[7] = hand["index_tip"][1] / h
            # Center
            vec[8] = hand["center"][0] / w
            vec[9] = hand["center"][1] / h
            # Hand presence flag
            vec[10] = 1.0
            # Handedness encoding (Left=0, Right=1)
            vec[11] = 1.0 if hand["label"] == "Right" else 0.0

            # If second hand exists
            if len(hands) > 1:
                h2 = hands[1]
                b2 = h2["box"]
                vec[12] = b2[0] / w
                vec[13] = b2[1] / h
                vec[14] = b2[2] / w
                vec[15] = b2[3] / h
                vec[16] = h2["wrist"][0] / w
                vec[17] = h2["wrist"][1] / h
                vec[18] = h2["index_tip"][0] / w
                vec[19] = h2["index_tip"][1] / h
                vec[20] = h2["center"][0] / w
                vec[21] = h2["center"][1] / h
                vec[22] = 1.0  # second hand present
                vec[23] = 1.0 if h2["label"] == "Right" else 0.0

            # Fill remaining dims with hand box area, aspect ratio, etc.
            bw = (box[2] - box[0]) / w
            bh = (box[3] - box[1]) / h
            vec[24] = bw * bh  # hand area ratio
            vec[25] = bw / max(bh, 1e-6)  # aspect ratio

            # Fingertip-to-wrist distance (proxy for hand openness)
            dx = (hand["index_tip"][0] - hand["wrist"][0]) / w
            dy = (hand["index_tip"][1] - hand["wrist"][1]) / h
            vec[26] = np.sqrt(dx**2 + dy**2)

        return vec

    def extract_flow_feature(
        self, prev_frame: np.ndarray | None, curr_frame: np.ndarray, hand_box: tuple | None
    ) -> np.ndarray:
        """
        Optical flow between consecutive frames → 8-dim feature.
        [vx, vy, magnitude, dir_right, dir_left, dir_up, dir_down, stationary]
        """
        vec = np.zeros(self.config.flow_dim, dtype=np.float32)
        if prev_frame is None:
            vec[7] = 1.0  # stationary
            return vec

        result = self.flow_analyzer.analyze(prev_frame, curr_frame, hand_box)
        vec[0] = result["vector_x"]
        vec[1] = result["vector_y"]
        vec[2] = result["magnitude"]

        # Direction one-hot encoding
        direction = result["direction"]
        vec[3] = 1.0 if direction == "right" else 0.0
        vec[4] = 1.0 if direction == "left" else 0.0
        vec[5] = 1.0 if direction == "upward" else 0.0
        vec[6] = 1.0 if direction == "downward" else 0.0
        vec[7] = 1.0 if direction == "stationary" else 0.0

        return vec


# ── Clip-Level Feature Extraction ──────────────────────────────────────────────

def sample_frames_from_clip(
    video_path: str,
    start_sec: float,
    stop_sec: float,
    num_frames: int,
) -> list[np.ndarray]:
    """
    Sample `num_frames` evenly spaced frames from a video segment.
    Returns list of BGR numpy arrays.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return []

    start_frame = int(start_sec * fps)
    stop_frame = int(stop_sec * fps)
    total = stop_frame - start_frame

    if total < num_frames:
        # Not enough frames — sample all available
        indices = list(range(start_frame, stop_frame))
    else:
        # Evenly space num_frames samples
        step = total / num_frames
        indices = [int(start_frame + i * step) for i in range(num_frames)]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    return frames


def extract_clip_features(
    extractor: MultiModalExtractor,
    frames: list[np.ndarray],
    target_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract all 4 modality features for a list of frames.
    Pads/truncates to target_len.
    Returns (clip_feat, obj_feat, hand_feat, flow_feat) each (target_len, dim).
    """
    cfg = extractor.config

    clip_feats = np.zeros((target_len, cfg.clip_dim), dtype=np.float32)
    obj_feats = np.zeros((target_len, cfg.obj_dim), dtype=np.float32)
    hand_feats = np.zeros((target_len, cfg.hand_dim), dtype=np.float32)
    flow_feats = np.zeros((target_len, cfg.flow_dim), dtype=np.float32)

    prev_frame = None
    for i, frame in enumerate(frames[:target_len]):
        clip_feats[i] = extractor.extract_clip_feature(frame)
        obj_feats[i] = extractor.extract_object_feature(frame)
        hand_feats[i] = extractor.extract_hand_feature(frame)

        # Optical flow needs previous frame
        hand_box = None
        hands = extractor.hand_detector.detect(frame, draw=False)
        if hands:
            hand_box = hands[0]["box"]
        flow_feats[i] = extractor.extract_flow_feature(prev_frame, frame, hand_box)

        prev_frame = frame

    return clip_feats, obj_feats, hand_feats, flow_feats


# ── Main Extraction Pipeline ──────────────────────────────────────────────────

def run_extraction(
    config: ExtractionConfig,
    participants: list[str] | None = None,
    max_clips: int | None = None,
):
    """
    Full extraction pipeline:
    1. Load annotations
    2. Filter to top-N verb/noun classes
    3. Extract multi-modal features per clip
    4. Save everything to a .pt file
    """
    # Load and filter annotations
    segments = load_annotations(config.annotation_csv, participants)
    segments, verb_remap, noun_remap = filter_top_classes(
        segments, config.top_n_verbs, config.top_n_nouns
    )

    if max_clips:
        segments = segments[:max_clips]
        print(f"[Extraction] Limited to {max_clips} clips for testing")

    # Initialize feature extractor
    extractor = MultiModalExtractor(config)

    # Storage
    all_clip_feats = []
    all_obj_feats = []
    all_hand_feats = []
    all_flow_feats = []
    all_verb_labels = []
    all_noun_labels = []
    skipped = 0

    # Process each action segment
    for idx, seg in enumerate(segments):
        # Skip short clips
        duration = seg.stop_sec - seg.start_sec
        if duration < config.min_clip_duration_sec:
            skipped += 1
            continue

        # Find video file
        video_path = os.path.join(config.video_dir, f"{seg.video_id}.MP4")
        if not os.path.exists(video_path):
            # Try lowercase
            video_path = os.path.join(config.video_dir, f"{seg.video_id}.mp4")
        if not os.path.exists(video_path):
            skipped += 1
            continue

        # Sample frames
        frames = sample_frames_from_clip(
            video_path, seg.start_sec, seg.stop_sec, config.frames_per_clip
        )
        if len(frames) < 2:
            skipped += 1
            continue

        # Extract features
        clip_f, obj_f, hand_f, flow_f = extract_clip_features(
            extractor, frames, config.frames_per_clip
        )

        all_clip_feats.append(clip_f)
        all_obj_feats.append(obj_f)
        all_hand_feats.append(hand_f)
        all_flow_feats.append(flow_f)
        all_verb_labels.append(verb_remap[seg.verb_class])
        all_noun_labels.append(noun_remap[seg.noun_class])

        if (idx + 1) % 100 == 0:
            print(f"  [{idx + 1}/{len(segments)}] Processed {len(all_clip_feats)} clips, skipped {skipped}")

    # Convert to tensors and save
    dataset = {
        "clip_features": torch.tensor(np.stack(all_clip_feats)),   # (N, T, 512)
        "obj_features": torch.tensor(np.stack(all_obj_feats)),     # (N, T, 80)
        "hand_features": torch.tensor(np.stack(all_hand_feats)),   # (N, T, 42)
        "flow_features": torch.tensor(np.stack(all_flow_feats)),   # (N, T, 8)
        "verb_labels": torch.tensor(all_verb_labels, dtype=torch.long),  # (N,)
        "noun_labels": torch.tensor(all_noun_labels, dtype=torch.long),  # (N,)
        "verb_remap": verb_remap,
        "noun_remap": noun_remap,
        "config": {
            "frames_per_clip": config.frames_per_clip,
            "top_n_verbs": config.top_n_verbs,
            "top_n_nouns": config.top_n_nouns,
        },
    }

    os.makedirs(os.path.dirname(config.output_path) or ".", exist_ok=True)
    torch.save(dataset, config.output_path)
    print(f"\n[DONE] Saved {len(all_clip_feats)} clips to {config.output_path}")
    print(f"  Skipped: {skipped}")
    print(f"  clip_features shape: {dataset['clip_features'].shape}")
    print(f"  File size: {os.path.getsize(config.output_path) / 1e6:.1f} MB")


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract multi-modal features from EPIC-KITCHENS")
    parser.add_argument("--video-dir", type=str, default="data/epic_kitchens/videos",
                        help="Directory containing EPIC-KITCHENS MP4 files")
    parser.add_argument("--annotations", type=str, default="data/epic_kitchens/EPIC_100_train.csv",
                        help="Path to EPIC_100_train.csv")
    parser.add_argument("--output", type=str, default="data/extracted_features.pt",
                        help="Output .pt file path")
    parser.add_argument("--participants", type=str, nargs="+", default=None,
                        help="Filter to specific participants (e.g., P01 P02 P03)")
    parser.add_argument("--max-clips", type=int, default=None,
                        help="Limit number of clips (for testing)")
    parser.add_argument("--frames-per-clip", type=int, default=8,
                        help="Number of frames to sample per action clip")
    args = parser.parse_args()

    config = ExtractionConfig(
        video_dir=args.video_dir,
        annotation_csv=args.annotations,
        output_path=args.output,
        frames_per_clip=args.frames_per_clip,
    )

    run_extraction(config, participants=args.participants, max_clips=args.max_clips)
