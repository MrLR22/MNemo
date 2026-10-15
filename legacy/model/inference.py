# model/inference.py
# Run the trained EgoActionRecognizer on new egocentric video.
#
# This is the BRIDGE between your model and your friend's storage/retrieval code.
#
# Input:  Any egocentric .mp4 video
# Output: List of detected action events with timestamps, printed as JSON
#
# Each event:
#   {
#     "timestamp_start": 12.5,
#     "timestamp_end": 15.0,
#     "verb": "take",
#     "verb_confidence": 0.87,
#     "noun": "cup",
#     "noun_confidence": 0.73,
#     "location_room": "kitchen",
#     "location_surface": "counter",
#     "hand_detected": true,
#     "flow_direction": "upward"
#   }
#
# Your friend takes this output → stores in SQLite → builds query interface.
#
# Usage:
#   python model/inference.py --video test.mp4 --checkpoint checkpoints/best_model.pt

import os
import json
import argparse
import cv2
import numpy as np
import torch

from model.architecture import EgoActionRecognizer
from model.feature_extractor import MultiModalExtractor, ExtractionConfig
from modules.location_detector import LocationDetector
from modules.action_classifier import ActionClassifier


class EgoInference:
    """
    Runs the full MNemo inference pipeline on a video file.

    Slides a window of T frames across the video, extracts multi-modal features
    per window, runs the trained model, and emits action events.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "auto",
        window_frames: int = 8,
        stride_frames: int = 4,
        sample_fps: float = 2.0,
        verb_threshold: float = 0.3,
        noun_threshold: float = 0.2,
    ):
        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.window_frames = window_frames
        self.stride_frames = stride_frames
        self.sample_fps = sample_fps
        self.verb_threshold = verb_threshold
        self.noun_threshold = noun_threshold

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=self.device)
        embed_dim = checkpoint.get("embed_dim", 128)
        config = checkpoint["config"]
        self.verb_remap = checkpoint["verb_remap"]
        self.noun_remap = checkpoint["noun_remap"]

        # Reverse maps: model index → original class ID/name
        self.verb_id_to_name = {v: k for k, v in self.verb_remap.items()}
        self.noun_id_to_name = {v: k for k, v in self.noun_remap.items()}

        # Load model
        self.model = EgoActionRecognizer(
            num_verbs=config["top_n_verbs"],
            num_nouns=config["top_n_nouns"],
            embed_dim=embed_dim,
            max_clip_frames=config["frames_per_clip"],
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # Feature extractor (loads CLIP, YOLO, MediaPipe, Flow)
        ext_config = ExtractionConfig(frames_per_clip=window_frames)
        self.extractor = MultiModalExtractor(ext_config)

        # Location detector (reuses CLIP from ActionClassifier)
        self.location_detector = LocationDetector(
            model=self.extractor.action_classifier.model,
            preprocess=self.extractor.action_classifier.preprocess,
        )

        print(f"[Inference] Ready on {self.device}")

    def process_video(self, video_path: str) -> list[dict]:
        """
        Process an entire video and return a list of detected action events.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        print(f"[Inference] Video: {video_path}")
        print(f"  FPS: {fps:.1f}, Duration: {duration:.1f}s, Total frames: {total_frames}")

        # Sample frames at target FPS
        frame_interval = max(1, int(fps / self.sample_fps))
        sampled_frames = []
        sampled_timestamps = []

        frame_idx = 0
        while frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            sampled_frames.append(frame)
            sampled_timestamps.append(frame_idx / fps)
            frame_idx += frame_interval

        cap.release()
        print(f"  Sampled {len(sampled_frames)} frames at {self.sample_fps} FPS")

        # Slide window across sampled frames
        events = []
        num_windows = max(1, (len(sampled_frames) - self.window_frames) // self.stride_frames + 1)

        for w in range(num_windows):
            start_idx = w * self.stride_frames
            end_idx = min(start_idx + self.window_frames, len(sampled_frames))
            window_frames = sampled_frames[start_idx:end_idx]

            if len(window_frames) < 2:
                continue

            t_start = sampled_timestamps[start_idx]
            t_end = sampled_timestamps[end_idx - 1]

            # Extract features for this window
            event = self._process_window(window_frames, t_start, t_end, video_path)
            if event is not None:
                events.append(event)

            if (w + 1) % 20 == 0:
                print(f"  Window {w+1}/{num_windows}, events so far: {len(events)}")

        # Deduplicate overlapping events
        events = self._deduplicate(events)

        print(f"[Inference] Detected {len(events)} action events")
        return events

    def _process_window(
        self,
        frames: list[np.ndarray],
        t_start: float,
        t_end: float,
        video_source: str,
    ) -> dict | None:
        """Process one temporal window and return an event dict or None."""
        T = self.window_frames
        cfg = self.extractor.config

        clip_feats = np.zeros((T, cfg.clip_dim), dtype=np.float32)
        obj_feats = np.zeros((T, cfg.obj_dim), dtype=np.float32)
        hand_feats = np.zeros((T, cfg.hand_dim), dtype=np.float32)
        flow_feats = np.zeros((T, cfg.flow_dim), dtype=np.float32)

        hand_detected = False
        last_flow_dir = "stationary"
        prev_frame = None

        for i, frame in enumerate(frames[:T]):
            clip_feats[i] = self.extractor.extract_clip_feature(frame)
            obj_feats[i] = self.extractor.extract_object_feature(frame)
            hand_feats[i] = self.extractor.extract_hand_feature(frame)

            hands = self.extractor.hand_detector.detect(frame, draw=False)
            hand_box = hands[0]["box"] if hands else None
            if hands:
                hand_detected = True

            flow_feats[i] = self.extractor.extract_flow_feature(prev_frame, frame, hand_box)
            if flow_feats[i][2] > 1.0:  # magnitude > threshold
                for d_idx, d_name in enumerate(["right", "left", "upward", "downward"]):
                    if flow_feats[i][3 + d_idx] > 0.5:
                        last_flow_dir = d_name
                        break

            prev_frame = frame

        # Run model
        with torch.no_grad():
            clip_t = torch.tensor(clip_feats).unsqueeze(0).to(self.device)
            obj_t = torch.tensor(obj_feats).unsqueeze(0).to(self.device)
            hand_t = torch.tensor(hand_feats).unsqueeze(0).to(self.device)
            flow_t = torch.tensor(flow_feats).unsqueeze(0).to(self.device)

            verb_logits, noun_logits = self.model(clip_t, obj_t, hand_t, flow_t)
            verb_probs = torch.softmax(verb_logits, dim=1)[0]
            noun_probs = torch.softmax(noun_logits, dim=1)[0]

        verb_conf, verb_idx = verb_probs.max(0)
        noun_conf, noun_idx = noun_probs.max(0)

        verb_conf = float(verb_conf)
        noun_conf = float(noun_conf)

        # Threshold check
        if verb_conf < self.verb_threshold:
            return None

        # Get location from middle frame
        mid_frame = frames[len(frames) // 2]
        room_info = self.location_detector.detect_room(mid_frame)
        surface_info = self.location_detector.detect_surface(mid_frame)

        # Clean up prompt prefixes for readability
        room = room_info.get("room", "unknown")
        surface = surface_info.get("surface", "unknown")
        for prefix in ["a photo of a ", "a photo of an ", "a photo of "]:
            if room.startswith(prefix):
                room = room[len(prefix):]
            if surface.startswith(prefix):
                surface = surface[len(prefix):]

        return {
            "timestamp_start": round(t_start, 2),
            "timestamp_end": round(t_end, 2),
            "verb": str(self.verb_id_to_name.get(int(verb_idx), int(verb_idx))),
            "verb_confidence": round(verb_conf, 3),
            "noun": str(self.noun_id_to_name.get(int(noun_idx), int(noun_idx))),
            "noun_confidence": round(noun_conf, 3),
            "location_room": room,
            "location_surface": surface,
            "hand_detected": hand_detected,
            "flow_direction": last_flow_dir,
            "video_source": os.path.basename(video_source),
        }

    def _deduplicate(self, events: list[dict], overlap_threshold: float = 0.5) -> list[dict]:
        """Remove duplicate events from overlapping windows — keep higher confidence."""
        if not events:
            return events

        # Sort by start time
        events.sort(key=lambda e: e["timestamp_start"])

        deduplicated = [events[0]]
        for event in events[1:]:
            prev = deduplicated[-1]

            # Check temporal overlap
            overlap_start = max(prev["timestamp_start"], event["timestamp_start"])
            overlap_end = min(prev["timestamp_end"], event["timestamp_end"])
            overlap_duration = max(0, overlap_end - overlap_start)

            prev_duration = prev["timestamp_end"] - prev["timestamp_start"]
            if prev_duration > 0 and overlap_duration / prev_duration > overlap_threshold:
                # Same action in overlapping window — keep higher confidence
                if event["verb"] == prev["verb"] and event["noun"] == prev["noun"]:
                    if event["verb_confidence"] > prev["verb_confidence"]:
                        deduplicated[-1] = event
                    continue

            deduplicated.append(event)

        return deduplicated


# ── Entry Point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run action recognition on egocentric video")
    parser.add_argument("--video", type=str, required=True, help="Path to .mp4 video")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--verb-threshold", type=float, default=0.3)
    parser.add_argument("--output", type=str, default=None,
                        help="Save events to JSON file (default: print to stdout)")
    args = parser.parse_args()

    engine = EgoInference(
        checkpoint_path=args.checkpoint,
        device=args.device,
        sample_fps=args.sample_fps,
        verb_threshold=args.verb_threshold,
    )

    events = engine.process_video(args.video)

    # Output
    output_data = {
        "video": args.video,
        "num_events": len(events),
        "events": events,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSaved {len(events)} events to {args.output}")
    else:
        print(f"\n{json.dumps(output_data, indent=2)}")


if __name__ == "__main__":
    main()
