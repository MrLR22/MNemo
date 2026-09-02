"""
demo_module2.py
Module 2: Parallel Signal Extraction using YOLOv8, Optical Flow, and CLIP

Purpose:
Demonstrates the independent, parallel extraction of three modalities:
1. Spatial Objects & Hands (YOLOv8 + MediaPipe)
2. Motion & Velocity Dynamics (Farneback Optical Flow)
3. Semantic Context & Action (OpenAI CLIP)

No event binding or fusion is performed here (that belongs to Module 3).
"""

import cv2
import numpy as np
from config import VIDEO_SOURCE, CLIP_MODEL, YOLO_MODEL
from modules.video_source import get_frames
from modules.motion_detector import MotionDetector
from modules.object_detector import ObjectDetector
from modules.hand_detector import HandDetector
from modules.optical_flow import OpticalFlowAnalyzer
from modules.action_classifier import ActionClassifier
from modules.location_detector import LocationDetector


def clean(text: str) -> str:
    for p in ["a photo of a person ", "a photo of an ", "a photo of a ", "a photo of "]:
        if text.lower().startswith(p):
            return text[len(p):]
    return text


def main():
    print("=" * 72)
    print("  MODULE 2: PARALLEL SIGNAL EXTRACTION (YOLOv8 + Optical Flow + CLIP)")
    print("=" * 72)
    print("Initializing perception models...")

    motion_det = MotionDetector()
    obj_det = ObjectDetector()
    hand_det = HandDetector()
    flow_analyzer = OpticalFlowAnalyzer()
    action_clf = ActionClassifier()
    loc_det = LocationDetector(model=action_clf.model, preprocess=action_clf.preprocess)

    print("\nModels loaded successfully.")
    print("Controls: [SPACE] Pause/Resume (to take screenshots) | [Q] Quit\n")

    window_name = "Module 2: Parallel Signal Extraction Preview"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    prev_frame = None
    paused = False

    try:
        for timestamp, frame in get_frames():
            if not motion_det.is_active(frame):
                prev_frame = frame.copy()
                continue

            # ── SIGNAL 1: YOLOv8 (Objects & Hands) ──────────────────────────
            objects = obj_det.detect(frame)
            hands = hand_det.detect(frame)

            # ── SIGNAL 2: Optical Flow (Motion Dynamics) ────────────────────
            flow_result = {"direction": "stationary", "magnitude": 0.0, "vector_x": 0.0, "vector_y": 0.0}
            if prev_frame is not None:
                for hand in hands:
                    flow = flow_analyzer.analyze(prev_frame, frame, hand["box"])
                    hand["flow"] = flow
                if hands and any("flow" in h for h in hands):
                    flow_result = max(
                        (h["flow"] for h in hands if "flow" in h),
                        key=lambda f: f["magnitude"],
                        default=flow_result
                    )
            prev_frame = frame.copy()

            # ── SIGNAL 3: CLIP (Zero-Shot Action & Location) ────────────────
            actions = action_clf.classify(frame)
            room = loc_det.detect_room(frame)
            surface = loc_det.detect_surface(frame, objects[0]["box"] if objects else None)

            # ── PRINT PARALLEL SIGNALS TO TERMINAL (FOR PPT) ───────────────
            top_action = actions[0] if actions else {"action": "None", "score": 0.0}
            obj_summary = ", ".join([f"{o['label']} ({o['confidence']:.2f})" for o in objects[:3]]) or "None"
            hand_summary = ", ".join([f"{h['label']} Hand" for h in hands]) or "None"

            print(f"\n{'━'*72}")
            print(f" [Timestamp: {timestamp:5.2f}s]  MODULE 2: PARALLEL SIGNALS EXTRACTED")
            print(f"{'━'*72}")
            print(f"  [SIGNAL 1: YOLOv8 & MediaPipe - Spatial Perception]")
            print(f"    • Detected Objects : {obj_summary}")
            print(f"    • Detected Hands   : {hand_summary}")
            print(f"")
            print(f"  [SIGNAL 2: Optical Flow - Temporal Motion Dynamics]")
            print(f"    • Hand Flow Direction : {flow_result['direction'].upper()}")
            print(f"    • Flow Velocity Vector: dx={flow_result['vector_x']:+.2f}, dy={flow_result['vector_y']:+.2f}")
            print(f"    • Motion Magnitude    : {flow_result['magnitude']:.2f} px/frame")
            print(f"")
            print(f"  [SIGNAL 3: OpenAI CLIP - Semantic Perception]")
            print(f"    • Top Action Score    : {clean(top_action['action'])} ({top_action['score'] * 100:.1f}%)")
            print(f"    • Predicted Room      : {clean(room.get('room', 'Unknown'))} ({room.get('confidence', 0)*100:.1f}%)")
            print(f"    • Predicted Surface   : {clean(surface.get('surface', 'Unknown'))} ({surface.get('confidence', 0)*100:.1f}%)")
            print(f"{'━'*72}")

            # Draw visual boxes on preview
            for o in objects:
                x1, y1, x2, y2 = o["box"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"YOLO: {o['label']} {o['confidence']:.2f}", (x1, max(18, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            for h in hands:
                x1, y1, x2, y2 = h["box"]
                d = h.get("flow", {}).get("direction", "")
                flow_lbl = f" | Flow: {d}" if d else ""
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{h['label']} Hand{flow_lbl}", (x1, max(18, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Top action banner
            h_f, w_f = frame.shape[:2]
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h_f - 40), (w_f, h_f), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, f"CLIP Action: {clean(top_action['action'])} ({top_action['score']:.2f})",
                        (20, h_f - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
                print("\n>>> PAUSED (Press SPACE to resume) <<<" if paused else "\n>>> RESUMED <<<")
                while paused:
                    k = cv2.waitKey(30) & 0xFF
                    if k == ord(' '):
                        paused = False
                        print(">>> RESUMED <<<")
                    elif k == ord('q'):
                        return

    finally:
        cv2.destroyAllWindows()
        print("\nModule 2 Demo finished.")


if __name__ == "__main__":
    main()
