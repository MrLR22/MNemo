"""
demo_module3.py
Module 3: Understanding, Binding and Event Construction

Purpose:
Demonstrates taking the 3 parallel signals (YOLOv8, Optical Flow, CLIP)
and fusing them into a unified EventRecord:
1. Hand-Object Spatial Interaction (IoU Calculation)
2. Confidence Fusion Equation (CLIP 50% + IoU 30% + Flow 20%)
3. Semantic Filtering (Ignore background objects like chairs/walls)
4. Event Construction (Immutable EventRecord: Action + Object + Location)
"""

import cv2
from config import VIDEO_SOURCE, IOU_THRESHOLD, CONFIDENCE_THRESHOLD
from modules.video_source import get_frames
from modules.motion_detector import MotionDetector
from modules.object_detector import ObjectDetector
from modules.hand_detector import HandDetector
from modules.optical_flow import OpticalFlowAnalyzer
from modules.action_classifier import ActionClassifier
from modules.location_detector import LocationDetector
from modules.event_binder import EventBinder, _iou, _SKIP_LABELS


def clean(text: str) -> str:
    for p in ["a photo of a person ", "a photo of an ", "a photo of a ", "a photo of "]:
        if text.lower().startswith(p):
            return text[len(p):]
    return text


def main():
    print("=" * 75)
    print("  MODULE 3: UNDERSTANDING, BINDING AND EVENT CONSTRUCTION")
    print("=" * 75)
    print("Initializing perception models & EventBinder...")

    motion_det = MotionDetector()
    obj_det = ObjectDetector()
    hand_det = HandDetector()
    flow_analyzer = OpticalFlowAnalyzer()
    action_clf = ActionClassifier()
    loc_det = LocationDetector(model=action_clf.model, preprocess=action_clf.preprocess)
    binder = EventBinder()

    print("\nModels loaded.")
    print("Controls: [SPACE] Pause/Resume | [Q] Quit\n")

    window_name = "Module 3: Understanding & Event Construction Preview"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    prev_frame = None
    prev_timestamp = None
    paused = False

    try:
        for timestamp, frame in get_frames():
            if not motion_det.is_active(frame):
                prev_frame = frame.copy()
                prev_timestamp = timestamp
                continue

            # Extract signals
            objects = obj_det.detect(frame)
            hands = hand_det.detect(frame)

            flow_result = {"direction": "stationary", "magnitude": 0.0, "vector_x": 0.0, "vector_y": 0.0}
            if prev_frame is not None:
                for hand in hands:
                    hand["flow"] = flow_analyzer.analyze(prev_frame, frame, hand["box"])
                if hands and any("flow" in h for h in hands):
                    flow_result = max(
                        (h["flow"] for h in hands if "flow" in h),
                        key=lambda f: f["magnitude"],
                        default=flow_result
                    )

            actions = action_clf.classify(frame)
            room = loc_det.detect_room(frame)
            surface = loc_det.detect_surface(frame, objects[0]["box"] if objects else None)

            # ── MODULE 3: BINDING & EVENT CONSTRUCTION ──────────────────────
            # 1. Run EventBinder (Fuses signals into EventRecord)
            event = binder.bind(
                yolo_detections=objects,
                hand_detections=hands,
                flow_result=flow_result,
                action_scores=actions,
                location_result={"room": room, "surface": surface},
                timestamp_start=prev_timestamp or timestamp,
                timestamp_end=timestamp,
                video_source=VIDEO_SOURCE,
            )

            # Spatial IoU & candidate calculation for display
            candidate_objs = [obj for obj in objects if obj["label"] not in _SKIP_LABELS]
            best_obj, best_iou = None, 0.0
            if hands and candidate_objs:
                for hand in hands:
                    for obj in candidate_objs:
                        iou = _iou(hand["box"], obj["box"])
                        if iou > best_iou:
                            best_iou = iou
                            best_obj = obj
                if (best_obj is None or best_iou < IOU_THRESHOLD):
                    for hand in hands:
                        hx, hy = hand.get("center", ((hand["box"][0] + hand["box"][2])//2, (hand["box"][1] + hand["box"][3])//2))
                        for obj in candidate_objs:
                            ox, oy = obj.get("center", ((obj["box"][0] + obj["box"][2])//2, (obj["box"][1] + obj["box"][3])//2))
                            dist = ((hx - ox)**2 + (hy - oy)**2)**0.5
                            if dist < 120:
                                best_obj = obj
                                best_iou = max(best_iou, 0.15)
                                break
            elif candidate_objs:
                best_obj = candidate_objs[0]
                best_iou = 0.10

            # Terminal printout formatted specifically for PPT explanation
            top_action = actions[0] if actions else {"action": "None", "score": 0.0}
            flow_score = min(flow_result["magnitude"] / 5.0, 1.0)
            combined_score = round((top_action["score"] * 0.5) + (min(best_iou, 1.0) * 0.3) + (flow_score * 0.2), 3)

            print(f"\n{'━'*75}")
            print(f" [Time: {timestamp:5.2f}s]  MODULE 3: UNDERSTANDING & FUSION CALCULATION")
            print(f"{'━'*75}")
            print(f"  Step 1: Hand-Object Spatial Interaction (IoU / Proximity)")
            if best_obj:
                print(f"    • Interacted Object : Hand <-> '{best_obj['label']}' (Overlap IoU = {best_iou:.3f} | Min = {IOU_THRESHOLD})")
            else:
                print(f"    • Interacted Object : None (No target object in hand area)")

            print(f"  Step 2: Multi-Signal Fusion Formula")
            print(f"    • CLIP Action Score (50%) : {top_action['score']:.3f} * 0.50 = {top_action['score']*0.5:.3f}")
            print(f"    • Hand-Object IoU   (30%) : {min(best_iou, 1.0):.3f} * 0.30 = {min(best_iou, 1.0)*0.3:.3f}")
            print(f"    • Flow Magnitude    (20%) : {flow_score:.3f} * 0.20 = {flow_score*0.2:.3f}")
            print(f"    • Combined Confidence     : {combined_score:.3f} (Threshold = {CONFIDENCE_THRESHOLD})")

            print(f"  Step 3: Event Construction Output")
            if event:
                print(f"    ✅ [EVENT CONSTRUCTED]")
                print(f"       Action   : {clean(event.action)}")
                print(f"       Object   : {event.object_label}")
                print(f"       Location : {clean(event.location_room)} -> {clean(event.location_surface)}")
                print(f"       Duration : {event.timestamp_start:.2f}s to {event.timestamp_end:.2f}s")
                print(f"       Flow Dir : {event.flow_direction}")
            else:
                print(f"    ⏳ [NO EVENT BOUND] Signals did not satisfy interaction/confidence thresholds.")
            print(f"{'━'*75}")

            # Draw visual boxes on preview
            for o in objects:
                x1, y1, x2, y2 = o["box"]
                is_bound = event and event.object_label == o["label"]
                color = (0, 0, 255) if is_bound else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{o['label']} (IoU: {best_iou:.2f})" if is_bound else o["label"],
                            (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            for h in hands:
                x1, y1, x2, y2 = h["box"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Event Banner at top
            h_f, w_f = frame.shape[:2]
            banner_bg = (0, 180, 0) if event else (40, 40, 40)
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w_f, 48), banner_bg, -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            banner_text = f"EVENT: {clean(event.action)} | Obj: {event.object_label} | Conf: {event.confidence:.2f}" if event else "ANALYZING INTERACTION..."
            cv2.putText(frame, banner_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            cv2.imshow(window_name, frame)

            prev_frame = frame.copy()
            prev_timestamp = timestamp

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
        print("\nModule 3 Demo finished.")


if __name__ == "__main__":
    main()
