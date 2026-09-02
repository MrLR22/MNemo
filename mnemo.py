import cv2
import pprint
from modules.video_source import get_frames
from pipeline import MnemoPipeline


def _clean_prompt(text: str) -> str:
    """Strip standard prefix from CLIP prompts for cleaner presentation."""
    if not text:
        return "None"
    for prefix in ["a photo of a person ", "a photo of an ", "a photo of a ", "a photo of "]:
        if text.lower().startswith(prefix):
            return text[len(prefix):]
    return text


def main():
    print("Starting Mnemo Pipeline...")
    print("Controls: [SPACE] Pause/Resume  |  [Q] Quit")
    pipeline = MnemoPipeline()
    paused = False

    # Setup resizable window so it doesn't fill the entire screen
    cv2.namedWindow("Mnemo System", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Mnemo System", 960, 540)

    try:
        # get_frames() returns a generator yielding (timestamp, frame)
        for timestamp, frame in get_frames():
            # Process frame
            results = pipeline.process_frame(frame, timestamp)

            # ── PRINT 3 SIGNALS (only when motion is detected) ──────────────────
            if results["motion"]:
                # Signal 1: Action (CLIP)
                top_action_raw = results["actions"][0]["action"] if results["actions"] else "None"
                action_score = results["actions"][0]["score"] if results["actions"] else 0.0
                action_clean = _clean_prompt(top_action_raw)

                # Signal 2: Object (YOLO + Hand Interaction)
                if results.get("event"):
                    object_str = f"{results['event'].object_label} (Interacting with hand)"
                elif results["objects"]:
                    top_obj = results["objects"][0]
                    object_str = f"{top_obj['label']} (conf: {top_obj['confidence']:.2f})"
                else:
                    object_str = "None detected"

                # Signal 3: Location (Room & Surface)
                room_raw = results["room"].get("room", "") if results.get("room") else ""
                surface_raw = results["surface"].get("surface", "") if results.get("surface") else ""
                room_clean = _clean_prompt(room_raw)
                surface_clean = _clean_prompt(surface_raw)
                if room_clean and surface_clean:
                    location_str = f"{room_clean} (surface: {surface_clean})"
                elif room_clean:
                    location_str = room_clean
                else:
                    location_str = "Unknown"

                # Hand flow / movement direction
                flow_info = ""
                if results["hands"]:
                    for h in results["hands"]:
                        if "flow" in h and h["flow"].get("direction") and h["flow"]["direction"] != "stationary":
                            flow_info = f" | Hand Flow: {h['flow']['direction']} (mag: {h['flow'].get('magnitude', 0.0):.1f})"
                            break

                print(f"\n{'━'*64}")
                print(f" [Time: {timestamp:5.2f}s]  3 SIGNALS DETECTED:")
                print(f"  1. ACTION   : {action_clean} (conf: {action_score:.2f})")
                print(f"  2. OBJECT   : {object_str}")
                print(f"  3. LOCATION : {location_str}{flow_info}")
                if results.get("event"):
                    print(f"  --------------------------------------------------------------")
                    print(f"  ⭐ EVENT BOUND: {results['event']}")
                print(f"{'━'*64}")
            # Draw bounding boxes if objects or hands were found
            if results["motion"]:
                if results["objects"]:
                    for obj in results["objects"]:
                        x1, y1, x2, y2 = obj["box"]
                        label = obj["label"]
                        conf = obj["confidence"]
                        # Draw rectangle and label
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            frame,
                            f"{label} {conf:.2f}",
                            (x1, max(15, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )
                if results["hands"]:
                    for hand in results["hands"]:
                        x1, y1, x2, y2 = hand["box"]
                        label = hand["label"]
                        flow_text = ""
                        if "flow" in hand:
                            flow_text = f" | {hand['flow']['direction']}"
                        # Draw rectangle and label for hands
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(
                            frame,
                            f"{label} Hand{flow_text}",
                            (x1, max(15, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 0, 0),
                            2,
                        )

            # ── Action subtitle overlay (separate from pause banner) ──────────
            # Shows top predicted action as a subtitle at the bottom of the frame.
            # Remove or comment out this block if you no longer want on-screen actions.
            h_f, w_f = frame.shape[:2]
            subtitle_text = ""
            if results.get("actions"):
                top_action = results["actions"][0]
                subtitle_text = f"Action: {top_action['action']}  ({top_action['score']:.2f})"
            if subtitle_text:
                sub_font       = cv2.FONT_HERSHEY_SIMPLEX
                sub_scale      = 0.55
                sub_thickness  = 1
                (tw, th), _    = cv2.getTextSize(subtitle_text, sub_font, sub_scale, sub_thickness)
                bar_h          = th + 20          # padding above and below text
                bar_y1         = h_f - bar_h - 4  # near the very bottom
                # Translucent dark strip
                sub_overlay    = frame.copy()
                cv2.rectangle(sub_overlay, (0, bar_y1), (w_f, h_f), (0, 0, 0), -1)
                cv2.addWeighted(sub_overlay, 0.6, frame, 0.4, 0, frame)
                # White subtitle text, centred horizontally
                text_x = (w_f - tw) // 2
                text_y = bar_y1 + th + 8
                cv2.putText(
                    frame,
                    subtitle_text,
                    (text_x, text_y),
                    sub_font,
                    sub_scale,
                    (255, 255, 255),
                    sub_thickness,
                    cv2.LINE_AA,
                )
            # ── end subtitle overlay ──────────────────────────────────────────

            # Display frame
            cv2.imshow("Mnemo System", frame)
            # Key handling: SPACE = pause/resume, Q = quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Quitting...")
                break
            elif key == ord(" "):
                paused = not paused
                print("[Mnemo] Paused." if paused else "[Mnemo] Resumed.")

            # Pause loop — freeze the current frame until SPACE is pressed again
            while paused:
                # Draw a small semi-transparent PAUSED hint near the top-centre
                overlay = frame.copy()
                h, w = overlay.shape[:2]
                hint_y1 = int(h * 0.08)   # ~8 % from the top
                hint_y2 = hint_y1 + 28
                cv2.rectangle(overlay, (0, hint_y1), (w, hint_y2), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
                cv2.putText(
                    frame,
                    "PAUSED  |  SPACE = resume   Q = quit",
                    (w // 2 - 195, hint_y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow("Mnemo System", frame)
                pause_key = cv2.waitKey(30) & 0xFF
                if pause_key == ord(" "):
                    paused = False
                    print("[Mnemo] Resumed.")
                elif pause_key == ord("q"):
                    print("Quitting...")
                    paused = False   # exit inner loop
                    raise KeyboardInterrupt  # bubble up to break the outer loop
    except KeyboardInterrupt:
        pass  # clean exit from quit-while-paused path
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        cv2.destroyAllWindows()
        print("Pipeline finished.")


if __name__ == "__main__":
    main()
