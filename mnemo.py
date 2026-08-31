import cv2
import pprint
from modules.video_source import get_frames
from pipeline import MnemoPipeline


def main():
    print("Starting Mnemo Pipeline...")
    print("Controls: [SPACE] Pause/Resume  |  [Q] Quit")
    pipeline = MnemoPipeline()
    paused = False
    try:
        # get_frames() returns a generator yielding (timestamp, frame)
        for timestamp, frame in get_frames():
            # Process frame
            results = pipeline.process_frame(frame, timestamp)
            # Print results to terminal
            motion_str = "YES" if results["motion"] else "NO "
            num_objects = len(results["objects"])
            num_hands = len(results["hands"])
            hand_details = ""
            if num_hands > 0:
                hand_dirs = []
                for h in results["hands"]:
                    d = h.get("flow", {}).get("direction", "unknown")
                    hand_dirs.append(f"{h['label']}: {d}")
                hand_details = f" | Hand Flow: {', '.join(hand_dirs)}"

            # Top 2 actions (only when motion detected)
            action_str = ""
            if results["actions"]:
                top1 = results["actions"][0]
                top2 = results["actions"][1]
                action_str = f" | Actions: 1. {top1['action']} ({top1['score']:.4f}), 2. {top2['action']} ({top2['score']:.4f})"

            # Location info
            location_str = ""
            if results["room"]:
                r = results["room"]
                s = results["surface"]
                location_str = f" | Room: {r['room']} ({r['confidence']:.2f}) | Surface: {s['surface']} ({s['confidence']:.2f})"

            # Only print if motion is detected, to avoid spamming the console on static frames
            if results["motion"]:
                output_dict = {
                    "timestamp": round(timestamp, 2),
                    **results,
                    "actions": results["actions"][:1] if results.get("actions") else [],
                    "event": results["event"].to_dict() if results.get("event") else None,
                }
                pprint.pprint(output_dict, sort_dicts=False)
                if results["event"]:
                    print(f"\n{'━'*60}")
                    print(f"  EVENT BOUND: {results['event']}")
                    print(f"{'━'*60}\n")
            # Draw bounding boxes if objects or hands were found
            if results["motion"]:
                if num_objects > 0:
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
                if num_hands > 0:
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
