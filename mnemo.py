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
                # Draw a semi-transparent PAUSED banner on the frozen frame
                overlay = frame.copy()
                h, w = overlay.shape[:2]
                cv2.rectangle(overlay, (0, h // 2 - 35), (w, h // 2 + 35), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
                cv2.putText(
                    frame,
                    "PAUSED  [SPACE] to resume  [Q] to quit",
                    (w // 2 - 260, h // 2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    2,
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
