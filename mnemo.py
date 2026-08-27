import cv2
from modules.video_source import get_frames
from pipeline import MnemoPipeline


def main():
    print("Starting Mnemo Pipeline...")
    pipeline = MnemoPipeline()
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

            # Top action (only when motion detected)
            action_str = ""
            if results["actions"]:
                top = results["actions"][0]
                action_str = f" | Action: {top['action']} ({top['score']:.4f})"

            print(
                f"[{timestamp:.2f}s] Motion: {motion_str} | Objects: {num_objects} | Hands: {num_hands}{hand_details}{action_str}"
            )

            # Full ranked action scores (only when motion detected)
            if results["actions"]:
                for rank, r in enumerate(results["actions"], start=1):
                    print(f"    {rank:2}. [{r['score']:.4f}]  {r['action']}")
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
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Quitting...")
                break
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        cv2.destroyAllWindows()
        print("Pipeline finished.")


if __name__ == "__main__":
    main()
