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
            print(
                f"[{timestamp:.2f}s] Motion: {motion_str} | Objects Found: {num_objects}"
            )
            # Draw bounding boxes if objects were found
            if results["motion"] and num_objects > 0:
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
