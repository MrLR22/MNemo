import cv2
from modules.motion_detector import MotionDetector
from config import VIDEO_SOURCE


def main():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("Cannot open video:", VIDEO_SOURCE)
        return

    detector = MotionDetector(sensitivity=5000, threshold_limit=25)

    for i in range(1000):
        ret, frame = cap.read()
        if not ret:
            print("No more frames or failed to read frame", i)
            break
        motion = detector.is_active(frame)
        print(f"frame {i}: motion={motion}")

    cap.release()


if __name__ == "__main__":
    main()
