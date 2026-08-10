import cv2

from config import VIDEO_SOURCE, SAMPLE_FPS


def get_frames(source=None):
    if source is None:
        source = VIDEO_SOURCE
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise ValueError(f"Error : could not open the video file {source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps == 0:
        cap.release()
        raise ValueError("Error : could not read the FPS")

    frame_interval = int(round(fps / SAMPLE_FPS))
    target_frame = 0

    try:
        while target_frame < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            if ret == False:
                break
            timestamp = target_frame / fps
            yield timestamp, frame

            target_frame += frame_interval
    finally:
        cap.release()


if __name__ == '__main__':
    count = 0
    for timestamp, frame in get_frames():
        print(f"[{timestamp}s] frame shape : {frame.shape}")
        count += 1
    print(f"total frames yielded {count}")
