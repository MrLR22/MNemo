"""
demo_motion.py
run : "python demo_motion.py"
Module 1: Motion / Change Detection Demo
Designed for PPT presentations and screenshots.

Features:
- Visual HUD overlay showing [MOTION DETECTED] / [NO MOTION]
- Displays pixel change count vs threshold
- Side-by-side or picture-in-picture motion mask view
- Clean terminal logs with timestamps
- Controls:
    [SPACE] : Pause/Resume to easily take screenshots
    [S]     : Save high-res screenshot directly to 'screenshots/' folder
    [M]     : Toggle Motion Mask (shows algorithm's detection mask)
    [Q]     : Quit
"""

import os
import cv2
import numpy as np
from config import VIDEO_SOURCE, MOTION_THRESHOLD


class VisualMotionDetector:
    def __init__(self, sensitivity=5000, threshold_limit=MOTION_THRESHOLD):
        self.sensitivity = sensitivity
        self.threshold_limit = threshold_limit
        self.prev_gray = None

    def detect(self, frame):
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.GaussianBlur(current_gray, (21, 21), 0)

        if self.prev_gray is None:
            self.prev_gray = current_gray
            h, w = frame.shape[:2]
            return False, 0, np.zeros((h, w), dtype=np.uint8)

        # 1. Absolute difference between consecutive frames
        frame_delta = cv2.absdiff(self.prev_gray, current_gray)

        # 2. Binary thresholding
        thresh = cv2.threshold(
            frame_delta, self.threshold_limit, 255, cv2.THRESH_BINARY
        )[1]

        # 3. Morphological dilation to fill holes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.dilate(thresh, kernel, iterations=2)

        # 4. Count non-zero (changed) pixels
        non_zero_count = int(cv2.countNonZero(thresh))
        motion_detected = non_zero_count > self.sensitivity

        self.prev_gray = current_gray
        return motion_detected, non_zero_count, thresh


def draw_hud(frame, motion, pixel_count, sensitivity, frame_idx, timestamp, show_mask, mask):
    display = frame.copy()
    h, w = display.shape[:2]

    # Status colours (BGR)
    if motion:
        badge_color = (0, 200, 0)      # Bright Green
        badge_text = "MOTION DETECTED"
        text_color = (255, 255, 255)
    else:
        badge_color = (70, 70, 70)      # Neutral dark gray
        badge_text = "NO MOTION (STATIC)"
        text_color = (200, 200, 200)

    # Top banner bar
    banner_h = 75
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, display, 0.25, 0, display)

    # Status Pill / Badge
    pill_w = 320
    pill_h = 42
    pill_x1 = 20
    pill_y1 = 16
    pill_x2 = pill_x1 + pill_w
    pill_y2 = pill_y1 + pill_h

    cv2.rectangle(display, (pill_x1, pill_y1), (pill_x2, pill_y2), badge_color, -1, cv2.LINE_AA)
    cv2.rectangle(display, (pill_x1, pill_y1), (pill_x2, pill_y2), (255, 255, 255), 2, cv2.LINE_AA)

    # Badge text
    cv2.putText(
        display,
        badge_text,
        (pill_x1 + 18, pill_y1 + 28),
        cv2.FONT_HERSHEY_DUPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Metrics on the right side of top bar
    info_x = pill_x2 + 30
    cv2.putText(
        display,
        f"Frame: #{frame_idx}  |  Time: {timestamp:.2f}s",
        (info_x, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )

    pct = min(100.0, (pixel_count / sensitivity) * 100) if sensitivity > 0 else 0
    cv2.putText(
        display,
        f"Changed Pixels: {pixel_count:,} / {sensitivity:,} ({pct:.0f}%)",
        (info_x, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255) if motion else (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    # Bottom controls bar
    bar_h = 32
    overlay_bot = display.copy()
    cv2.rectangle(overlay_bot, (0, h - bar_h), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay_bot, 0.7, display, 0.3, 0, display)
    cv2.putText(
        display,
        "[SPACE] Pause/Resume  |  [S] Save Screenshot  |  [M] Toggle Mask  |  [Q] Quit",
        (15, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    # Optional Mini Motion Mask in bottom-right corner
    if show_mask and mask is not None:
        mw, mh = int(w * 0.22), int(h * 0.22)
        mini_mask = cv2.resize(mask, (mw, mh))
        mini_bgr = cv2.cvtColor(mini_mask, cv2.COLOR_GRAY2BGR)
        # Add border
        cv2.rectangle(mini_bgr, (0, 0), (mw, mh), (0, 255, 255), 2)
        cv2.putText(mini_bgr, "MOTION MASK", (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        x_offset = w - mw - 15
        y_offset = h - mh - bar_h - 10
        display[y_offset:y_offset+mh, x_offset:x_offset+mw] = mini_bgr

    return display


def main():
    print("=" * 65)
    print("       MNEMO - MODULE 1: MOTION DETECTION DEMO")
    print("=" * 65)
    print(f"Source video : {VIDEO_SOURCE}")
    print("Controls     : [SPACE] Pause | [S] Save Screenshot | [M] Mask | [Q] Quit")
    print("-" * 65)

    os.makedirs("screenshots", exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Error: Could not open video source '{VIDEO_SOURCE}'.")
        print("Tip: If using webcam, change VIDEO_SOURCE = 0 in config.py")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    detector = VisualMotionDetector(sensitivity=5000, threshold_limit=MOTION_THRESHOLD)

    # Setup resizable window
    window_name = "Mnemo - Module 1: Motion Detector (PPT Demo)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

    frame_idx = 0
    paused = False
    show_mask = True
    screenshot_counter = 1

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("\n[INFO] End of video reached.")
                    break

                frame_idx += 1
                timestamp = frame_idx / fps

                # Run motion detection
                motion, pixel_count, mask = detector.detect(frame)

                # Format terminal log
                status_str = "[ MOTION DETECTED ]" if motion else "[    NO MOTION    ]"
                color_code = "\033[92m" if motion else "\033[90m"
                reset_code = "\033[0m"

                print(
                    f"Frame {frame_idx:04d} ({timestamp:6.2f}s) | "
                    f"{color_code}{status_str}{reset_code} | "
                    f"Changed Pixels: {pixel_count:6d} / 5000"
                )

                # Render Visual HUD
                display = draw_hud(
                    frame, motion, pixel_count, detector.sensitivity,
                    frame_idx, timestamp, show_mask, mask
                )

            # Show frame
            cv2.imshow(window_name, display)

            # Handle keys (wait 30ms or wait indefinitely if paused)
            key = cv2.waitKey(0 if paused else 30) & 0xFF

            if key == ord('q'):
                print("\nExiting...")
                break
            elif key == ord(' '):
                paused = not paused
                print(f"\n{'>>> PAUSED: Press [SPACE] to resume or [S] to save screenshot <<<' if paused else '>>> RESUMED <<<'}")
            elif key == ord('m'):
                show_mask = not show_mask
                print(f"[HUD] Mini motion mask overlay: {'ON' if show_mask else 'OFF'}")
            elif key == ord('s'):
                filename = f"screenshots/motion_demo_{screenshot_counter:02d}_{'active' if motion else 'idle'}.png"
                cv2.imwrite(filename, display)
                print(f"\n[SAVED SCREENSHOT] -> {filename}")
                screenshot_counter += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("-" * 65)
        print("Demo closed. Screenshots (if any) are in the 'screenshots/' folder.")
        print("=" * 65)


if __name__ == "__main__":
    main()
