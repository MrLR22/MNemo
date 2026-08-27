"""
test_action_classifier.py
Streams VIDEO_SOURCE and prints live CLIP action scores on the terminal.
Scores refresh in-place every CLASSIFY_EVERY frames.

Run from project root:
    python test_action_classifier.py
"""

import sys
import cv2
from modules.action_classifier import ActionClassifier
from config import VIDEO_SOURCE, ACTION_PROMPTS

CLASSIFY_EVERY = 15   # classify once per this many frames (tune to taste)
BAR_WIDTH      = 30   # width of the score bar in the terminal


def score_bar(score: float, width: int = BAR_WIDTH) -> str:
    filled = round(score * width * 10)   # softmax scores are small; scale up
    filled = min(filled, width)
    return "█" * filled + "░" * (width - filled)


def clear_lines(n: int) -> None:
    """Move cursor up n lines and clear them."""
    for _ in range(n):
        sys.stdout.write("\033[F\033[K")


def print_scores(results: list[dict], frame_no: int, fps: float) -> None:
    timestamp = frame_no / fps if fps > 0 else 0
    print(f"  Frame {frame_no:>6}  |  {timestamp:>6.1f}s")
    print(f"  {'Action':<48}  {'Score':>6}  Bar")
    print(f"  {'─' * 48}  {'─' * 6}  {'─' * BAR_WIDTH}")
    for r in results:
        bar   = score_bar(r["score"])
        label = r["action"][:48]
        print(f"  {label:<48}  {r['score']:.4f}  {bar}")


def main() -> None:
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {VIDEO_SOURCE}")

    fps       = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_prompts = len(ACTION_PROMPTS)
    header_lines = 3 + n_prompts   # lines printed by print_scores

    classifier = ActionClassifier()
    print(f"\n  Streaming: {VIDEO_SOURCE}  |  classify every {CLASSIFY_EVERY} frames\n")

    frame_no    = 0
    last_scores = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_no % CLASSIFY_EVERY == 0:
            results = classifier.classify(frame)

            if last_scores is not None:
                clear_lines(header_lines)

            print_scores(results, frame_no, fps)
            last_scores = results

        frame_no += 1

    cap.release()
    print("\n  [done]")


if __name__ == "__main__":
    main()
