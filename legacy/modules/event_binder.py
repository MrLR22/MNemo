# modules/event_binder.py
# Combines YOLO + optical flow + CLIP to bind action → object → location.
# Rule-based fusion now; MLP replaces rules in Milestone 5 (TASK-019).

from config import IOU_THRESHOLD, CONFIDENCE_THRESHOLD, ACTION_THRESHOLD
from modules.event_record import EventRecord

# Labels that are structural/permanent — interacting with them is never
# the *object* of an action (you pick up a bottle, not a wall).
_SKIP_LABELS = frozenset({
    "person", "chair", "couch", "bed", "tv", "monitor", "wall"
})


def _iou(box1: list | tuple, box2: list | tuple) -> float:
    """Intersection-over-Union for two [x1, y1, x2, y2] boxes."""
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


class EventBinder:
    """
    Rule-based signal fusion.

    Inputs (from pipeline):
        yolo_detections   list[dict]  keys: label, box, confidence
        hand_detections   list[dict]  keys: label, box, flow (optional)
        flow_result       dict        keys: direction, magnitude, vector_x, vector_y
        action_scores     list[dict]  keys: action, score  (sorted desc)
        location_result   dict        keys: room (dict), surface (dict)
        timestamp_start   float
        timestamp_end     float
        video_source      str

    Returns EventRecord or None.
    """

    def bind(
        self,
        yolo_detections: list[dict],
        hand_detections: list[dict],
        flow_result: dict,
        action_scores: list[dict],
        location_result: dict,
        timestamp_start: float,
        timestamp_end: float,
        video_source: str = "",
    ) -> EventRecord | None:

        # ── 1. Guard: need objects and action scores ──────────────
        if not yolo_detections or not action_scores:
            return None

        top_action = action_scores[0]

        # Action probability check (must exceed uniform chance among 22 prompts)
        if top_action["score"] < ACTION_THRESHOLD:
            return None

        if "empty room" in top_action["action"].lower():
            return None

        # ── 2. Find the candidate object with highest hand-IoU / proximity ───
        candidate_objs = [obj for obj in yolo_detections if obj["label"] not in _SKIP_LABELS]
        if not candidate_objs:
            return None

        best_obj, best_iou = None, 0.0

        if hand_detections:
            for hand in hand_detections:
                for obj in candidate_objs:
                    iou = _iou(hand["box"], obj["box"])
                    if iou > best_iou:
                        best_iou = iou
                        best_obj = obj

            # If strict IoU is small, check center-distance proximity
            if best_obj is None or best_iou < IOU_THRESHOLD:
                for hand in hand_detections:
                    hx, hy = hand.get("center", ((hand["box"][0] + hand["box"][2])//2, (hand["box"][1] + hand["box"][3])//2))
                    for obj in candidate_objs:
                        ox, oy = obj.get("center", ((obj["box"][0] + obj["box"][2])//2, (obj["box"][1] + obj["box"][3])//2))
                        dist = ((hx - ox)**2 + (hy - oy)**2)**0.5
                        # If hand is within 120 pixels of object center, treat as interaction
                        if dist < 120 and (best_obj is None or dist < 120):
                            best_obj = obj
                            best_iou = max(best_iou, 0.15)
                            break
        else:
            # When hand is briefly occluded / not detected, fall back to top detected object
            best_obj = candidate_objs[0]
            best_iou = 0.10

        if best_obj is None or best_iou < IOU_THRESHOLD:
            return None

        # ── 3. Combined confidence: CLIP 50% + IoU 30% + flow 20% ───────────
        flow_score = min(flow_result.get("magnitude", 0.0) / 5.0, 1.0)
        combined_confidence = round(
            top_action["score"] * 0.5
            + min(best_iou, 1.0) * 0.3
            + flow_score * 0.2,
            3,
        )

        if combined_confidence < CONFIDENCE_THRESHOLD:
            return None

        # ── 4. Unpack location (two separate dicts from LocationDetector) ────
        room_info = location_result.get("room") or {}
        surface_info = location_result.get("surface") or {}

        return EventRecord(
            action=top_action["action"],
            object_label=best_obj["label"],
            location_room=room_info.get("room", "unknown room"),
            location_surface=surface_info.get("surface", "unknown surface"),
            timestamp_start=timestamp_start,
            timestamp_end=timestamp_end,
            confidence=combined_confidence,
            flow_direction=flow_result.get("direction", "unknown"),
            video_source=video_source,
        )
