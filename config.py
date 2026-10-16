# config.py
# Central configuration for Mnemo. Every tunable value lives here.
# Values marked "TUNE" are starting points: measure them on your own footage.

import os

ROOT = os.path.dirname(os.path.abspath(__file__))


def _p(*parts):
    return os.path.join(ROOT, *parts)


# ── M1: FRAME SOURCE ──
PROCESS_FPS = 10              # frames analysed per second (pick/put needs several frames per second)
PROCESS_WIDTH = 640           # frames are downscaled to this width (YOLO/MediaPipe/CLIP resize smaller anyway)
BUFFER_SECONDS = 12           # RAM-only rolling buffer length (privacy: never written to disk)

# ── TRACKED OBJECTS ──
TARGET_OBJECTS = ["keys", "wallet", "phone", "pill_bottle"]

# Spoken / typed names → canonical object name (used by the query interface)
OBJECT_SYNONYMS = {
    "keys": ["key", "keys", "car keys", "house keys", "keychain"],
    "wallet": ["wallet", "purse"],
    "phone": ["phone", "mobile", "cell phone", "cellphone", "smartphone"],
    "pill_bottle": ["pill", "pills", "pill bottle", "medicine", "medication",
                    "tablet", "tablets", "meds"],
}



def set_objects(names: list[str]):
    """Track a different set of objects at runtime (e.g. cup/plate/lid for an EPIC-KITCHENS demo).
    Names are used directly as YOLO-World prompts ('pill_bottle' → 'pill bottle')."""
    global TARGET_OBJECTS, YOLOWORLD_PROMPTS, OBJECT_SYNONYMS
    TARGET_OBJECTS = list(names)
    YOLOWORLD_PROMPTS = {n: YOLOWORLD_PROMPTS.get(n, n.replace("_", " ")) for n in names}
    OBJECT_SYNONYMS = {n: OBJECT_SYNONYMS.get(n, sorted({n, n.replace("_", " "), n.replace("_", " ") + "s"}))
                       for n in names}


# ── M2: PERCEPTION ──
HAND_MODEL = _p("hand_landmarker.task")
OBJECT_BACKEND = "auto"       # "auto" | "finetuned" | "yoloworld" | "coco"  (see modules/object_detector.py)
FINETUNED_YOLO = _p("models", "mnemo_objects.pt")   # produced by detect/train_yolo.py
COCO_YOLO = _p("yolov8n.pt")
# YOLO-World: open-vocabulary detector, finds objects from text names, needs no training photos.
YOLOWORLD_MODEL = "yolov8s-worldv2.pt"               # downloaded automatically by ultralytics
YOLOWORLD_PROMPTS = {"keys": "keys", "wallet": "wallet", "phone": "mobile phone",
                     "pill_bottle": "pill bottle"}  # TUNE: try other wordings on test images
YOLOWORLD_CONFIDENCE = 0.15   # TUNE: open-vocabulary scores are lower than closed-set YOLO
# Until the fine-tuned model exists, fall back to COCO. COCO has NO keys/wallet class,
# so only phone (and a rough bottle→pill_bottle stand-in) work in fallback mode.
COCO_CLASS_MAP = {"cell phone": "phone", "bottle": "pill_bottle"}
YOLO_CONFIDENCE = 0.35        # TUNE
YOLO_IMGSZ = 640

HAND_MIN_CONFIDENCE = 0.5
# MediaPipe assumes a mirrored (selfie) image when it labels Left/Right.
# A chest camera is not mirrored, so the labels are swapped. Verify on your footage.
FLIP_HANDEDNESS = True
# "Own hand" filter: on a chest camera your own arms enter from the lower part of the
# frame. Hands whose wrist is above this line (0 = top, 1 = bottom) are treated as
# someone else's and ignored. TUNE on your footage.
OWN_HAND_MIN_WRIST_Y = 0.30

# ── M3: INTERACTION ENGINE (rule-based state machine) ──
# Counts are in processed frames (PROCESS_FPS per second). All TUNE.
CONTACT_DIST = 1.0            # object-centre-to-hand distance, in units of hand size, that counts as contact
CONTACT_FRAMES = 3            # min contact frames before a pickup can fire (ignores brushes/nudges)
MOVE_TOGETHER_FRAMES = 3      # frames object and hand must move together
MOVE_MIN_SPEED = 0.008        # min ego-compensated speed (fraction of frame width per frame)
MOVE_DIR_COS = 0.5            # min cosine between hand and object velocity to call it "together"
STILL_SPEED = 0.004           # object speed below this = still
RELEASE_FRAMES = 5            # object still AND not in contact for this many frames = putdown
OCCLUDED_GRASP_FRAMES = 4     # object vanished while in contact and hand moved away = pickup (hidden in grip)
EVENT_COOLDOWN_S = 2.0        # same (object, action) cannot fire twice within this window
EVENT_MODEL_THRESHOLD = 0.5   # model mode: min peak probability for an event. TUNE on validation set

# ── M4/M5: LANDMARKS ──
CLIP_ARCH = "ViT-B-32"
CLIP_PRETRAINED = "openai"
ENROLL_DIR = _p("data", "enroll")
LANDMARK_INDEX = _p("models", "landmark_index.npz")
ENROLL_SAMPLE_FPS = 2
LANDMARK_TOPK = 5
LANDMARK_MIN_SIM = 0.80       # TUNE with `mnemo.py enroll --check`: CLIP image-image similarity is high
                              # even for unrelated scenes (an unseen scene scored 0.84 in our check)
LANDMARK_MIN_MARGIN = 0.01    # TUNE: best landmark must beat the runner-up by this much

# Zero-shot places: used when no home-tour enrollment exists. CLIP compares the put-down frame
# with these text descriptions. Generic surfaces only ("desk"), not "MY desk". TUNE wording.
PLACE_PROMPTS = {
    "desk": "a photo of a desk or study table",
    "bed": "a photo of a bed",
    "shelf": "a photo of a shelf or cupboard",
    "table": "a photo of a dining or side table",
    "kitchen_counter": "a photo of a kitchen counter",
    "floor": "a photo of the floor",
    "bag": "a photo of the inside of a bag",
    "sofa_or_chair": "a photo of a sofa or chair",
    "bathroom_sink": "a photo of a bathroom sink",
    # Rejection options (names starting with "_"): if one of these wins, the place is "unknown".
    # Without them CLIP must pick a surface even for a street (tested: street photo → "floor" 54%).
    "_outdoors": "a photo of a street or outdoors",
    "_person": "a photo of a person's face or body",
    "_blurry": "a blurry photo with motion blur",
    "_hand": "a close-up photo of a hand",
}
PLACE_MIN_PROB = 0.35         # TUNE: below this the place is "unknown"

# ── M6: EVENT STORE ──
DB_PATH = _p("outputs", "mnemo.db")
SIM_DB_PATH = _p("outputs", "mnemo_simulated.db")   # simulated days live in a separate, clearly-named DB

# ── M8: HABITS ──
HABIT_LOOKBACK_DAYS = 14
HABIT_MIN_DAYS = 3            # need at least this many days of history
HABIT_MIN_DAY_FRACTION = 0.6  # event must happen on >=60% of observed days to be a habit
HABIT_CLUSTER_GAP_MIN = 90    # events further apart than this (minutes of day) are different habits
HABIT_MIN_WINDOW_MIN = 45     # habit window is at least +-45 min around the typical time
HABIT_MAX_SPREAD_MIN = 30     # median absolute deviation above this = scattered, not a routine

# ── STAGE 1: VISION-LANGUAGE ACTION LOGGER (any action, any object) ──
# Qwen3-VL-4B: 8.9 GB of weights, fits a 16 GB T4 in float16 (the 8B does not). Swap freely:
# "Qwen/Qwen3-VL-2B-Instruct" (faster), "Qwen/Qwen2.5-VL-3B-Instruct".
VLM_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
VLM_WINDOW_S = 4.0            # each description covers this many seconds
VLM_FRAMES = 8                # frames per window (even number: the model pairs frames)
VLM_FRAME_W = 448             # frames are downscaled to this width before the model
VLM_MAX_NEW_TOKENS = 300      # actions + object states; a cut-off answer is still parsed per item
VLM_TRACK_STATES = True       # also ask for fridge/door/tap/stove/light state → derive opens/closes
VLM_FORMAT = "lines"          # "lines" (v2: short lines, fixed verb list, ~half the tokens) | "json" (v1)
VLM_PROMPT_VERSION = "lines-v3"   # names cached answers/results: json (v1), lines (v2), lines-v3 (states first,
                                  # state check, loop stop)
VLM_MAX_ACTIONS = 8           # at most this many actions per window (stops repetition loops)
STATE_CHECK_MAX_AGE_S = 600   # keep asking about an open/on object for up to 10 min after it was last seen
QA_MAX_LOG_LINES = 60         # log lines given to the language model when answering

# Interaction trigger (modules/hoi_gate.py): HOI-TCN decides which windows reach the VLM.
GATE_THRESHOLD = 0.6          # window score = max over the window of 1 - P(no pick/put). 0.6 = the knee of the
                              # EPIC benchmark curve (8 test minutes): 69% of VLM calls, 93% of actions seen
GATE_HEARTBEAT_S = 20.0       # describe at least one window this often even when nothing happens
GATE_HANGOVER = 0             # describe this many extra windows after a triggered one (did not help on EPIC)
GATE_MOTION = None            # TUNE: hand-speed threshold (hoi_gate.window_motion); None = rule off

# ── DEEP LEARNING MODEL (HOI-TCN) ──
MODEL_CHECKPOINT = _p("models", "hoi_tcn.pt")
FEATURE_DIR = _p("data", "features")
OWN_VIDEO_DIR = _p("data", "own", "videos")
OWN_LABELS_CSV = _p("data", "own", "labels.csv")
