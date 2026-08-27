# config.py
# Central configuration for Mnemo system
# All settings live here. Never hardcode values in other files.

import os

# ── VIDEO SOURCE ──
VIDEO_SOURCE = "data/test_videos/test.mp4"
SAMPLE_FPS = 1
MOTION_THRESHOLD = 25

# ── MODEL PATHS ──
YOLO_MODEL = "yolov8n.pt"
CLIP_MODEL = "ViT-B/32"
OLLAMA_MODEL = "phi3-mini"
MLP_MODEL_PATH = "mlp/trained_mlp.pth"

# ── THRESHOLDS ──
CONFIDENCE_THRESHOLD = 0.60
IOU_THRESHOLD = 0.30
CLIP_SIMILARITY_CHANGE_THRESHOLD = 0.85
FLOW_MAGNITUDE_THRESHOLD = 2.0

# ── STORAGE PATHS ──
CHROMA_PATH = "outputs/mnemo_db"
SQLITE_PATH = "outputs/mnemo.db"

# ── ACTION PROMPTS ──
ACTION_PROMPTS = [
    "a person picking up an object from a surface",
    "a person placing an object on a surface",
    "a person drinking from a bottle or cup",
    "a person eating food",
    "a person opening a door or drawer or container",
    "a person closing a door or drawer or container",
    "a person using a phone or remote",
    "a person washing their hands",
    "a person sitting down or standing up",
    "an empty room with no person or activity",
]

# ── LOCATION PROMPTS ──
ROOM_PROMPTS = [
    "a kitchen with appliances and counters",
    "a bedroom with a bed",
    "a living room with a couch and television",
    "a bathroom with a sink",
    "a hallway or corridor",
]

SURFACE_PROMPTS = [
    "a kitchen counter or dining table surface",
    "a wall hook or coat hanger",
    "a shelf or cabinet or cupboard",
    "the floor",
    "a bed or sofa or couch",
    "inside a bag drawer or container",
]

# ── WINDOW SETTINGS ──
WINDOW_BUFFER_SIZE = 10
