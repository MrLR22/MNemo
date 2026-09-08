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
CONFIDENCE_THRESHOLD = 0.20
IOU_THRESHOLD = 0.05
ACTION_THRESHOLD = 0.12
CLIP_SIMILARITY_CHANGE_THRESHOLD = 0.85
FLOW_MAGNITUDE_THRESHOLD = 1.0

# ── STORAGE PATHS ──
CHROMA_PATH = "outputs/mnemo_db"
SQLITE_PATH = "outputs/mnemo.db"

# ── ACTION PROMPTS ──
# "a photo of" prefix is the standard CLIP accuracy trick from the original paper.
# Keep the list focused — too many similar prompts dilute the softmax and kill separation.
ACTION_PROMPTS = [
    "a photo of a person picking up an object",
    "a photo of a person placing down an object",
    "a photo of a person drinking from a cup or bottle",
    "a photo of a person eating food with hands or utensils",
    "a photo of a person cooking at a stove",
    "a photo of a person opening a door",
    "a photo of a person using a phone",
    "a photo of a person typing on a laptop keyboard",
    "a photo of a person reading a book",
    "a photo of a person washing hands under running water",
    "a photo of a person sitting on a chair",
    "a photo of a person standing in a room",
    "a photo of a person walking",
    "a photo of a person cleaning a table",
    "a photo of a person wearing shoes",
    "a photo of a person watching television",
    "a photo of a person brushing teeth in a bathroom",
    "a photo of a person folding clothes",
    "a photo of a person carrying an object while walking",
    "a photo of a person lying down on a bed or sofa",
    "a photo of two people talking to each other",
    "a photo of an empty room with no people",
]

# ── LOCATION PROMPTS ──
ROOM_PROMPTS = [
    "a photo of a kitchen with appliances and counters",
    "a photo of a bedroom with a bed",
    "a photo of a living room with a couch and television",
    "a photo of a bathroom with a sink and mirror",
    "a photo of a hallway or corridor",
    "a photo of an office or study room with a desk",
]

SURFACE_PROMPTS = [
    "a photo of a kitchen counter or dining table",
    "a photo of a wall hook or coat hanger",
    "a photo of a shelf or cabinet",
    "a photo of the floor",
    "a photo of a bed or sofa",
    "a photo of the inside of a bag or drawer",
    "a photo of a desk or work surface",
]

# ── WINDOW SETTINGS ──
# WINDOW_BUFFER_SIZE = 10
