# Mnemo: In-Home Personal Memory Assistant

A chest-worn camera watches for one thing: **your own hand picking up or putting down** your
tracked objects (keys, wallet, phone, pill bottle). On a put-down it works out which household
landmark the object was left at, stores a short **text** note (object, place, time) and discards
the video. Later you ask *"where is my wallet?"*. A habit engine learns routines (e.g. pills at
~9 PM) and warns when one is missed. It's built for people with short-term memory difficulties.

## Architecture

```
video / camera ─► M1 frames (10 fps, 640 px, RAM-only 12 s buffer)
                   │
                   ▼
              M2 perception: MediaPipe hands (own hands only) + YOLOv8n (your 4 objects)
                   │          + camera-motion estimate (sparse optical flow)
                   ▼
              shared 105-d feature vector per frame (modules/features.py)
                   │
       ┌───────────┴────────────┐
       ▼                        ▼
 M3a RuleEngine            M3b HOI-TCN model (causal temporal CNN, 240K params)
 IDLE→HOLDING→IDLE         per-frame P(none / pickup / putdown) → peak picking
       └───────────┬────────────┘
                   ▼  pickup / putdown event + the frame showing where the object rests
              M5 landmark: CLIP ViT-B/32 embedding → kNN vs home-tour index → name or "unknown"
                   ▼
              M6 SQLite (text only) ─► M7 "where is X?"  ─► M8 habits + missed-routine alerts
```
M4 builds the landmark index offline from short enrollment clips. M9 (privacy) is enforced by
tests that fail if the runtime ever writes an image or video.

| Module | File |
|---|---|
| M0 types / config | `modules/types.py`, `config.py` |
| M1 frame source + RAM buffer | `modules/video_source.py` |
| M2 perception | `modules/hand_detector.py`, `modules/object_detector.py`, `modules/ego_motion.py`, `modules/perception.py` |
| M3 interaction engine | `modules/interaction.py` (rules + model decoding), `modules/features.py` |
| Deep model | `model/architecture.py` (HOI-TCN), `model/data.py`, `model/extract_features.py`, `model/train.py`, `model/evaluate.py` |
| M4/M5 landmarks | `modules/landmarks.py` |
| M6 store | `modules/storage.py` |
| M7 query | `query/query_engine.py` |
| M8 habits | `modules/habits.py` |
| M9 privacy | `tests/test_privacy_and_pipeline.py` |
| M10 evaluation | `evaluation/evaluate_pipeline.py`, `model/evaluate.py`, `model/metrics.py` |
| Tools | `tools/label_events.py` (label your videos), `detect/train_yolo.py` (fine-tune YOLO), `colab/train_on_colab.py` |
| Old prototype | `legacy/` (reference only) |

### Why the deep model looks like this
Telling a pick-up from a put-down depends on **how the hand and object move over 1–3 seconds**,
not on what the room looks like. So HOI-TCN reads hand landmarks, the object's position relative
to the hand, and motion with camera movement subtracted, rather than CLIP scene features. That
avoids the "kitchen ⇒ take" shortcut. Its causal convolutions only look at the past, so the same
network runs live frame by frame (tested: streaming output equals full-sequence output).
It is pre-trained on EPIC-KITCHENS-100 `take`/`put` actions, then fine-tuned on your chest-camera
footage. Object identity comes from the fine-tuned YOLO, not from the model.

## Setup
```bash
python -m venv .mnemovenv && source .mnemovenv/bin/activate   # fish: source .mnemovenv/bin/activate.fish
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python -m pytest            # 33 tests, ~1-2 min
```

## Stage 1 + Stage 2: any action, any object (vision-language model, GPU)
```
STAGE 1  video ──(every 4 s, 8 frames)──► Qwen3-VL-4B ──► action log (text only):
         09:14:02 | closes | fridge door | kitchen
STAGE 2  question ──► matching log lines ──► same model answers ONLY from them (with times),
         or "I didn't see that in my records."
```
- Colab (T4): `colab/mnemo_vlm.ipynb` loads the model once, logs a video, prints the
  **real-time factor** (below 1.0 = keeps up with live video) and answers questions.
- Command line: `python mnemo.py log VIDEO`, `python mnemo.py qa "did I close the fridge?" [--llm]`,
  `python mnemo.py activities`. Without `--llm` (no GPU) `qa` lists the matching log rows.
- Model: `VLM_MODEL` in `config.py` (Qwen3-VL-4B fits a 16 GB T4; 2B is faster).
- **Object-state memory** (`VLM_TRACK_STATES`): the VLM also reports fridge/door/drawer/tap/stove/light
  open-closed / on-off; a change between windows becomes a row (`closes | fridge`), marked `[state]`.
- **Interaction trigger** (`modules/hoi_gate.py`, `python mnemo.py log VIDEO --trigger`): HOI-TCN scores
  each 4-s window; the VLM only runs where the hands interact (threshold `GATE_THRESHOLD`, plus one
  window every `GATE_HEARTBEAT_S`).
- **Kaggle (T4): `colab/mnemo_kaggle.ipynb`** runs everything: demo on one clip, then the benchmark
  (`evaluation/benchmark.py`) on random EPIC minutes of the 5 test people:
  - stage 1 log vs. EPIC ground truth (`evaluation/log_accuracy.py`, EPIC's own verb/noun classes)
  - auto-generated yes/no questions incl. a no-memory "ask the VLM directly" baseline
    (`evaluation/qa_benchmark.py`)
  - optionally Experiment A
- Results so far (8 random test minutes, 153 true actions, Qwen3-VL-4B on a T4, prompt v2):
  - Trigger: at the same number of VLM calls, it finds as many or more actions than every-N-th
    window (41% of calls: 24 vs 17 actions).
  - Real-time factor 1.35 every window → 1.03 at threshold 0.6.
  - Stage 2 hallucinates 2-4% vs 10% for asking the VLM directly, but the direct VLM wins on
    1-minute clips (80% vs 64% yes-recall). That is why Experiment C (`evaluation/long_qa.py`)
    tests a whole morning.
- v3 (`VLM_PROMPT_VERSION`):
  - state lines first
  - stop at repeated lines: 27% of v2 rows were repeats, using 32% of VLM time
  - state check: the VLM said "closed" only once in 120 windows, so open/on objects are asked about again
- Negative result: an "any action" head (`model/train_interaction.py`) trained on the busiest EPIC
  minutes (69% action frames) did not beat the pick/put trigger. Idle time is missing from that data.

## Training without personal data (pick/put model)
| Part | How it works | Data |
|---|---|---|
| Hands | MediaPipe (pre-trained) | none |
| Pick-up / put-down | **HOI-TCN, trained on Colab** | Charades-Ego (homes, 70 people), EPIC-KITCHENS-100, Ego4D FHO (when the licence arrives) |
| Which object | YOLO-World open-vocabulary detector (`OBJECT_BACKEND="auto"`) | none (just the names) |
| Which place | CLIP zero-shot generic places (`PLACE_PROMPTS`), with "unknown" rejection options | none |

1. `python -m tools.make_colab_zip`, then upload `mnemo_code.zip` to `My Drive/mnemo/`
2. Open `colab/mnemo_train.ipynb` in Colab (T4 GPU) and run the cells top to bottom
3. Copy `My Drive/mnemo/models/hoi_tcn_public.pt` to `models/hoi_tcn.pt`, then
   run `python mnemo.py run video.mp4 --mode model`

The train/val/test split is by **person** (EPIC participant, Charades subject, Ego4D source video),
so test numbers are always on people the model never saw.

## Step-by-step plan with your own footage (optional, improves accuracy)
1. **Mount the phone on your chest**, camera facing slightly down so both hands and the table
   in front of you are in view. Use the same mount every time.
2. **Object photos → fine-tune YOLO.** About 150 photos per object, taken from the chest camera:
   on tables, in hand, partly covered by fingers, different rooms and lighting. Label them in
   Roboflow/CVAT with the class names `keys, wallet, phone, pill_bottle`, export as YOLOv8, then:
   `python -m detect.train_yolo --data <export>/data.yaml` (or on Colab). Until you do this,
   only `phone` (and a rough `bottle → pill_bottle`) can be detected, because COCO has no keys/wallet class.
3. **Enroll landmarks.** For each place (5–8: `entryway_table`, `desk`, `kitchen_counter`, ...),
   record 2–3 short clips (5–10 s, different angles and lighting) into `data/enroll/<name>/`.
   Then run `python mnemo.py enroll --check`. The check prints leave-one-out accuracy and
   similarity percentiles; use them to set `LANDMARK_MIN_SIM` in `config.py`.
4. **Record real footage**: 30–40 minutes of normal activity, 150+ pick-ups/put-downs in total,
   spread over several sessions/days. Put the videos in `data/own/videos/`. Keep 2–3 videos aside
   as a **test set** that is never used for training or tuning.
5. **Label events**: `python -m tools.label_events data/own/videos/<file>.mp4`
   (1–4 choose the object, `p` = pick-up, `d` = put-down). Optionally fill in the `landmark`
   column of `data/own/labels.csv` for the test videos.
6. **Baseline first (no training):** `python mnemo.py run data/own/videos/<file>.mp4 --show`.
   Tune the `TUNE` values in `config.py` on the training videos only.
7. **Deep model:** EPIC-KITCHENS pre-training on Colab (`colab/train_on_colab.py`), then
   `python -m model.extract_features own` → fine-tune with `python -m model.train ... --init`.
8. **Evaluate on the test videos:**
   `python -m evaluation.evaluate_pipeline --videos ... --labels ... --mode rules` and `--mode model`,
   plus `python -m model.evaluate` (with modality ablation). Report: rules vs model.
9. **Habit demo:** `python mnemo.py simulate --days 14` then
   `python mnemo.py habits --simulated --now "<today> 22:30"`. Label it as **simulated** in the report.

## Everyday use
```bash
python mnemo.py run recording.mp4 --start "2026-10-15 08:30"     # or: run 0 --show (live camera)
python mnemo.py ask "where is my wallet?"
python mnemo.py ask "did I take my pills today?"
python mnemo.py habits
python mnemo.py events
```

## What has been verified and what hasn't
Verified by tests in this repo:
- the state machine on synthetic tracks: pick-up/put-down, brushes and nudges ignored,
  walking (camera motion) not mistaken for a pick-up, keys hidden in the fist
- causal, streaming HOI-TCN; the training pipeline reaches F1 ≥ 0.8 on a held-out synthetic task
- habit engine: 0 errors on 400 random simulated histories
- query answers, SQLite store, EPIC annotation parsing
- privacy: the real pipeline runs with every image-writing function blocked
- CPU speed (this laptop, 640 px): hands ~21 ms + YOLO ~53 ms + camera motion ~3 ms per frame,
  CLIP ~180 ms only per event

**Not yet measured (no real footage exists yet):** accuracy on real chest-camera video, the
`TUNE` thresholds, the own-hand filter, landmark accuracy in your home, and whether EPIC
pre-training helps. These are the experiments for the report. Do not quote numbers before
running them.

Known limits (state them in the report): a pill-bottle pick-up is a proxy for taking medicine;
the privacy guarantee covers stored data, not someone inspecting the running machine's memory;
COCO-only mode cannot see keys or wallets.

### Machine-specific note
On this laptop, importing MediaPipe pulls in `sounddevice`, and PortAudio's JACK/PipeWire
start-up got the process SIGKILLed. `modules/hand_detector.py` stubs `sounddevice` out (Mnemo
uses no audio).
