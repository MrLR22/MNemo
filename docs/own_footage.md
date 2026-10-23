# Using your own chest-camera footage

This guide covers the CPU **pick-up / put-down mode** (`python mnemo.py run`). It tracks a few personal
objects (keys, wallet, phone, pill bottle) and remembers the place where each one was put down. Everything
here is optional: the included `models/hoi_tcn.pt` and the zero-shot object and place models work without
any of your own data, but accuracy in your home will be better with it.

## 1. Mount the camera
Mount a phone or action camera on your chest, pointing slightly down, so both hands and the table in front of
you are in view. Use the same mount every time.

## 2. Objects: fine-tune YOLO (optional)
Without training, `OBJECT_BACKEND = "auto"` in `config.py` uses YOLO-World, which finds objects from their
names. To fine-tune instead:

1. Take about 150 photos per object with the chest camera: on tables, in your hand, partly covered by fingers,
   in different rooms and lighting.
2. Label them in Roboflow or CVAT with the class names `keys, wallet, phone, pill_bottle` and export as YOLOv8.
3. Train:
   ```bash
   python -m detect.train_yolo --data <export>/data.yaml          # add --device 0 on a GPU
   ```
   The result is saved as `models/mnemo_objects.pt` and is picked up automatically.

To track other objects, pass `--objects cup plate lid` to `mnemo.py run`.

## 3. Places: enroll landmarks
For each place (5–8 of them, e.g. `entryway_table`, `desk`, `kitchen_counter`), record 2–3 short clips
(5–10 s, from different angles and in different lighting) into `data/enroll/<place_name>/`. Then:
```bash
python mnemo.py enroll --check
```
The check prints leave-one-clip-out accuracy and similarity percentiles. Use them to set `LANDMARK_MIN_SIM`
in `config.py`. A put-down that matches no enrolled place below that similarity is stored as `unknown`.

## 4. Record and label footage
1. Record 30–40 minutes of normal activity with 150 or more pick-ups and put-downs, spread over several
   sessions or days. Put the videos in `data/own/videos/`.
2. Keep 2–3 videos aside as a **test set** and never use them for training or tuning.
3. Label the events:
   ```bash
   python -m tools.label_events data/own/videos/<file>.mp4
   ```
   Keys `1`–`4` choose the object, `p` marks a pick-up and `d` a put-down. Rows go to `data/own/labels.csv`.
   For the test videos, you can also fill in the `landmark` column.

## 5. Baseline first: rules, no training
```bash
python mnemo.py run data/own/videos/<file>.mp4 --show
```
Tune the values marked `TUNE` in `config.py` on the training videos only.

## 6. Fine-tune HOI-TCN on your footage
```bash
python -m model.extract_features own                                   # → data/features/own
python -m model.train --features data/features/own --init models/hoi_tcn.pt --out models/hoi_tcn_own.pt
```
To retrain from scratch on public data, use `colab/mnemo_train.ipynb`, which needs a GPU.

## 7. Evaluate on the test videos
```bash
python -m evaluation.evaluate_pipeline --videos data/own/test_videos --labels data/own/test_labels.csv --mode rules
python -m evaluation.evaluate_pipeline --videos data/own/test_videos --labels data/own/test_labels.csv --mode model
python -m model.evaluate --checkpoint models/hoi_tcn_own.pt --features data/features/own
```
`model.evaluate` also runs the modality ablation, which removes hands, object or camera motion at test time.

## 8. Habit demo
```bash
python mnemo.py simulate --days 14
python mnemo.py habits --simulated --now "2026-10-23 22:30"
```
This history is **simulated**. Label it that way in any report.

## Known limits
- Picking up a pill bottle is only a stand-in for taking the medicine.
- Without fine-tuning, the COCO fallback (`OBJECT_BACKEND = "coco"`) cannot see keys or wallets.
- The privacy guarantee covers stored data. It does not protect against someone inspecting the running
  machine's memory.
- Some Linux setups crash when MediaPipe imports `sounddevice` (PortAudio with JACK/PipeWire).
  `modules/hand_detector.py` replaces that module with an empty stub, because Mnemo uses no audio.
