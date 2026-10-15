# legacy/ (old prototype, reference only)

These files are the first prototype (Aug–Sep 2026), kept for history and for slides that showed
"Module 1/2/3" demos. They are **not used** by the current system and some imports no longer resolve.

Why they were replaced:
- `modules/action_classifier.py`: CLIP zero-shot on single frames. One frame can't tell pick-up
  from put-down.
- `modules/event_binder.py`: per-frame rules with no memory. It created events even when no hand
  was visible (fallback IoU 0.10 > threshold 0.05).
- `modules/location_detector.py`: generic room/surface prompts instead of your enrolled landmarks.
- `model/` + `colab_train.py`: a 20-verb x 30-noun EPIC-KITCHENS clip classifier. Its kitchen nouns
  don't include keys/wallet, the Colab model's layer names didn't match `architecture.py` (checkpoints
  couldn't load), the Colab and local features differed, and the random clip split leaked videos
  between train and validation.
- `motion_detector.py`, `optical_flow.py`: frame differencing at 1 fps; on a chest camera the
  wearer's own movement triggers it constantly. Replaced by `modules/ego_motion.py`.
