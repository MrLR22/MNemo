# tools/make_colab_zip.py
# Package the project code for Google Colab (no `zip` command needed).
#
#   python -m tools.make_colab_zip            → mnemo_code.zip in the project folder
#
# Colab: upload mnemo_code.zip to Drive. Kaggle: upload it as (a new version of) the dataset used by
# colab/mnemo_kaggle.ipynb.
# Personal data, the virtualenv, git history and the old prototype are left out.

import os
import zipfile

import config as C

SKIP_DIRS = {".mnemovenv", "data", "legacy", ".git", "outputs", "models", "runs", "__pycache__",
             ".pytest_cache", "checkpoints", "evaluation_results", "screenshots", "weights",
             "mnemo_code", "results"}      # weights/: CLIP cache (re-downloaded); mnemo_code/: an unzipped old copy
SKIP_FILES = {"yolov8s-worldv2.pt", "mnemo_code.zip"}       # YOLO-World downloads itself on Colab
EXTRA_FILES = [os.path.join("models", "hoi_tcn.pt")]       # the trained trigger (1 MB, public data only)


def main(out: str = os.path.join(C.ROOT, "mnemo_code.zip")):
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for dirpath, dirnames, filenames in os.walk(C.ROOT):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for f in filenames:
                if f in SKIP_FILES or f.endswith(".pyc") or f.upper().startswith("PROJECT CONTEXT BRIEF"):
                    continue                                                 # the brief has personal details
                path = os.path.join(dirpath, f)
                z.write(path, os.path.relpath(path, C.ROOT))
                n += 1
        for rel in EXTRA_FILES:
            if os.path.exists(os.path.join(C.ROOT, rel)):
                z.write(os.path.join(C.ROOT, rel), rel)
                n += 1
            else:
                print(f"note: {rel} not found, not included")
    print(f"{out}: {n} files, {os.path.getsize(out) / 1e6:.1f} MB")
    print('Upload it to Google Drive → "My Drive/mnemo/mnemo_code.zip", then open colab/mnemo_train.ipynb in Colab.')


if __name__ == "__main__":
    main()
