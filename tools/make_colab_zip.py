# tools/make_colab_zip.py
# Package the project code for Google Colab (no `zip` command needed).
#
#   python -m tools.make_colab_zip            → mnemo_code.zip in the project folder
#
# Upload mnemo_code.zip to "My Drive/mnemo/" and open colab/mnemo_train.ipynb in Colab.
# Personal data, the virtualenv, git history and the old prototype are left out.

import os
import zipfile

import config as C

SKIP_DIRS = {".mnemovenv", "data", "legacy", ".git", "outputs", "models", "runs", "__pycache__",
             ".pytest_cache", "checkpoints", "evaluation_results", "screenshots", "weights"}   # weights/: CLIP cache, re-downloaded
SKIP_FILES = {"yolov8s-worldv2.pt", "mnemo_code.zip"}       # YOLO-World downloads itself on Colab


def main(out: str = os.path.join(C.ROOT, "mnemo_code.zip")):
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for dirpath, dirnames, filenames in os.walk(C.ROOT):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for f in filenames:
                if f in SKIP_FILES or f.endswith(".pyc"):
                    continue
                path = os.path.join(dirpath, f)
                z.write(path, os.path.relpath(path, C.ROOT))
                n += 1
    print(f"{out}: {n} files, {os.path.getsize(out) / 1e6:.1f} MB")
    print('Upload it to Google Drive → "My Drive/mnemo/mnemo_code.zip", then open colab/mnemo_train.ipynb in Colab.')


if __name__ == "__main__":
    main()
