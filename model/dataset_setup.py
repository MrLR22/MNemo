# model/dataset_setup.py
# Downloads and sets up EPIC-KITCHENS-100 annotations and guides video download.
#
# Usage:
#   python model/dataset_setup.py                    # download annotations only
#   python model/dataset_setup.py --download-videos  # also download P01 P02 P03 videos

import os
import subprocess
import sys


DATA_DIR = "data/epic_kitchens"
ANNOTATIONS_REPO = "https://github.com/epic-kitchens/epic-kitchens-100-annotations.git"
DOWNLOAD_SCRIPTS_REPO = "https://github.com/epic-kitchens/epic-kitchens-download-scripts.git"
DEFAULT_PARTICIPANTS = ["P01", "P02", "P03"]


def setup_annotations():
    """Clone EPIC-KITCHENS-100 annotations repo."""
    ann_dir = os.path.join(DATA_DIR, "annotations")
    if os.path.exists(ann_dir):
        print(f"[Setup] Annotations already exist at {ann_dir}")
        return ann_dir

    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"[Setup] Cloning annotations...")
    subprocess.run(
        ["git", "clone", ANNOTATIONS_REPO, ann_dir],
        check=True,
    )

    # Symlink the main CSV for easy access
    train_csv = os.path.join(ann_dir, "EPIC_100_train.csv")
    link_path = os.path.join(DATA_DIR, "EPIC_100_train.csv")
    if os.path.exists(train_csv) and not os.path.exists(link_path):
        os.symlink(os.path.abspath(train_csv), link_path)
        print(f"  Linked: {link_path} → {train_csv}")

    print(f"[Setup] Annotations ready at {ann_dir}")
    return ann_dir


def setup_download_scripts():
    """Clone the official download scripts."""
    scripts_dir = os.path.join(DATA_DIR, "download_scripts")
    if os.path.exists(scripts_dir):
        print(f"[Setup] Download scripts already exist at {scripts_dir}")
        return scripts_dir

    print(f"[Setup] Cloning download scripts...")
    subprocess.run(
        ["git", "clone", DOWNLOAD_SCRIPTS_REPO, scripts_dir],
        check=True,
    )
    print(f"[Setup] Download scripts ready at {scripts_dir}")
    return scripts_dir


def download_videos(participants: list[str]):
    """Download videos for specified participants using official script."""
    scripts_dir = setup_download_scripts()
    videos_dir = os.path.join(DATA_DIR, "videos")
    os.makedirs(videos_dir, exist_ok=True)

    # The official script expects to be run from its directory
    downloader = os.path.join(scripts_dir, "epic_downloader.py")
    if not os.path.exists(downloader):
        print(f"[ERROR] Download script not found at {downloader}")
        print("  Please download videos manually from:")
        print("  https://epic-kitchens.github.io/")
        print(f"  Place MP4 files in: {videos_dir}/")
        return

    for p in participants:
        print(f"\n[Setup] Downloading videos for {p}...")
        try:
            subprocess.run(
                [
                    sys.executable, downloader,
                    "--output-path", os.path.abspath(videos_dir),
                    "--participants", p,
                    "--extension", "MP4",
                    "--videos",
                ],
                cwd=scripts_dir,
                check=True,
            )
        except subprocess.CalledProcessError:
            print(f"  [WARNING] Failed to download {p} via script.")
            print(f"  Try Academic Torrents: https://academictorrents.com/")
            print(f"  Or manual download from: https://data.bris.ac.uk/data/dataset/")

    print(f"\n[Setup] Videos directory: {videos_dir}/")


def inspect_annotations():
    """Print a summary of the downloaded annotations."""
    import csv
    from collections import Counter

    csv_path = os.path.join(DATA_DIR, "EPIC_100_train.csv")
    if not os.path.exists(csv_path):
        # Try inside annotations subdir
        csv_path = os.path.join(DATA_DIR, "annotations", "EPIC_100_train.csv")

    if not os.path.exists(csv_path):
        print("[Inspect] EPIC_100_train.csv not found. Run setup first.")
        return

    verb_counter = Counter()
    noun_counter = Counter()
    participant_counter = Counter()
    total = 0

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            verb_counter[row["verb"]] += 1
            noun_counter[row["noun"]] += 1
            participant_counter[row["participant_id"]] += 1
            total += 1

    print(f"\n{'='*60}")
    print(f"EPIC-KITCHENS-100 Training Set Summary")
    print(f"{'='*60}")
    print(f"  Total action segments: {total:,}")
    print(f"  Unique verbs: {len(verb_counter)}")
    print(f"  Unique nouns: {len(noun_counter)}")
    print(f"  Participants: {len(participant_counter)}")

    print(f"\n  Top 20 verbs:")
    for verb, count in verb_counter.most_common(20):
        print(f"    {verb:<20} {count:>5} ({count/total:.1%})")

    print(f"\n  Top 20 nouns:")
    for noun, count in noun_counter.most_common(20):
        print(f"    {noun:<20} {count:>5} ({count/total:.1%})")

    # Show counts for default participants
    print(f"\n  Segments per participant (P01-P03):")
    for p in DEFAULT_PARTICIPANTS:
        print(f"    {p}: {participant_counter.get(p, 0):>5}")

    # Estimate for P01+P02+P03 with top-20 verbs and top-30 nouns
    top_verbs = {v for v, _ in verb_counter.most_common(20)}
    top_nouns = {n for n, _ in noun_counter.most_common(30)}
    filtered = 0
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row["participant_id"] in DEFAULT_PARTICIPANTS
                    and row["verb"] in top_verbs
                    and row["noun"] in top_nouns):
                filtered += 1

    print(f"\n  Usable clips (P01-P03, top-20 verbs, top-30 nouns): {filtered:,}")
    print(f"  This is what the feature extractor will process.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Set up EPIC-KITCHENS-100 dataset")
    parser.add_argument("--download-videos", action="store_true",
                        help="Also download video files for P01, P02, P03")
    parser.add_argument("--participants", nargs="+", default=DEFAULT_PARTICIPANTS,
                        help="Which participants to download")
    parser.add_argument("--inspect", action="store_true",
                        help="Print annotation statistics")
    args = parser.parse_args()

    print("[Setup] Setting up EPIC-KITCHENS-100 dataset...")
    print(f"  Data directory: {os.path.abspath(DATA_DIR)}")

    setup_annotations()

    if args.inspect:
        inspect_annotations()

    if args.download_videos:
        download_videos(args.participants)
    else:
        print(f"\n[Setup] To download videos, run:")
        print(f"  python model/dataset_setup.py --download-videos")
        print(f"\n  Or download manually from https://epic-kitchens.github.io/")
        print(f"  and place MP4 files in: {os.path.abspath(os.path.join(DATA_DIR, 'videos'))}/")

    print(f"\n[Setup] Done. Next steps:")
    print(f"  1. Ensure videos are in {DATA_DIR}/videos/")
    print(f"  2. Extract features (same code as the live system):")
    print(f"     python -m model.extract_features epic --videos {DATA_DIR}/videos \\")
    print(f"       --annotations {DATA_DIR}/EPIC_100_train.csv \\")
    print(f"       --verb-classes {DATA_DIR}/annotations/EPIC_100_verb_classes.csv --participants P01 P02 P03")
    print(f"  3. Pre-train HOI-TCN:")
    print(f"     python -m model.train --features data/features/epic --out models/hoi_tcn_epic.pt")


if __name__ == "__main__":
    main()
