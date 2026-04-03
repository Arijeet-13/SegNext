"""
Generate train/val split files for the IDDAW dataset.

Each line in the output file is a tab-separated pair:
    <relative_image_path>\t<relative_label_path>

Paths are relative to data_root (e.g., data/raw/).

Usage:
    python scripts/generate_splits.py --data-root data/raw --output-dir data/splits
"""

import os
import argparse
from pathlib import Path


def generate_split(data_root, split, output_dir):
    """Scan {data_root}/{split}/rgb/ and pair each image with its label."""
    rgb_dir = os.path.join(data_root, split, "rgb")
    lbl_dir = os.path.join(data_root, split, "gt_labels")

    if not os.path.isdir(rgb_dir):
        print(f"Warning: {rgb_dir} not found, skipping {split}")
        return

    pairs = []
    for drive_folder in sorted(os.listdir(rgb_dir)):
        drive_rgb = os.path.join(rgb_dir, drive_folder)
        if not os.path.isdir(drive_rgb):
            continue

        for fname in sorted(os.listdir(drive_rgb)):
            if not fname.endswith("_rgb.png"):
                continue

            base_id = fname.replace("_rgb.png", "")
            label_fname = f"{base_id}_mask.png"
            label_path = os.path.join(lbl_dir, drive_folder, label_fname)

            if not os.path.isfile(label_path):
                print(f"  Missing label: {label_path}")
                continue

            # Relative paths from data_root
            img_rel = f"{split}/rgb/{drive_folder}/{fname}"
            lbl_rel = f"{split}/gt_labels/{drive_folder}/{label_fname}"
            pairs.append(f"{img_rel}\t{lbl_rel}")

    # Write split file
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{split}.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(pairs) + "\n")

    print(f"[{split}] Wrote {len(pairs)} samples to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate IDDAW split files")
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/raw",
        help="Root directory with train/ and val/ subdirectories",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/splits",
        help="Directory to write train.txt and val.txt",
    )
    args = parser.parse_args()

    for split in ["train", "val"]:
        generate_split(args.data_root, split, args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
