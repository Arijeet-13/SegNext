import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class IDDDataset(Dataset):
    """

    Split file format:
        <relative_image_path>\t<relative_label_path>

    Example:
        train/FOG/rgb/img1.png\ttrain/FOG/gt_labels/mask1.png
    """

    def __init__(self, img_root, label_root, split_file, processor, augmentation=None):
        self.img_root = img_root
        self.label_root = label_root
        self.processor = processor
        # self.augmentation = augmentation  # None for val/test

        with open(split_file, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        self.samples = []
        for line in lines:
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(
                    f"Expected tab-separated image and label paths, got: '{line}'"
                )

            img_rel, lbl_rel = parts

            self.samples.append((
                os.path.join(self.img_root, img_rel),
                os.path.join(self.label_root, lbl_rel),
            ))

        print(f"[IDDDataset] Loaded {len(self.samples)} samples from {split_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        label = np.array(Image.open(lbl_path), dtype=np.int64)

        # Map out-of-range labels to ignore index (255)
        label[label >= 26] = 255

        # if self.augmentation is not None:
        #     image_np = np.array(image)
        #     image_np, label = self.augmentation(image_np, label)
        #     image = Image.fromarray(image_np)

        encoded = self.processor(
            images=image,
            segmentation_maps=label,
            return_tensors="pt",
        )

        pixel_values = encoded["pixel_values"].squeeze(0)
        labels = encoded["labels"].squeeze(0)

        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }
