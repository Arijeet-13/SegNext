"""
MMSeg-style augmentation pipeline using Albumentations for Hugging Face semantic segmentation.
Matches MMSeg train_pipeline: Resize -> RandomCrop -> RandomFlip -> PhotoMetricDistortion -> Pad.
"""

import random
from typing import Callable, List, Tuple, Union

import albumentations as A
import numpy as np


# ImageNet normalization (matches MMSeg img_norm_cfg with to_rgb=True)
IMAGENET_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMAGENET_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


class Compose:
    """Compose multiple transforms. Each transform receives (image, label) and returns (image, label)."""

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(
        self, image: np.ndarray, label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        for t in self.transforms:
            image, label = t(image, label)
        return image, label


class AlbumentationsWrapper:
    """Wrap Albumentations transform to signature (image, label) -> (image, label)."""

    def __init__(self, aug: A.BasicTransform):
        self.aug = aug

    def __call__(
        self, image: np.ndarray, label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        out = self.aug(image=image, mask=label)
        return out["image"], out["mask"]


class RandomCropWithMaxRatio:
    """
    Random crop with cat_max_ratio (MMSeg-style).
    Max ratio a single class can occupy in the crop.
    """

    def __init__(
        self,
        crop_size: Union[int, Tuple[int, int]],
        cat_max_ratio: float = 1.0,
        ignore_index: int = 255,
    ):
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        self.crop_size = crop_size  # (H, W)
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index

    def _crop(self, img: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        y1, y2, x1, x2 = bbox
        return img[y1:y2, x1:x2, ...].copy()

    def _get_crop_bbox(self, h: int, w: int) -> Tuple[int, int, int, int]:
        ch, cw = self.crop_size
        margin_h = max(h - ch, 0)
        margin_w = max(w - cw, 0)
        offset_h = random.randint(0, margin_h + 1)
        offset_w = random.randint(0, margin_w + 1)
        return offset_h, offset_h + ch, offset_w, offset_w + cw

    def __call__(
        self, image: np.ndarray, label: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        h, w = image.shape[:2]
        ch, cw = self.crop_size

        if h <= ch and w <= cw:
            return image, label

        for _ in range(10):
            bbox = self._get_crop_bbox(h, w)
            cropped_label = self._crop(label, bbox)

            if self.cat_max_ratio >= 1.0:
                break

            labels, cnt = np.unique(cropped_label, return_counts=True)
            cnt = cnt[labels != self.ignore_index]
            if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < self.cat_max_ratio:
                break

        cropped_image = self._crop(image, bbox)
        return cropped_image, cropped_label


def get_train_augmentation(
    resize_size: Tuple[int, int] = (1024, 768),
    crop_size: Tuple[int, int] = (512, 512),
    cat_max_ratio: float = 0.75,
    flip_prob: float = 0.5,
    ignore_index: int = 255,
) -> Compose:
    """
    Build training augmentation pipeline matching MMSeg IDD config using Albumentations.
    Order: Resize -> RandomCrop -> RandomFlip -> PhotoMetricDistortion -> Pad
    """
    w, h = resize_size
    ch, cw = crop_size if isinstance(crop_size, tuple) else (crop_size, crop_size)

    return Compose(
        [
            AlbumentationsWrapper(A.Resize(height=h, width=w)),
            RandomCropWithMaxRatio(
                crop_size=(ch, cw),
                cat_max_ratio=cat_max_ratio,
                ignore_index=ignore_index,
            ),
            AlbumentationsWrapper(A.HorizontalFlip(p=flip_prob)),
            AlbumentationsWrapper(
                A.Compose(
                    [
                        A.RandomBrightnessContrast(
                            brightness_limit=32 / 255.0,
                            contrast_limit=(0.5, 1.5),
                            p=0.5,
                        ),
                        A.HueSaturationValue(
                            hue_shift_limit=18,
                            sat_shift_limit=50,  # ~MMSeg saturation_range (0.5, 1.5)
                            val_shift_limit=0,
                            p=0.5,
                        ),
                    ]
                )
            ),
            AlbumentationsWrapper(
                A.PadIfNeeded(
                    min_height=ch,
                    min_width=cw,
                    border_mode=0,
                    value=0,
                    mask_value=ignore_index,
                )
            ),
        ]
    )
