"""Tests for augmentation pipeline integration in process_single_images.

Creates a tiny dataset and verifies that using augment_multiplier>1
produces additional training samples and corresponding labels.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from trainer_core.dataprep.dataset_builder import process_single_images

# Silence third-party deprecations emitted during imgaug/imageio usage in this module only.
pytestmark = [
    pytest.mark.filterwarnings(
        # Some libs set stacklevel so the warning appears at the caller (imgaug).
        "ignore:.*Starting with ImageIO v3.*:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:.*`pilmode` is deprecated.*:DeprecationWarning:imageio.plugins.pillow"
    ),
]


def _make_image(p: Path, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
    cv2.imwrite(str(p), img)


def test_augmentation_increases_train_samples(tmp_path: Path) -> None:
    # Layout: raw_data/train/source/images + labels.
    base_train = tmp_path / "raw_data" / "train"
    source_dir = base_train / "source"
    images_dir = source_dir / "images"
    labels_dir = source_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Create 3 images and simple labels.
    n = 3
    for i in range(n):
        img = images_dir / f"img{i}.jpg"
        _make_image(img, seed=42 + i)
        with open(labels_dir / f"img{i}.txt", "w", encoding="utf-8") as f:
            # one box centered with small size, class 0
            f.write("0 0.5 0.5 0.2 0.2\n")

    # Output dirs.
    processed_output = tmp_path / "processed_dataset"
    train_output_path = processed_output / "train"
    test_output_path = processed_output / "test"

    # With augment_multiplier=2 we expect n originals + n augmented.
    train_count, val_count, test_count = process_single_images(
        input_path=base_train,
        train_output_path=train_output_path,
        test_output_path=test_output_path,
        val_split=0.0,
        test_split=0.0,
        augment_multiplier=2,
    )

    assert val_count == 0 and test_count == 0
    assert train_count == n * 2, f"Expected {n*2} train images, got {train_count}"

    # Verify files exist one-to-one in train output.
    out_img_dir = train_output_path / "train" / "images"
    out_lbl_dir = train_output_path / "train" / "labels"
    imgs = list(out_img_dir.glob("*.jpg"))
    lbls = list(out_lbl_dir.glob("*.txt"))
    assert len(imgs) == len(lbls) == n * 2
