"""Tests for handling empty/missing labels during data preparation."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from typing import NamedTuple

from object_detector_trainer.dataprep.dataset_builder import process_single_images


class SourceDataset(NamedTuple):
    train_raw: Path
    images_dir: Path
    labels_dir: Path


@pytest.fixture
def source_dataset(tmp_path: Path) -> SourceDataset:
    raw_data = tmp_path / "raw_data"
    train_raw = raw_data / "train"
    train_raw.mkdir(parents=True, exist_ok=True)

    source = train_raw / "source1"
    images_dir = source / "images"
    labels_dir = source / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Create 5 unique dummy images with different patterns
    num_images = 5
    for i in range(1, num_images + 1):
        img_filename = f"image{i}.jpg"
        img_path = images_dir / img_filename
        # Create a unique pattern for each image
        np.random.seed(i * 100)  # Different seed for each image
        dummy_img = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
        # Add some distinct patterns to make images even more different
        cv2.rectangle(
            dummy_img,
            (50 * i, 50 * i),
            (200 + i * 20, 200 + i * 20),
            (i * 50, 255 - i * 40, i * 30),
            -1,
        )
        cv2.imwrite(str(img_path), dummy_img)

        if i in [1, 4, 5]:
            # Create a valid label file with one bounding box
            label_path = labels_dir / f"image{i}.txt"
            with open(label_path, "w") as f:
                f.write("0 0.5 0.5 0.2 0.2\n")
        elif i == 3:
            # Create an empty label file
            label_path = labels_dir / f"image{i}.txt"
            with open(label_path, "w") as f:
                f.write("")
        # For image2: no label file is created

    return SourceDataset(train_raw=train_raw, images_dir=images_dir, labels_dir=labels_dir)


def test_process_single_images(tmp_path: Path, source_dataset: SourceDataset):
    """Process mixed labels (present/empty/missing) and verify outputs.

    Confirms empty/missing labels are created as empty files, counts match,
    and every image in train has a corresponding label file.
    """
    train_raw, images_dir, labels_dir = source_dataset

    # Create output directories
    processed_output = tmp_path / "processed_dataset"
    train_output_path = processed_output / "train"
    test_output_path = processed_output / "test"

    # Process images with no validation or test split
    train_count, val_count, test_count = process_single_images(
        input_path=train_raw,
        train_output_path=train_output_path,
        test_output_path=test_output_path,
        val_split=0.0,
        test_split=0.0,
        augment_multiplier=1,
    )

    # We expect all 5 images to be processed
    assert train_count == 5
    assert val_count == 0
    assert test_count == 0

    # Check that all images have corresponding label files
    train_images_dir = train_output_path / "train" / "images"
    train_labels_dir = train_output_path / "train" / "labels"
    train_images = list(train_images_dir.glob("*.jpg"))
    train_labels = list(train_labels_dir.glob("*.txt"))
    assert len(train_images) == 5
    assert len(train_labels) == 5

    # Verify content of label files
    for image_file in train_images:
        label_file = train_labels_dir / image_file.with_suffix(".txt").name
        with open(label_file, "r") as f:
            content = f.read().strip()
        if "image2" in image_file.name or "image3" in image_file.name:
            assert content == "", f"Label for {image_file.name} should be empty"
        else:
            assert content != "", f"Label for {image_file.name} should not be empty"
