"""Tests for handling empty/missing labels during data preparation."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from typing import NamedTuple

from object_detector_trainer.dataprep.dataset_builder import create_dataset_from_raw, process_single_images


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


def _write_labeled_image(images_dir: Path, labels_dir: Path, stem: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 255, (96, 96, 3), dtype=np.uint8)
    cv2.rectangle(image, (8, 8), (40 + seed, 40), (255, 50 + seed, 20), -1)
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(images_dir / f"{stem}.jpg"), image)
    (labels_dir / f"{stem}.txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")


def test_create_dataset_fails_when_validation_split_has_no_frames(tmp_path: Path) -> None:
    train_raw = tmp_path / "raw_data" / "train"
    images_dir = train_raw / "manual" / "images"
    labels_dir = train_raw / "manual" / "labels"
    _write_labeled_image(images_dir, labels_dir, "one_image", 1)

    with pytest.raises(ValueError, match="0 validation frames"):
        create_dataset_from_raw(
            dataset_path=tmp_path / "datasets" / "waste",
            training_path=tmp_path / "datasets" / "waste" / "train",
            test_path=tmp_path / "datasets" / "waste" / "test",
            train_image_input_path=train_raw,
            test_image_input_path=tmp_path / "raw_data" / "test",
            val_split=0.25,
            test_split=0.25,
            augment_multiplier=1,
            custom_classes=["waste"],
            use_coco_classes=False,
            folder_subsets={},
            class_mapping_config={},
            test_data_exists=False,
            recreate_dataset=True,
        )


def test_create_dataset_rejects_splits_that_leave_no_training_frames(tmp_path: Path) -> None:
    train_raw = tmp_path / "raw_data" / "train"
    images_dir = train_raw / "manual" / "images"
    labels_dir = train_raw / "manual" / "labels"
    for index in range(4):
        _write_labeled_image(images_dir, labels_dir, f"image_{index}", index)

    with pytest.raises(ValueError, match=r"val_split \+ prepare.test_split must be < 1"):
        create_dataset_from_raw(
            dataset_path=tmp_path / "datasets" / "waste",
            training_path=tmp_path / "datasets" / "waste" / "train",
            test_path=tmp_path / "datasets" / "waste" / "test",
            train_image_input_path=train_raw,
            test_image_input_path=tmp_path / "raw_data" / "test",
            val_split=0.8,
            test_split=0.2,
            augment_multiplier=1,
            custom_classes=["waste"],
            use_coco_classes=False,
            folder_subsets={},
            class_mapping_config={},
            test_data_exists=False,
            recreate_dataset=True,
        )


def test_create_dataset_fails_when_full_pipeline_has_no_test_frames(tmp_path: Path) -> None:
    train_raw = tmp_path / "raw_data" / "train"
    images_dir = train_raw / "manual" / "images"
    labels_dir = train_raw / "manual" / "labels"
    for index in range(4):
        _write_labeled_image(images_dir, labels_dir, f"image_{index}", index)

    with pytest.raises(ValueError, match="0 test frames"):
        create_dataset_from_raw(
            dataset_path=tmp_path / "datasets" / "waste",
            training_path=tmp_path / "datasets" / "waste" / "train",
            test_path=tmp_path / "datasets" / "waste" / "test",
            train_image_input_path=train_raw,
            test_image_input_path=tmp_path / "raw_data" / "test",
            val_split=0.25,
            test_split=0.0,
            augment_multiplier=1,
            custom_classes=["waste"],
            use_coco_classes=False,
            folder_subsets={},
            class_mapping_config={},
            test_data_exists=False,
            recreate_dataset=True,
        )
