"""Tests for handling empty/missing labels and minimal training run.

This suite checks two things:
- Data prep creates empty label files when labels are missing or empty and
  ensures every image has a corresponding label file with correct counts.
- A minimal `train_model` run succeeds even when the training set contains
  a background-only (empty-label) sample.
"""

import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
from typing import NamedTuple

from trainer_core.dataprep.dataset_builder import process_single_images
from trainer_core.dataprep.dataset_yaml import create_dataset_yaml
from trainer_core.backends.yolo import train_model


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


def test_train_model_minimal(tmp_path: Path, source_dataset: SourceDataset):
    """Minimal training run including a background-only (empty-label) sample.

    Smoke-checks that `train_model` runs without error and produces outputs.
    """
    _, images_dir, labels_dir = source_dataset

    # Create the dataset directory that will be used for training
    dataset_dir = tmp_path / "datasets" / "test_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Create the required directory structure for YOLO training
    (dataset_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)

    # Create a proper dataset YAML using the utility function
    create_dataset_yaml(dataset_dir)

    # First populate the dataset directory with some sample data
    # Copy a few images and labels to the train and val directories
    src_img = images_dir / "image1.jpg"
    src_label = labels_dir / "image1.txt"

    # Add one image with objects to train
    shutil.copy(src_img, dataset_dir / "train" / "images" / "image1.jpg")
    shutil.copy(src_label, dataset_dir / "train" / "labels" / "image1.txt")

    # Add one image without objects to train
    empty_img = images_dir / "image2.jpg"
    shutil.copy(empty_img, dataset_dir / "train" / "images" / "image2.jpg")
    # Create empty label file
    with open(dataset_dir / "train" / "labels" / "image2.txt", "w") as f:
        pass

    # Add an image to validation set as well
    shutil.copy(src_img, dataset_dir / "val" / "images" / "image1.jpg")
    shutil.copy(src_label, dataset_dir / "val" / "labels" / "image1.txt")

    # Try a minimal training run
    model, results, train_output_dir = train_model(
        dataset_path=dataset_dir,
        checkpoint="yolov8n.pt",  # Use smallest model
        image_size=320,  # Use small image size for faster test
        batch_size=1,
        experiment_name="test_experiment",
        epochs=1,
    )
    # If we get here, the training ran without error
    assert train_output_dir.exists()
