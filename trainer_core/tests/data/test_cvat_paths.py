"""Tests for CVAT split file path handling (pytest).

CVAT exports sometimes write split files (e.g., train.txt) where image paths
are recorded with a leading "data/" prefix, like "data/images/img1.jpg".
This verifies that `process_single_images` resolves those paths, copies images
into the YOLO train subset, and creates empty label files when they are missing.
"""

from pathlib import Path
import cv2
import numpy as np

from trainer_core.dataprep.dataset_builder import process_single_images


def test_cvat_train_txt_paths(tmp_path: Path):
    # Simulate a CVAT export structure under raw_data/train/source
    input_root = tmp_path / "raw_data" / "train" / "source_cvat"
    (input_root / "images").mkdir(parents=True, exist_ok=True)
    (input_root / "labels").mkdir(parents=True, exist_ok=True)

    # Create two images
    img1 = np.full((64, 64, 3), 128, dtype=np.uint8)
    img2 = np.full((64, 64, 3), 200, dtype=np.uint8)
    cv2.imwrite(str(input_root / "images" / "img1.jpg"), img1)
    cv2.imwrite(str(input_root / "images" / "img2.jpg"), img2)

    # CVAT-style train.txt listing with extra 'data' prefix
    train_txt = input_root / "train.txt"
    train_txt.write_text("\n".join([
        "data/images/img1.jpg",
        "data/images/img2.jpg",
    ]))

    # Output dataset folders
    train_out = tmp_path / "dataset" / "train"
    test_out = tmp_path / "dataset" / "test"
    (train_out / "train" / "images").mkdir(parents=True, exist_ok=True)
    (train_out / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (train_out / "val" / "images").mkdir(parents=True, exist_ok=True)
    (train_out / "val" / "labels").mkdir(parents=True, exist_ok=True)
    (test_out / "val" / "images").mkdir(parents=True, exist_ok=True)
    (test_out / "val" / "labels").mkdir(parents=True, exist_ok=True)

    train_count, val_count, test_count = process_single_images(
        input_path=input_root.parent,  # one level up to include source_cvat
        train_output_path=train_out,
        test_output_path=test_out,
        val_split=0.0,
        test_split=0.0,
        augment_multiplier=1,
    )

    assert train_count == 2
    assert val_count == 0
    assert test_count == 0

    # Check images and labels copied
    copied_imgs = list((train_out / "train" / "images").glob("*.jpg"))
    copied_lbls = list((train_out / "train" / "labels").glob("*.txt"))
    assert len(copied_imgs) == 2
    assert len(copied_lbls) == 2

    # Ensure label files exist and are possibly empty
    for img_path in copied_imgs:
        lbl = (train_out / "train" / "labels" / img_path.with_suffix(".txt").name)
        assert lbl.exists()
