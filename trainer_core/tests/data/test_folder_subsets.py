"""Tests for per-folder subset configuration (under/oversampling).
This suite verifies that `process_single_images` honors the `folder_subsets`
ratios per source folder:
- Subsampling (ratio < 1.0) reduces the number of pairs taken from a folder
- Oversampling (ratio > 1.0) duplicates pairs in-memory (not files on disk)

The images include simple geometric patterns to avoid false-positive duplicate
removal by the smart dedup stage.
"""

from pathlib import Path
import random
import cv2
import numpy as np
import pytest

from trainer_core.dataprep.dataset_builder import process_single_images


pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:.*use of fork\\(\\) may lead to deadlocks in the child.*:DeprecationWarning"
    )
]


def _prepare_sources(base_path: Path) -> Path:
    input_root = base_path / "raw_data" / "train"
    for folder, count, base_val in [("a", 4, 25), ("b", 2, 200)]:
        images = input_root / folder / "images"
        labels = input_root / folder / "labels"
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            img = np.full((64, 64, 3), base_val + i, dtype=np.uint8)
            if folder == "a":
                cv2.rectangle(img, (5 + i, 5 + i), (30 + i, 30 + i), (255, 255, 255), -1)
            else:
                cv2.circle(img, (32, 32), 10 + i, (255, 255, 255), -1)
            name = f"{folder}_img_{i}"
            cv2.imwrite(str(images / f"{name}.jpg"), img)
            with open(labels / f"{name}.txt", "w") as f:
                f.write("0 0.5 0.5 0.2 0.2\n")
    return input_root


def _base_name(name: str) -> str:
    if "__dup" not in name:
        return name
    stem_part, _ = name.split("__dup", 1)
    return f"{stem_part}{Path(name).suffix}"


def test_over_under_sampling(tmp_path: Path):
    input_root = _prepare_sources(tmp_path)

    train_out = tmp_path / "dataset" / "train"
    test_out = tmp_path / "dataset" / "test"

    folder_subsets = {"a": 0.5, "b": 2.0}

    random.seed(0)

    train_count, val_count, test_count = process_single_images(
        input_path=input_root,
        train_output_path=train_out,
        test_output_path=test_out,
        val_split=0.0,
        test_split=0.0,
        augment_multiplier=1,
        folder_subsets=folder_subsets,
    )

    assert val_count == 0
    assert test_count == 0
    assert train_count == 6

    train_images_dir = train_out / "train" / "images"
    train_labels_dir = train_out / "train" / "labels"
    img_paths = sorted(train_images_dir.glob("*.jpg"))
    lbl_paths = sorted(train_labels_dir.glob("*.txt"))

    assert len(img_paths) == len(lbl_paths) == train_count

    names = [p.name for p in img_paths]
    assert len(set(names)) == len(names)

    dup_names = [n for n in names if "__dup" in n]
    assert dup_names

    counts_by_stem = {}
    for name in names:
        key = _base_name(name)
        counts_by_stem[key] = counts_by_stem.get(key, 0) + 1

    b_stems = {
        stem
        for stem in counts_by_stem
        if Path(stem).stem.startswith("b_img_")
    }
    assert b_stems
    for stem in b_stems:
        assert counts_by_stem[stem] == 2

    for stem, count in counts_by_stem.items():
        if Path(stem).stem.startswith("a_img_"):
            assert count == 1


def test_oversampling_does_not_leak_to_validation(tmp_path: Path):
    input_root = _prepare_sources(tmp_path)

    train_out = tmp_path / "dataset_split" / "train"
    test_out = tmp_path / "dataset_split" / "test"

    folder_subsets = {"b": 2.0}

    random.seed(0)

    process_single_images(
        input_path=input_root,
        train_output_path=train_out,
        test_output_path=test_out,
        val_split=0.5,
        test_split=0.0,
        augment_multiplier=1,
        folder_subsets=folder_subsets,
    )

    dup_train_imgs = list((train_out / "train" / "images").glob("*__dup*.jpg"))
    dup_train_lbls = list((train_out / "train" / "labels").glob("*__dup*.txt"))
    assert dup_train_imgs, "Expected duplicated samples in train split"
    assert len(dup_train_imgs) == len(dup_train_lbls)

    dup_val_imgs = list((train_out / "val" / "images").glob("*__dup*.jpg"))
    dup_val_lbls = list((train_out / "val" / "labels").glob("*__dup*.txt"))
    dup_test_imgs = list((test_out / "val" / "images").glob("*__dup*.jpg"))
    dup_test_lbls = list((test_out / "val" / "labels").glob("*__dup*.txt"))

    assert not dup_val_imgs
    assert not dup_val_lbls
    assert not dup_test_imgs
    assert not dup_test_lbls
