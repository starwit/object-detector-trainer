from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np

from object_detector_trainer.config.loader import load_config
from object_detector_trainer.dataprep.find_duplicates import DuplicateDetector
from object_detector_trainer.dataprep.dataset_builder import create_dataset_from_raw
from object_detector_trainer.dataprep.sampling import resolve_folder_subsets
from object_detector_trainer.dataprep.source_ingest import (
    check_for_test_images,
)

logger = logging.getLogger(__name__)


def _seed_prepare_stage(args) -> None:
    raw_seed = getattr(args, "seed", None)
    if raw_seed is None:
        return

    seed = int(raw_seed)
    random.seed(seed)
    np.random.seed(seed)


def run_prepare_stage(args, config=None) -> Path:
    cfg = config or load_config(getattr(args, "config", "params.yaml"), args=args)
    _seed_prepare_stage(args)

    dataset_name = Path(getattr(args, "dataset_name", None) or cfg.data.dataset_name)
    raw_val_split = getattr(args, "val_split", None)
    val_split = float(cfg.prepare.val_split if raw_val_split is None else raw_val_split)
    recreate_dataset = bool(getattr(args, "recreate_dataset", False))
    raw_augment_multiplier = getattr(args, "augment_multiplier", None)
    augment_multiplier = int(
        cfg.prepare.augment_multiplier if raw_augment_multiplier is None else raw_augment_multiplier
    )

    folder_subsets = resolve_folder_subsets(
        cfg.prepare.folder_subsets,
        getattr(args, "folder_subset", None),
    )

    custom_classes = list(cfg.data.custom_classes or [])
    use_coco_classes = bool(cfg.data.use_coco_classes)
    class_mapping_config = dict(cfg.data.class_mapping or {})

    base_input_path = Path("raw_data")
    train_image_input_path = base_input_path / "train"
    test_image_input_path = base_input_path / "test"

    dataset_path = Path("datasets") / dataset_name
    training_path = dataset_path / "train"
    test_path = dataset_path / "test"

    raw_test_split = getattr(args, "test_split", None)
    configured_test_split = float(
        cfg.prepare.test_split if raw_test_split is None else raw_test_split
    )
    test_data_exists = check_for_test_images(test_image_input_path)
    if test_data_exists:
        if configured_test_split > 0:
            print(
                "raw_data/test contains test images; prepare.test_split is ignored "
                "because the explicit test folder is used."
            )
        test_split = 0.0
    else:
        test_split = configured_test_split

    if not dataset_path.exists() or recreate_dataset:
        total_train_frames, total_val_frames, total_test_frames = create_dataset_from_raw(
            dataset_path=dataset_path,
            training_path=training_path,
            test_path=test_path,
            train_image_input_path=train_image_input_path,
            test_image_input_path=test_image_input_path,
            val_split=val_split,
            test_split=test_split,
            augment_multiplier=augment_multiplier,
            custom_classes=custom_classes,
            use_coco_classes=use_coco_classes,
            folder_subsets=folder_subsets,
            class_mapping_config=class_mapping_config,
            test_data_exists=test_data_exists,
            recreate_dataset=recreate_dataset,
        )
        logger.info("Total training frames: %s", total_train_frames)
        logger.info("Total validation frames: %s", total_val_frames)
        logger.info("Total test frames: %s", total_test_frames)
    else:
        logger.info("Dataset '%s' already exists. Skipping dataset creation.", dataset_name)

    logger.info("Testing for duplicates between train and test folders...")
    detector = DuplicateDetector(phash_threshold=2, ssim_threshold=0.95)
    matches = detector.compare_folders(training_path, test_path)
    detector.print_folder_comparison_results(matches)

    return dataset_path
