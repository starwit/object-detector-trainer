from __future__ import annotations

import logging
from pathlib import Path

from trainer_core.config.loader import load_config
from trainer_core.dataprep.find_duplicates import DuplicateDetector
from trainer_core.dataprep.dataset_builder import create_dataset_from_raw
from trainer_core.dataprep.sampling import resolve_folder_subsets
from trainer_core.dataprep.source_ingest import (
    check_for_test_images,
)

logger = logging.getLogger(__name__)


def run_prepare_stage(args, config=None) -> Path:
    cfg = config or load_config(getattr(args, "config", "params.yaml"), args=args)

    dataset_name = Path(getattr(args, "dataset_name", None) or cfg.data.dataset_name)
    val_split = float(getattr(args, "val_split", cfg.prepare.val_split))
    recreate_dataset = bool(getattr(args, "recreate_dataset", False))
    augment_multiplier = int(getattr(args, "augment_multiplier", cfg.prepare.augment_multiplier))

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

    test_data_exists = check_for_test_images(test_image_input_path)
    if not test_data_exists:
        test_split = float(getattr(args, "test_split", cfg.prepare.test_split))
    else:
        test_split = 0.0

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
