from __future__ import annotations

import os
import random
import shutil
from pathlib import Path
from typing import Dict, Set

import cv2

from trainer_core.dataprep.augmentation import YOLOAugmenter
from trainer_core.dataprep.class_mapping import (
    apply_class_mapping_config,
    get_class_mapping,
    remap_labels_with_class_mapping,
)
from trainer_core.dataprep.dataset_yaml import create_dataset_yaml
from trainer_core.dataprep.find_duplicates import DuplicateDetector
from trainer_core.dataprep.sampling import oversample_train_pairs
from trainer_core.dataprep.source_ingest import (
    process_cvat_folder,
    process_manual_folder,
    sorted_iterdir,
)
from trainer_core.dataprep.types import ImageLabelPair
from trainer_core.utils.path_ops import link_or_copy


def dedupe_pairs(
    image_label_pairs: list[ImageLabelPair],
    folder_subsets: Dict[str, int | float],
) -> list[ImageLabelPair]:
    detector = DuplicateDetector(phash_threshold=2, ssim_threshold=0.95)
    folder_groups: Dict[str, list[ImageLabelPair]] = {}
    for img_path, lbl_path, scene_name in image_label_pairs:
        folder_groups.setdefault(scene_name, []).append(ImageLabelPair(img_path, lbl_path, scene_name))

    def _is_oversample(value) -> bool:
        return isinstance(value, float) and value > 1.0

    oversampled_folders: Set[str] = {
        folder_name for folder_name, ratio in folder_subsets.items() if _is_oversample(ratio)
    }

    processed_pairs: list[ImageLabelPair] = []
    unique_images_per_folder: Dict[str, Set[Path]] = {}

    for folder_name, folder_pairs in folder_groups.items():
        folder_images = [img_path for img_path, _, _ in folder_pairs]
        if folder_name in oversampled_folders:
            print(f"Folder '{folder_name}': Keeping all {len(folder_pairs)} images (oversampled folder)")
            processed_pairs.extend(folder_pairs)
            unique_images_per_folder[folder_name] = set(detector.get_unique_images(folder_images))
        else:
            clusters = detector.find_duplicates(folder_images)
            if clusters:
                print(f"Found {len(clusters)} duplicate clusters in folder '{folder_name}':")
                detector.print_duplicate_clusters(clusters)
            unique_folder_images = set(detector.get_unique_images(folder_images))
            unique_folder_pairs = [
                ImageLabelPair(img, lbl, scene)
                for img, lbl, scene in folder_pairs
                if img in unique_folder_images
            ]
            print(
                f"Folder '{folder_name}': {len(unique_folder_pairs)}/{len(folder_pairs)} "
                "unique images after duplicate removal"
            )
            processed_pairs.extend(unique_folder_pairs)
            unique_images_per_folder[folder_name] = unique_folder_images

    if len(folder_groups) > 1:
        print("\nChecking for duplicates between different folders...")
        all_unique_images: list[Path] = []
        for unique_images in unique_images_per_folder.values():
            all_unique_images.extend(list(unique_images))
        cross_clusters = detector.find_duplicates(all_unique_images)
        if cross_clusters:
            print(f"Found {len(cross_clusters)} duplicate clusters between folders:")
            detector.print_duplicate_clusters(cross_clusters)

            scene_by_path: Dict[Path, str] = {}
            for scene_name, folder_pairs in folder_groups.items():
                for img_path, _lbl, _scene in folder_pairs:
                    scene_by_path[img_path] = scene_name

            cross_keep: Set[Path] = set()
            for imgs in cross_clusters.values():
                def rank(path: Path) -> tuple[int, int, str]:
                    scene = scene_by_path.get(path, "")
                    is_replay = 1 if scene == "replay" or "/replay/" in str(path) else 0
                    is_oversampled = 1 if scene in oversampled_folders else 0
                    return (-is_replay, -is_oversampled, str(path))

                keep = sorted(imgs, key=rank)[0]
                cross_keep.add(keep)

            in_any_cluster = {path for imgs in cross_clusters.values() for path in imgs}
            cross_unique_images = set(path for path in all_unique_images if path not in in_any_cluster) | cross_keep
            processed_pairs = [
                ImageLabelPair(img_path, lbl_path, scene_name)
                for img_path, lbl_path, scene_name in processed_pairs
                if img_path in cross_unique_images
            ]
        else:
            print("No duplicates found between different folders")

    return processed_pairs


def augment(image_label_pairs: list[ImageLabelPair], augment_multiplier: int = 1):
    print(f"Applying augmentation with multiplier {augment_multiplier}...")
    augmenter = YOLOAugmenter(multiplier=augment_multiplier)
    augmented_pairs: list[ImageLabelPair] = []
    temp_folders: list[Path] = []

    for img_path, label_path, scene_name in image_label_pairs:
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with label_path.open("r", encoding="utf-8") as handle:
            labels = [list(map(float, line.strip().split())) for line in handle.readlines()]

        augmented_results = augmenter.augment_image_and_labels(image, labels)

        for idx, (aug_image, aug_labels) in enumerate(augmented_results):
            if idx == 0:
                continue

            aug_img_folder_path = Path(f"{img_path.parent.parent}-aug") / img_path.parent.name
            aug_label_folder_path = Path(f"{label_path.parent.parent}-aug") / label_path.parent.name

            if not os.path.exists(aug_img_folder_path):
                os.makedirs(aug_img_folder_path)
            if not os.path.exists(aug_label_folder_path):
                os.makedirs(aug_label_folder_path)

            temp_folders.append(aug_img_folder_path.parent)

            aug_img_path = (
                Path(f"{img_path.parent.parent}-aug")
                / img_path.parent.name
                / f"{img_path.stem}_aug{idx}{img_path.suffix}"
            )
            aug_label_path = (
                Path(f"{label_path.parent.parent}-aug")
                / label_path.parent.name
                / f"{label_path.stem}_aug{idx}{label_path.suffix}"
            )

            aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(aug_img_path), aug_image_bgr)

            with aug_label_path.open("w", encoding="utf-8") as handle:
                for label in aug_labels:
                    handle.write(" ".join(map(str, label)) + "\n")

            augmented_pairs.append(ImageLabelPair(aug_img_path, aug_label_path, scene_name))

    print(f"Added {len(augmented_pairs)} augmented images")
    return augmented_pairs, list(set(temp_folders))


def process_single_images(
    input_path: Path,
    train_output_path: Path,
    test_output_path: Path,
    val_split: float,
    test_split: float,
    augment_multiplier: int,
    custom_classes=None,
    use_coco_classes: bool = True,
    folder_subsets=None,
    class_mapping_config=None,
):
    """Build train/val/test YOLO datasets from raw folder inputs."""
    if folder_subsets is None:
        folder_subsets = {}

    train_img_output_path = train_output_path / "train" / "images"
    val_img_output_path = train_output_path / "val" / "images"
    test_img_output_path = test_output_path / "val" / "images"
    train_label_output_path = train_output_path / "train" / "labels"
    val_label_output_path = train_output_path / "val" / "labels"
    test_label_output_path = test_output_path / "val" / "labels"

    for path in (
        train_img_output_path,
        val_img_output_path,
        test_img_output_path,
        train_label_output_path,
        val_label_output_path,
        test_label_output_path,
    ):
        os.makedirs(path, exist_ok=True)

    if not input_path.exists():
        print(f"Skipping {input_path} - not found.")
        return 0, 0, 0

    print(f"Processing images from {input_path}")

    image_label_pairs: list[ImageLabelPair] = []
    temp_folders: list[Path] = []
    empty_label_count = 0
    skip_count = 0

    is_test_data = (test_output_path == train_output_path) and (val_split == 1)

    target_class_mapping = get_class_mapping(custom_classes, use_coco_classes)

    original_class_list = custom_classes if custom_classes else []
    final_class_list = custom_classes if custom_classes else []
    source_to_target_map: dict[str, str] = {}

    if class_mapping_config and custom_classes:
        final_class_list, source_to_target_map = apply_class_mapping_config(
            custom_classes,
            class_mapping_config,
        )
        print("\nClass mapping will be applied to all label files.")
        print(f"  Source-to-target mapping: {source_to_target_map}")

    for some_folder in sorted_iterdir(input_path):
        if not some_folder.is_dir():
            continue

        folder_to_process = some_folder
        scene_name = some_folder.name
        train_txt = folder_to_process / "train.txt"
        if train_txt.exists():
            cvat_result = process_cvat_folder(
                some_folder,
                folder_to_process,
                scene_name,
                target_class_mapping,
                folder_subsets,
                temp_folders,
            )
            empty_label_count += cvat_result.empty_label_count
            skip_count += cvat_result.skip_count
            image_label_pairs.extend(cvat_result.pairs)
        else:
            manual_result = process_manual_folder(
                some_folder,
                folder_to_process,
                scene_name,
                target_class_mapping,
                folder_subsets,
                temp_folders,
            )
            empty_label_count += manual_result.empty_label_count
            image_label_pairs.extend(manual_result.pairs)

    print(f"Included {empty_label_count} images with empty labels (no objects).")
    print(f"Skipped {skip_count} images that couldn't be found.")

    if not image_label_pairs:
        print("No valid image-label pairs found.")
        for temp_folder in temp_folders:
            shutil.rmtree(temp_folder)
        return 0, 0, 0

    print("Find duplicate images...")
    processed_pairs = dedupe_pairs(image_label_pairs, folder_subsets)

    print(f"\nOriginal number of images: {len(image_label_pairs)}")
    print(f"Number of images after smart duplicate removal: {len(processed_pairs)}")

    image_label_pairs = processed_pairs
    random.shuffle(image_label_pairs)

    total_images = len(image_label_pairs)
    test_size = int(total_images * test_split)
    val_size = int(total_images * val_split)

    test_pairs = image_label_pairs[:test_size]
    val_pairs = image_label_pairs[test_size : test_size + val_size]
    train_pairs = image_label_pairs[test_size + val_size :]

    train_pairs = oversample_train_pairs(train_pairs, folder_subsets)

    augmented_pairs, aug_temp_folders = augment(train_pairs, augment_multiplier)
    temp_folders.extend(aug_temp_folders)

    random.shuffle(augmented_pairs)
    train_pairs.extend(augmented_pairs)

    def copy_pairs(pairs, img_path: Path, label_path: Path) -> int:
        count = 0

        def with_dup_suffix(name: str, duplicate_index: int) -> str:
            if duplicate_index == 0:
                return name
            stem, ext = os.path.splitext(name)
            return f"{stem}__dup{duplicate_index}{ext}"

        name_counts: Dict[str, int] = {}

        for img_file, label_file, scene_name in pairs:
            if is_test_data:
                base_img_name = f"{img_file.stem}__scene_{scene_name}{img_file.suffix}"
                base_label_name = f"{label_file.stem}__scene_{scene_name}{label_file.suffix}"
            else:
                base_img_name = img_file.name
                base_label_name = label_file.name

            duplicate_index = name_counts.get(base_img_name, 0)
            name_counts[base_img_name] = duplicate_index + 1

            new_img_name = with_dup_suffix(base_img_name, duplicate_index)
            new_label_name = with_dup_suffix(base_label_name, duplicate_index)

            target_img = img_path / new_img_name
            target_label = label_path / new_label_name

            link_or_copy(img_file, target_img)
            link_or_copy(label_file, target_label)

            if source_to_target_map:
                remap_labels_with_class_mapping(
                    target_label,
                    source_to_target_map,
                    original_class_list,
                    final_class_list,
                )

            count += 1
        return count

    test_count = copy_pairs(test_pairs, test_img_output_path, test_label_output_path)
    val_count = copy_pairs(val_pairs, val_img_output_path, val_label_output_path)
    train_count = copy_pairs(train_pairs, train_img_output_path, train_label_output_path)

    for temp_folder in temp_folders:
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)

    return train_count, val_count, test_count


def create_dataset_from_raw(
    *,
    dataset_path: Path,
    training_path: Path,
    test_path: Path,
    train_image_input_path: Path,
    test_image_input_path: Path,
    val_split: float,
    test_split: float,
    augment_multiplier: int,
    custom_classes,
    use_coco_classes: bool,
    folder_subsets: Dict[str, int | float],
    class_mapping_config: dict,
    test_data_exists: bool,
    recreate_dataset: bool,
) -> tuple[int, int, int]:
    if dataset_path.exists() and recreate_dataset:
        print(f"Recreating dataset '{dataset_path.name}'...")
        shutil.rmtree(dataset_path)

    dataset_path.mkdir(parents=True, exist_ok=True)
    for path in (training_path, test_path):
        path.mkdir(parents=True, exist_ok=True)

    for path in (
        training_path / "train" / "images",
        training_path / "train" / "labels",
        training_path / "val" / "images",
        training_path / "val" / "labels",
        test_path / "val" / "images",
        test_path / "val" / "labels",
    ):
        path.mkdir(parents=True, exist_ok=True)

    total_train_frames, total_val_frames, total_test_frames = process_single_images(
        input_path=train_image_input_path,
        train_output_path=training_path,
        test_output_path=test_path,
        val_split=val_split,
        test_split=test_split,
        augment_multiplier=augment_multiplier,
        custom_classes=custom_classes,
        use_coco_classes=use_coco_classes,
        folder_subsets=folder_subsets,
        class_mapping_config=class_mapping_config,
    )
    if total_train_frames <= 0:
        raise ValueError(
            "Prepare stage produced 0 training frames. "
            f"(train={total_train_frames}, val={total_val_frames}, test={total_test_frames}). "
            "Ensure raw_data/train contains labeled images and split/subset settings leave at least one training sample."
        )

    create_dataset_yaml(training_path, custom_classes, use_coco_classes, class_mapping_config)

    test_folder_frame_count = 0
    if test_data_exists:
        _, test_folder_frame_count, _ = process_single_images(
            input_path=test_image_input_path,
            train_output_path=test_path,
            test_output_path=test_path,
            val_split=1,
            test_split=0,
            augment_multiplier=1,
            custom_classes=custom_classes,
            use_coco_classes=use_coco_classes,
            folder_subsets={},
            class_mapping_config=class_mapping_config,
        )

    create_dataset_yaml(test_path, custom_classes, use_coco_classes, class_mapping_config)
    return total_train_frames, total_val_frames, total_test_frames + test_folder_frame_count


__all__ = ["augment", "create_dataset_from_raw", "dedupe_pairs", "process_single_images"]
