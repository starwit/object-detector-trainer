from __future__ import annotations

import random
import shutil
from pathlib import Path

import cv2

from object_detector_trainer.dataprep.augmentation import YOLOAugmenter
from object_detector_trainer.dataprep.class_mapping import (
    apply_class_mapping_config,
    get_class_mapping,
    remap_labels_with_class_mapping,
)
from object_detector_trainer.dataprep.dataset_yaml import create_dataset_yaml
from object_detector_trainer.dataprep.find_duplicates import DuplicateDetector
from object_detector_trainer.dataprep.sampling import oversample_train_pairs
from object_detector_trainer.dataprep.source_ingest import (
    process_cvat_folder,
    process_manual_folder,
)
from object_detector_trainer.dataprep.types import ImageLabelPair
from object_detector_trainer.utils.path_ops import link_or_copy


_GENERATED_SOURCE_SUFFIXES = ("_temp", "-aug")


def _remove_generated_source_folders(input_path: Path) -> None:
    for source_folder in sorted(input_path.iterdir(), key=lambda p: p.name):
        if source_folder.is_dir() and source_folder.name.endswith(_GENERATED_SOURCE_SUFFIXES):
            shutil.rmtree(source_folder)


def _unique_images_from_clusters(
    image_paths: list[Path],
    clusters: dict[Path, list[Path]],
) -> set[Path]:
    duplicates: set[Path] = set()
    for cluster_images in clusters.values():
        keep = min(cluster_images)
        duplicates.update(set(cluster_images) - {keep})
    return set(image_paths) - duplicates


def _select_cross_folder_images(
    *,
    all_unique_images: list[Path],
    cross_clusters: dict[Path, list[Path]],
    folder_groups: dict[str, list[ImageLabelPair]],
    oversampled_folders: set[str],
) -> set[Path]:
    scene_by_path = {
        img_path: scene_name
        for scene_name, folder_pairs in folder_groups.items()
        for img_path, _lbl, _scene in folder_pairs
    }

    cross_keep: set[Path] = set()
    for cluster_images in cross_clusters.values():
        def rank(path: Path) -> tuple[int, int, str]:
            scene = scene_by_path[path]
            is_replay = 1 if scene == "replay" or "/replay/" in str(path) else 0
            is_oversampled = 1 if scene in oversampled_folders else 0
            return (-is_replay, -is_oversampled, str(path))

        cross_keep.add(sorted(cluster_images, key=rank)[0])

    clustered_images = {path for imgs in cross_clusters.values() for path in imgs}
    return (set(all_unique_images) - clustered_images) | cross_keep


def dedupe_pairs(
    image_label_pairs: list[ImageLabelPair],
    folder_subsets: dict[str, int | float],
) -> list[ImageLabelPair]:
    detector = DuplicateDetector(phash_threshold=2, ssim_threshold=0.95)
    folder_groups: dict[str, list[ImageLabelPair]] = {}
    for img_path, lbl_path, scene_name in image_label_pairs:
        folder_groups.setdefault(scene_name, []).append(ImageLabelPair(img_path, lbl_path, scene_name))

    oversampled_folders: set[str] = {
        folder_name
        for folder_name, ratio in folder_subsets.items()
        if isinstance(ratio, float) and ratio > 1.0
    }

    processed_pairs: list[ImageLabelPair] = []
    unique_images_per_folder: dict[str, set[Path]] = {}

    for folder_name, folder_pairs in folder_groups.items():
        folder_images = [img_path for img_path, _, _ in folder_pairs]
        clusters = detector.find_duplicates(folder_images)
        unique_folder_images = _unique_images_from_clusters(folder_images, clusters)
        unique_images_per_folder[folder_name] = unique_folder_images

        if folder_name in oversampled_folders:
            print(f"Folder '{folder_name}': Keeping all {len(folder_pairs)} images (oversampled folder)")
            processed_pairs.extend(folder_pairs)
            continue

        if clusters:
            print(f"Found {len(clusters)} duplicate clusters in folder '{folder_name}':")
            detector.print_duplicate_clusters(clusters)
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

    if len(folder_groups) > 1:
        print("\nChecking for duplicates between different folders...")
        all_unique_images: list[Path] = []
        for unique_images in unique_images_per_folder.values():
            all_unique_images.extend(list(unique_images))
        cross_clusters = detector.find_duplicates(all_unique_images)
        if cross_clusters:
            print(f"Found {len(cross_clusters)} duplicate clusters between folders:")
            detector.print_duplicate_clusters(cross_clusters)
            cross_unique_images = _select_cross_folder_images(
                all_unique_images=all_unique_images,
                cross_clusters=cross_clusters,
                folder_groups=folder_groups,
                oversampled_folders=oversampled_folders,
            )
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
        if image is None:
            raise FileNotFoundError(f"Could not read image for augmentation: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with label_path.open("r", encoding="utf-8") as handle:
            labels = [list(map(float, line.strip().split())) for line in handle.readlines()]

        augmented_results = augmenter.augment_image_and_labels(image, labels)

        for idx, (aug_image, aug_labels) in enumerate(augmented_results):
            if idx == 0:
                continue

            aug_img_folder_path = Path(f"{img_path.parent.parent}-aug") / img_path.parent.name
            aug_label_folder_path = Path(f"{label_path.parent.parent}-aug") / label_path.parent.name
            aug_img_folder_path.mkdir(parents=True, exist_ok=True)
            aug_label_folder_path.mkdir(parents=True, exist_ok=True)

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


def _ensure_output_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def _validate_training_splits(
    *,
    val_split: float,
    test_split: float,
    test_data_exists: bool,
) -> None:
    if not 0 <= val_split < 1:
        raise ValueError(f"prepare.val_split must be >= 0 and < 1, got {val_split}.")
    if test_split < 0:
        raise ValueError(f"prepare.test_split must be >= 0, got {test_split}.")
    if not test_data_exists and test_split >= 1:
        raise ValueError(
            f"prepare.test_split must be < 1 when raw_data/test is absent, got {test_split}."
        )
    if not test_data_exists and val_split + test_split >= 1:
        raise ValueError(
            "prepare.val_split + prepare.test_split must be < 1 when raw_data/test is absent. "
            f"Got val_split={val_split}, test_split={test_split}; no samples would remain for training."
        )


def _collect_image_label_pairs(
    *,
    input_path: Path,
    target_class_mapping: dict[int, str],
    folder_subsets: dict[str, int | float],
) -> tuple[list[ImageLabelPair], list[Path], int]:
    image_label_pairs: list[ImageLabelPair] = []
    temp_folders: list[Path] = []
    empty_label_count = 0

    for source_folder in sorted(input_path.iterdir(), key=lambda p: p.name):
        if not source_folder.is_dir():
            continue

        train_txt = source_folder / "train.txt"
        if train_txt.exists():
            result = process_cvat_folder(
                source_folder,
                target_class_mapping,
                folder_subsets,
            )
        else:
            result = process_manual_folder(
                source_folder,
                target_class_mapping,
                folder_subsets,
            )

        empty_label_count += result.empty_label_count
        image_label_pairs.extend(result.pairs)
        temp_folders.extend(result.temp_folders)

    return image_label_pairs, temp_folders, empty_label_count


def _split_pairs(
    image_label_pairs: list[ImageLabelPair],
    *,
    val_split: float,
    test_split: float,
) -> tuple[list[ImageLabelPair], list[ImageLabelPair], list[ImageLabelPair]]:
    total_images = len(image_label_pairs)
    test_size = int(total_images * test_split)
    val_size = int(total_images * val_split)

    test_pairs = image_label_pairs[:test_size]
    val_pairs = image_label_pairs[test_size : test_size + val_size]
    train_pairs = image_label_pairs[test_size + val_size :]
    return train_pairs, val_pairs, test_pairs


def _copy_pairs_to_split(
    *,
    pairs: list[ImageLabelPair],
    image_dir: Path,
    label_dir: Path,
    include_scene_in_name: bool,
    source_to_target_map: dict[str, str],
    original_class_list: list[str],
    final_class_list: list[str],
) -> int:
    name_counts: dict[str, int] = {}

    def add_duplicate_suffix(name: str, duplicate_index: int) -> str:
        if duplicate_index == 0:
            return name
        path = Path(name)
        return f"{path.stem}__dup{duplicate_index}{path.suffix}"

    for img_file, label_file, scene_name in pairs:
        if include_scene_in_name:
            base_img_name = f"{img_file.stem}__scene_{scene_name}{img_file.suffix}"
            base_label_name = f"{label_file.stem}__scene_{scene_name}{label_file.suffix}"
        else:
            base_img_name = img_file.name
            base_label_name = label_file.name

        duplicate_index = name_counts.get(base_img_name, 0)
        name_counts[base_img_name] = duplicate_index + 1

        target_img = image_dir / add_duplicate_suffix(base_img_name, duplicate_index)
        target_label = label_dir / add_duplicate_suffix(base_label_name, duplicate_index)

        link_or_copy(img_file, target_img)
        shutil.copy2(label_file, target_label)

        if source_to_target_map:
            remap_labels_with_class_mapping(
                target_label,
                source_to_target_map,
                original_class_list,
                final_class_list,
            )

    return len(pairs)


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

    _ensure_output_dirs(
        train_img_output_path,
        val_img_output_path,
        test_img_output_path,
        train_label_output_path,
        val_label_output_path,
        test_label_output_path,
    )

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    _remove_generated_source_folders(input_path)

    print(f"Processing images from {input_path}")

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

    image_label_pairs, temp_folders, empty_label_count = _collect_image_label_pairs(
        input_path=input_path,
        target_class_mapping=target_class_mapping,
        folder_subsets=folder_subsets,
    )
    print(f"Included {empty_label_count} images with empty labels (no objects).")

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

    train_pairs, val_pairs, test_pairs = _split_pairs(
        image_label_pairs,
        val_split=val_split,
        test_split=test_split,
    )
    train_pairs = oversample_train_pairs(train_pairs, folder_subsets)

    augmented_pairs, aug_temp_folders = augment(train_pairs, augment_multiplier)
    temp_folders.extend(aug_temp_folders)

    random.shuffle(augmented_pairs)
    train_pairs.extend(augmented_pairs)

    test_count = _copy_pairs_to_split(
        pairs=test_pairs,
        image_dir=test_img_output_path,
        label_dir=test_label_output_path,
        include_scene_in_name=is_test_data,
        source_to_target_map=source_to_target_map,
        original_class_list=original_class_list,
        final_class_list=final_class_list,
    )
    val_count = _copy_pairs_to_split(
        pairs=val_pairs,
        image_dir=val_img_output_path,
        label_dir=val_label_output_path,
        include_scene_in_name=is_test_data,
        source_to_target_map=source_to_target_map,
        original_class_list=original_class_list,
        final_class_list=final_class_list,
    )
    train_count = _copy_pairs_to_split(
        pairs=train_pairs,
        image_dir=train_img_output_path,
        label_dir=train_label_output_path,
        include_scene_in_name=is_test_data,
        source_to_target_map=source_to_target_map,
        original_class_list=original_class_list,
        final_class_list=final_class_list,
    )

    for temp_folder in temp_folders:
        if temp_folder.exists():
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
    folder_subsets: dict[str, int | float],
    class_mapping_config: dict,
    test_data_exists: bool,
    recreate_dataset: bool,
) -> tuple[int, int, int]:
    if dataset_path.exists() and recreate_dataset:
        print(f"Recreating dataset '{dataset_path.name}'...")
        shutil.rmtree(dataset_path)

    _ensure_output_dirs(
        dataset_path,
        training_path,
        test_path,
        training_path / "train" / "images",
        training_path / "train" / "labels",
        training_path / "val" / "images",
        training_path / "val" / "labels",
        test_path / "val" / "images",
        test_path / "val" / "labels",
    )

    _validate_training_splits(
        val_split=val_split,
        test_split=test_split,
        test_data_exists=test_data_exists,
    )

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
    if val_split > 0 and total_val_frames <= 0:
        raise ValueError(
            "Prepare stage produced 0 validation frames. "
            f"(train={total_train_frames}, val={total_val_frames}, test={total_test_frames}). "
            "Increase prepare.val_split, add more unique training images, or relax subset/duplicate-removal settings."
        )
    if test_split > 0 and total_test_frames <= 0:
        raise ValueError(
            "Prepare stage produced 0 test frames from prepare.test_split. "
            f"(train={total_train_frames}, val={total_val_frames}, test={total_test_frames}). "
            "Increase prepare.test_split, add more unique training images, or provide raw_data/test."
        )
    if not test_data_exists and test_split <= 0:
        raise ValueError(
            "Prepare stage produced 0 test frames. "
            "Provide raw_data/test or set prepare.test_split > 0 before running the full DVC pipeline."
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
        if test_folder_frame_count <= 0:
            raise ValueError(
                "Prepare stage produced 0 test frames from raw_data/test. "
                "Ensure raw_data/test contains valid images."
            )

    create_dataset_yaml(test_path, custom_classes, use_coco_classes, class_mapping_config)
    return total_train_frames, total_val_frames, total_test_frames + test_folder_frame_count


__all__ = ["augment", "create_dataset_from_raw", "dedupe_pairs", "process_single_images"]
