from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict

import yaml

from trainer_core.dataprep.class_mapping import map_class_names_to_ids
from trainer_core.dataprep.labels import convert_polygons_to_bboxes_inplace
from trainer_core.dataprep.sampling import apply_subset_sampling
from trainer_core.dataprep.types import ImageLabelPair, ProcessedFolder


def sorted_iterdir(path: Path) -> list[Path]:
    return sorted(path.iterdir(), key=lambda p: p.name)


def sorted_glob(paths) -> list[Path]:
    return sorted(paths, key=lambda p: p.name)


def check_for_test_images(test_image_input_path: Path) -> bool:
    if not test_image_input_path.exists():
        return False
    for image_folder in sorted_iterdir(test_image_input_path):
        if image_folder.is_dir():
            return True
    return False


def remap_yaml_dataset_labels(dataset_dir: Path, target_class_mapping: dict[int, str]) -> None:
    """Remap dataset labels to match target class IDs by class name."""
    yaml_file = dataset_dir / "data.yaml"
    if not yaml_file.exists():
        return

    print(f"\nProcessing dataset in: {dataset_dir}")

    with yaml_file.open("r", encoding="utf-8") as handle:
        dataset_config = yaml.safe_load(handle)

    class_mapping = map_class_names_to_ids(dataset_config["names"], target_class_mapping)

    for label_file in dataset_dir.rglob("*.txt"):
        if "labels" not in str(label_file.parent):
            continue

        with label_file.open("r", encoding="utf-8") as handle:
            lines = [line.strip().split() for line in handle.readlines()]

        new_lines: list[str] = []
        for parts in lines:
            if parts:
                orig_class_id = int(parts[0])
                if orig_class_id in class_mapping:
                    parts[0] = str(class_mapping[orig_class_id])
                    new_lines.append(" ".join(parts) + "\n")
                else:
                    print(f"Skipping label with unmapped class ID: {orig_class_id}")

        with label_file.open("w", encoding="utf-8") as handle:
            handle.writelines(new_lines)


def process_cvat_folder(
    some_folder: Path,
    folder_to_process: Path,
    scene_name: str,
    target_class_mapping: Dict[int, str],
    folder_subsets: Dict[str, int | float],
    temp_folders: list[Path],
) -> ProcessedFolder:
    folder_pairs: list[ImageLabelPair] = []
    empty_label_count = 0
    skip_count = 0

    train_txt = folder_to_process / "train.txt"
    with train_txt.open("r", encoding="utf-8") as handle:
        image_paths = [line.strip() for line in handle.readlines()]

    for image_rel_path in image_paths:
        path = Path(image_rel_path)
        image_rel_path = Path(*path.parts[1:])

        image_path = folder_to_process / image_rel_path
        label_rel_path = Path("labels") / image_rel_path.relative_to("images").with_suffix(".txt")
        label_path = folder_to_process / label_rel_path

        if image_path.exists():
            if not label_path.exists() or label_path.stat().st_size == 0:
                label_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.touch()
                empty_label_count += 1
            convert_polygons_to_bboxes_inplace(label_path)
            folder_pairs.append(ImageLabelPair(image_path, label_path, scene_name))
        else:
            skip_count += 1

    if (folder_to_process / "data.yaml").exists():
        temp_folder = some_folder.parent / f"{some_folder.name}_temp"
        if temp_folder.exists():
            shutil.rmtree(temp_folder)
        shutil.copytree(folder_to_process, temp_folder)
        try:
            remap_yaml_dataset_labels(temp_folder, target_class_mapping)
        except (KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
            shutil.rmtree(temp_folder, ignore_errors=True)
            raise ValueError(
                f"Failed to apply class mapping for CVAT folder '{folder_to_process.name}': {exc}"
            ) from exc

        temp_folders.append(temp_folder)
        folder_pairs = [
            ImageLabelPair(
                temp_folder.joinpath(pair.image.relative_to(folder_to_process)),
                temp_folder.joinpath(pair.label.relative_to(folder_to_process)),
                pair.scene,
            )
            for pair in folder_pairs
        ]

    folder_name = some_folder.name
    if folder_name in folder_subsets:
        folder_pairs = apply_subset_sampling(
            folder_name,
            folder_pairs,
            folder_subsets[folder_name],
        )

    return ProcessedFolder(folder_pairs, temp_folders, empty_label_count, skip_count)


def process_manual_folder(
    some_folder: Path,
    folder_to_process: Path,
    scene_name: str,
    target_class_mapping: Dict[int, str],
    folder_subsets: Dict[str, int | float],
    temp_folders: list[Path],
) -> ProcessedFolder:
    temp_pairs: list[ImageLabelPair] = []
    empty_label_count = 0

    images_folder = folder_to_process / "images"
    labels_folder = folder_to_process / "labels"
    if not images_folder.exists():
        print(f"Skipping {folder_to_process.name}: Missing 'images' folder.")
        return ProcessedFolder([], temp_folders, empty_label_count)

    if not labels_folder.exists():
        labels_folder.mkdir(parents=True, exist_ok=True)

    for image_file in sorted_glob(images_folder.glob("*")):
        if image_file.is_file() and image_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
            label_file = labels_folder / image_file.with_suffix(".txt").name
            if not label_file.exists() or label_file.stat().st_size == 0:
                label_file.touch()
                empty_label_count += 1
            convert_polygons_to_bboxes_inplace(label_file)
            temp_pairs.append(ImageLabelPair(image_file, label_file, scene_name))

    data_yaml_path = folder_to_process / "data.yaml"
    if data_yaml_path.exists():
        print(f"Found data.yaml in manual structure: {folder_to_process.name}")
        temp_folder = some_folder.parent / f"{some_folder.name}_temp"
        with data_yaml_path.open("r", encoding="utf-8") as handle:
            yaml_config = yaml.safe_load(handle) or {}
        if not isinstance(yaml_config, dict):
            raise ValueError(
                f"Invalid data.yaml in '{folder_to_process.name}': expected mapping, got {type(yaml_config)}"
            )
        if "names" not in yaml_config:
            raise ValueError(f"Invalid data.yaml in '{folder_to_process.name}': missing 'names' section")

        yaml_classes = yaml_config["names"]
        mapping = map_class_names_to_ids(yaml_classes, target_class_mapping)
        if not mapping:
            print(f"Warning: No classes in {data_yaml_path} can be mapped to target classes")
            if isinstance(yaml_classes, list):
                source_classes = yaml_classes
            else:
                source_classes = list(yaml_classes.values())
            print(f"Source classes: {source_classes}")
            print(f"Target classes: {list(target_class_mapping.values())}")

        if temp_folder.exists():
            shutil.rmtree(temp_folder)
        shutil.copytree(folder_to_process, temp_folder)
        try:
            remap_yaml_dataset_labels(temp_folder, target_class_mapping)
        except (KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
            shutil.rmtree(temp_folder, ignore_errors=True)
            raise ValueError(
                f"Failed to apply class mapping for folder '{folder_to_process.name}': {exc}"
            ) from exc

        temp_folders.append(temp_folder)
        temp_pairs = [
            ImageLabelPair(
                temp_folder.joinpath(pair.image.relative_to(folder_to_process)),
                temp_folder.joinpath(pair.label.relative_to(folder_to_process)),
                pair.scene,
            )
            for pair in temp_pairs
        ]

    folder_name = some_folder.name
    if folder_name in folder_subsets:
        temp_pairs = apply_subset_sampling(
            folder_name,
            temp_pairs,
            folder_subsets[folder_name],
        )

    return ProcessedFolder(temp_pairs, temp_folders, empty_label_count)


__all__ = [
    "check_for_test_images",
    "process_cvat_folder",
    "process_manual_folder",
    "remap_yaml_dataset_labels",
    "sorted_glob",
    "sorted_iterdir",
]
