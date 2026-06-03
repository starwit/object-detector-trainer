from __future__ import annotations

import shutil
from pathlib import Path

import yaml

from object_detector_trainer.dataprep.class_mapping import map_class_names_to_ids
from object_detector_trainer.dataprep.labels import convert_polygons_to_bboxes_inplace
from object_detector_trainer.dataprep.sampling import apply_subset_sampling
from object_detector_trainer.dataprep.types import ImageLabelPair, ProcessedFolder

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def _contains_supported_images(root: Path) -> bool:
    for candidate in root.rglob("*"):
        if candidate.is_file() and candidate.suffix.lower() in _IMAGE_SUFFIXES:
            return True
    return False


def check_for_test_images(test_image_input_path: Path) -> bool:
    if not test_image_input_path.exists():
        return False
    for image_folder in sorted(test_image_input_path.iterdir(), key=lambda p: p.name):
        if image_folder.is_dir() and _contains_supported_images(image_folder):
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
    if not class_mapping:
        raise ValueError(
            f"No classes in {yaml_file} can be mapped to target classes: "
            f"{list(target_class_mapping.values())}"
        )

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
                    raise ValueError(
                        f"Label {label_file} references unmapped class ID {orig_class_id}."
                    )

        with label_file.open("w", encoding="utf-8") as handle:
            handle.writelines(new_lines)


def _copy_and_remap_yaml_dataset(
    *,
    source_folder: Path,
    temp_folder: Path,
    target_class_mapping: dict[int, str],
    error_label: str,
) -> Path:
    if temp_folder.exists():
        shutil.rmtree(temp_folder)
    shutil.copytree(source_folder, temp_folder)
    try:
        remap_yaml_dataset_labels(temp_folder, target_class_mapping)
    except (KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        shutil.rmtree(temp_folder, ignore_errors=True)
        raise ValueError(
            f"Failed to apply class mapping for {error_label} '{source_folder.name}': {exc}"
        ) from exc
    return temp_folder


def _move_pairs_to_temp_folder(
    pairs: list[ImageLabelPair],
    *,
    source_folder: Path,
    temp_folder: Path,
) -> list[ImageLabelPair]:
    return [
        ImageLabelPair(
            temp_folder.joinpath(pair.image.relative_to(source_folder)),
            temp_folder.joinpath(pair.label.relative_to(source_folder)),
            pair.scene,
        )
        for pair in pairs
    ]


def process_cvat_folder(
    source_folder: Path,
    target_class_mapping: dict[int, str],
    folder_subsets: dict[str, int | float],
) -> ProcessedFolder:
    scene_name = source_folder.name
    folder_pairs: list[ImageLabelPair] = []
    temp_folders: list[Path] = []
    empty_label_count = 0

    train_txt = source_folder / "train.txt"
    with train_txt.open("r", encoding="utf-8") as handle:
        image_paths = [line.strip() for line in handle if line.strip()]

    for image_rel_path in image_paths:
        path = Path(image_rel_path)
        if path.parts and path.parts[0] == "data":
            path = Path(*path.parts[1:])
        if not path.parts or path.parts[0] != "images":
            raise ValueError(
                f"CVAT train.txt path must point under images/: {image_rel_path!r}"
            )

        image_path = source_folder / path
        label_rel_path = Path("labels") / path.relative_to("images").with_suffix(".txt")
        label_path = source_folder / label_rel_path

        if not image_path.exists():
            raise FileNotFoundError(
                f"CVAT train.txt references an image that does not exist: {image_path}"
            )
        if not label_path.exists() or label_path.stat().st_size == 0:
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.touch()
            empty_label_count += 1
        convert_polygons_to_bboxes_inplace(label_path)
        folder_pairs.append(ImageLabelPair(image_path, label_path, scene_name))

    if (source_folder / "data.yaml").exists():
        temp_folder = source_folder.parent / f"{source_folder.name}_temp"
        temp_folders.append(
            _copy_and_remap_yaml_dataset(
                source_folder=source_folder,
                temp_folder=temp_folder,
                target_class_mapping=target_class_mapping,
                error_label="CVAT folder",
            )
        )
        folder_pairs = _move_pairs_to_temp_folder(
            folder_pairs,
            source_folder=source_folder,
            temp_folder=temp_folder,
        )

    if scene_name in folder_subsets:
        folder_pairs = apply_subset_sampling(
            scene_name,
            folder_pairs,
            folder_subsets[scene_name],
        )

    return ProcessedFolder(folder_pairs, temp_folders, empty_label_count)


def process_manual_folder(
    source_folder: Path,
    target_class_mapping: dict[int, str],
    folder_subsets: dict[str, int | float],
) -> ProcessedFolder:
    scene_name = source_folder.name
    temp_pairs: list[ImageLabelPair] = []
    temp_folders: list[Path] = []
    empty_label_count = 0

    images_folder = source_folder / "images"
    labels_folder = source_folder / "labels"
    if not images_folder.exists():
        raise FileNotFoundError(f"Missing images folder: {images_folder}")

    if not labels_folder.exists():
        labels_folder.mkdir(parents=True, exist_ok=True)

    for image_file in sorted(images_folder.glob("*"), key=lambda p: p.name):
        if image_file.is_file() and image_file.suffix.lower() in _IMAGE_SUFFIXES:
            label_file = labels_folder / image_file.with_suffix(".txt").name
            if not label_file.exists() or label_file.stat().st_size == 0:
                label_file.touch()
                empty_label_count += 1
            convert_polygons_to_bboxes_inplace(label_file)
            temp_pairs.append(ImageLabelPair(image_file, label_file, scene_name))

    data_yaml_path = source_folder / "data.yaml"
    if data_yaml_path.exists():
        print(f"Found data.yaml in manual structure: {source_folder.name}")
        temp_folder = source_folder.parent / f"{source_folder.name}_temp"
        with data_yaml_path.open("r", encoding="utf-8") as handle:
            yaml_config = yaml.safe_load(handle) or {}
        if not isinstance(yaml_config, dict):
            raise ValueError(
                f"Invalid data.yaml in '{source_folder.name}': expected mapping, got {type(yaml_config)}"
            )
        if "names" not in yaml_config:
            raise ValueError(f"Invalid data.yaml in '{source_folder.name}': missing 'names' section")

        temp_folders.append(
            _copy_and_remap_yaml_dataset(
                source_folder=source_folder,
                temp_folder=temp_folder,
                target_class_mapping=target_class_mapping,
                error_label="folder",
            )
        )
        temp_pairs = _move_pairs_to_temp_folder(
            temp_pairs,
            source_folder=source_folder,
            temp_folder=temp_folder,
        )

    if scene_name in folder_subsets:
        temp_pairs = apply_subset_sampling(
            scene_name,
            temp_pairs,
            folder_subsets[scene_name],
        )

    return ProcessedFolder(temp_pairs, temp_folders, empty_label_count)


__all__ = [
    "check_for_test_images",
    "process_cvat_folder",
    "process_manual_folder",
    "remap_yaml_dataset_labels",
]
