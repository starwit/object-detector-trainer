"""Build the prepared dataset from the folders under ``raw_data/``.

Input::

    raw_data/<export>/data.yaml           whole CVAT Ultralytics YOLO export, unpacked
    raw_data/train/<source>/images/*.jpg   manually assigned source
    raw_data/train/<source>/labels/*.txt   YOLO labels; a missing or empty file means background
    raw_data/val/<source>/...              optional fixed validation set
    raw_data/test/<source>/...             optional fixed test set

Exported subsets and outer folders supply split assignments. Duplicate and
near-duplicate images are removed within and across sets first; missing val/test
sets are then generated from the remaining training data.

Output, always rebuilt from scratch::

    datasets/<name>/dataset.yaml
    datasets/<name>/{train,val,test}/images/
    datasets/<name>/{train,val,test}/labels/
    datasets/<name>/manifest.csv     one row per image: where it came from and its classes
"""

from __future__ import annotations

import csv
import logging
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

from object_detector_trainer.classes import final_classes, final_name_by_configured_name, source_label_names
from object_detector_trainer.config import Config, DataConfig
from object_detector_trainer.find_duplicates import DuplicateDetector
from object_detector_trainer.labels import label_class_ids, normalize_yolo_label

log = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS = ("train", "val", "test")
MANIFEST_COLUMNS = ["split", "file", "source", "source_image", "source_classes"]


@dataclass
class Sample:
    """One labeled image from a source."""

    image: Path
    label: Path | None
    source: str
    split: str
    label_names: dict[int, str | None]  # label ID -> configured class name (None: ignored class)


def build_dataset(config: Config, workspace: Path) -> dict[str, list[Sample]]:
    """Collect sources, fill missing splits and write the prepared dataset."""
    folders = collect_raw_samples(workspace / "raw_data", config.data)
    check_repeated_images(folders)
    folders = remove_duplicates(folders)

    val_split = 0.0 if "val" in folders else config.prepare.val_split
    test_split = 0.0 if "test" in folders else config.prepare.test_split
    train, val, test = random_split(folders.get("train", []), val_split, test_split, config.seed)
    splits = {"train": train, "val": folders.get("val", val), "test": folders.get("test", test)}

    check_split_sizes(splits, validation_expected="val" in folders or val_split > 0)
    write_dataset(workspace / "datasets" / config.data.dataset_name, splits, config.data, workspace)
    return splits


# Collecting samples ---------------------------------------------------------


def split_name(name: str) -> str | None:
    """Normalize subset names, without changing file paths."""
    return {"train": "train", "val": "val", "valid": "val", "validation": "val", "test": "test"}.get(name.casefold())


def collect_raw_samples(raw_data: Path, data: DataConfig) -> dict[str, list[Sample]]:
    """Read whole exports at the root and sources in manually assigned folders."""
    if not raw_data.is_dir():
        raise FileNotFoundError(f"Input folder not found: {raw_data}")
    if (raw_data / "data.yaml").exists() or (raw_data / "images").is_dir():
        return collect_source_samples(raw_data, data)

    folders: dict[str, list[Sample]] = {}
    for folder in sorted(raw_data.iterdir()):
        if folder.name.startswith("."):
            continue
        is_source = (folder / "data.yaml").exists() or (folder / "images").is_dir()
        assigned_split = split_name(folder.name)
        if is_source and len(source_split_references(folder, read_source_metadata(folder))) > 1:
            assigned_split = None  # A whole project export can itself be named "train".
        sources = sorted(folder.iterdir()) if assigned_split and not is_source else [folder]
        for source in sources:
            if source.name.startswith("."):
                continue
            collected = collect_input_source(source, data, assigned_split)
            for split, samples in collected.items():
                folders.setdefault(split, []).extend(samples)
    return folders


def collect_input_source(source: Path, data: DataConfig, assigned_split: str | None = None) -> dict[str, list[Sample]]:
    """Read a source folder. ZIP files are rejected: DVC would store every new ZIP in full."""
    if source.suffix.lower() == ".zip":
        raise ValueError(
            f"{source} is a ZIP file. Unpack it first and delete the ZIP:\n"
            f"  unzip {source} -d {source.with_suffix('')}"
        )
    if source.is_dir():
        return collect_source_samples(source, data, assigned_split)
    return {}


def read_source_metadata(source_dir: Path) -> dict:
    path = source_dir / "data.yaml"
    if not path.exists():
        return {}
    content = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(content, dict) or "names" not in content:
        raise ValueError(f"Invalid source metadata in {path}: missing names.")
    return content


def source_split_references(source_dir: Path, metadata: dict) -> dict[str, str | list[str]]:
    """Prefer data.yaml; older exports can supply subset lists without metadata."""
    references = {}
    metadata_keys = {"path", "names", "nc", "channels", "download", "kpt_shape", "flip_idx"}
    for name, reference in metadata.items():
        split = split_name(str(name))
        if split is None:
            if name not in metadata_keys:
                raise ValueError(
                    f"Unsupported subset {name!r} in {source_dir / 'data.yaml'}. Use Train, Validation or Test."
                )
            continue
        if split in references:
            raise ValueError(f"Multiple entries for {split!r} in {source_dir / 'data.yaml'}.")
        references[split] = reference
    if not references:
        for path in sorted(source_dir.glob("*.txt")):
            split = split_name(path.stem)
            if split:
                if split in references:
                    raise ValueError(f"Multiple image lists for {split!r} in {source_dir}.")
                references[split] = path.name
    return references


def collect_source_samples(
    source_dir: Path, data: DataConfig, assigned_split: str | None = None
) -> dict[str, list[Sample]]:
    metadata = read_source_metadata(source_dir)
    label_names = source_label_names(metadata.get("names"), data)
    references = source_split_references(source_dir, metadata)
    if assigned_split and len(references) > 1:
        raise ValueError(f"{source_dir} contains several subsets. Place the whole export directly under raw_data/.")
    source_root = source_path(source_dir, metadata.get("path", "."))
    if not references:
        references = {"train": "images"}
    folders = {}
    for exported_split, reference in references.items():
        split = assigned_split or exported_split
        images = source_images(source_root, reference)
        if not images:
            raise ValueError(f"Declared {split!r} split contains no images: {source_dir}")
        folders[split] = [
            Sample(image, label_path_for(source_root, image), source_dir.name, split, label_names)
            for image in images
        ]
    return folders


def source_path(source_root: Path, value: str) -> Path:
    """Resolve export-relative paths, including CVAT's synthetic data/ prefix."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected a non-empty relative path in {source_root}: {value!r}")
    relative = Path(value.strip())
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Export path must stay inside {source_root}: {value!r}")
    path = source_root / relative
    if not path.exists() and relative.parts and relative.parts[0] == "data":
        path = source_root / Path(*relative.parts[1:])
    return path


def source_images(source_root: Path, reference: str | list[str]) -> list[Path]:
    """Read a subset's image directory or image lists, preserving path spelling."""
    images = []
    references = reference if isinstance(reference, list) else [reference]
    for value in references:
        path = source_path(source_root, value)
        if path.is_dir():
            images.extend(sorted(image for image in path.rglob("*") if is_image(image)))
        elif path.is_file() and path.suffix.lower() == ".txt":
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                if line.strip():
                    image = source_path(source_root, line)
                    if not is_image(image):
                        raise FileNotFoundError(f"Image list {path} references missing or unsupported image: {image}")
                    images.append(image)
        else:
            raise FileNotFoundError(
                f"Missing image directory or image list: {path}. "
                "An unpacked export needs its data.yaml or images/ folder directly inside its source folder."
            )
    return images


def label_path_for(source_dir: Path, image: Path) -> Path | None:
    """Replace the images/ component with labels/, preserving the remaining path."""
    parts = list(image.relative_to(source_dir).parts)
    if "images" not in parts:
        raise ValueError(f"Image must be below an images/ folder: {image}")
    parts[parts.index("images")] = "labels"
    label = (source_dir / Path(*parts)).with_suffix(".txt")
    return label if label.exists() else None


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def check_repeated_images(folders: dict[str, list[Sample]]) -> None:
    """Reject an image that the inputs reference more than once."""
    paths = set()
    for samples in folders.values():
        for sample in samples:
            path = sample.image.resolve()
            if path in paths:
                raise ValueError(f"Image listed more than once: {sample.image}")
            paths.add(path)


# Duplicates and splitting ---------------------------------------------------

# A group of duplicates is kept in the first of these sets that it occurs in.
KEEP_ORDER = ("test", "val", "train")


def remove_duplicates(folders: dict[str, list[Sample]]) -> dict[str, list[Sample]]:
    """Remove duplicate and near-duplicate images within and across sets.

    One image of each group is kept: in test if the group has a test image,
    otherwise in val, otherwise in train; within that set, the image with the
    first path. A training image that repeats an evaluation image is therefore
    dropped, so the scores are not measured on images the model learned from.
    """
    samples = {sample.image: sample for group in folders.values() for sample in group}
    detector = DuplicateDetector(phash_threshold=2, ssim_threshold=0.95)
    clusters = detector.find_duplicates(sorted(samples))

    removed: set[Path] = set()
    for images in clusters.values():
        keep = min(images, key=lambda image: (KEEP_ORDER.index(samples[image].split), image))
        removed.update(set(images) - {keep})

    result = {split: [sample for sample in group if sample.image not in removed] for split, group in folders.items()}
    per_split = [
        f"{split} {len(folders[split]) - len(result[split])}"
        for split in SPLITS if split in folders and len(result[split]) < len(folders[split])
    ]
    if removed:
        log.info(
            "Duplicate check (pHash distance <= 2, SSIM >= 0.95): removed %d of %d images (%s).",
            len(removed), len(samples), ", ".join(per_split),
        )
    else:
        log.info("Duplicate check (pHash distance <= 2, SSIM >= 0.95): no duplicates in %d images.", len(samples))
    return result


def random_split(
    samples: list[Sample], val_split: float, test_split: float, seed: int
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """Seeded split that does not depend on the order the files were found in."""
    if val_split + test_split >= 1:
        raise ValueError(
            "prepare.val_split + prepare.test_split must be < 1. "
            f"Got val_split={val_split}, test_split={test_split}."
        )
    shuffled = sorted(samples, key=lambda sample: (sample.source, str(sample.image)))
    random.Random(seed).shuffle(shuffled)
    test_count = int(len(shuffled) * test_split)
    val_count = int(len(shuffled) * val_split)
    test = shuffled[:test_count]
    val = shuffled[test_count:test_count + val_count]
    train = shuffled[test_count + val_count:]
    return train, val, test


def check_split_sizes(splits: dict[str, list[Sample]], validation_expected: bool) -> None:
    if not splits["train"]:
        raise ValueError("Prepare stage produced 0 training frames.")
    if validation_expected and not splits["val"]:
        raise ValueError("Prepare stage produced 0 validation frames.")
    if not splits["test"]:
        raise ValueError("Prepare stage produced 0 test frames.")


# Writing --------------------------------------------------------------------


def write_dataset(dataset_dir: Path, splits: dict[str, list[Sample]], data: DataConfig, workspace: Path) -> None:
    """Write the dataset into a temporary folder, then replace ``dataset_dir`` with it.

    An interrupted run therefore never leaves a half-written dataset behind.
    Within a split, files are written sorted by source and image path.
    Source paths are relative to the project so the dataset is identical in other checkouts.
    """
    class_names = final_classes(data)
    final_ids = {name: index for index, name in enumerate(class_names)}
    final_names = final_name_by_configured_name(data)

    temporary_dir = dataset_dir.with_name(dataset_dir.name + ".tmp")
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)

    manifest_rows = []
    for split in SPLITS:
        images_dir = temporary_dir / split / "images"
        labels_dir = temporary_dir / split / "labels"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)

        samples = sorted(splits[split], key=lambda sample: (sample.source, str(sample.image)))
        file_names = prepared_file_names(samples)
        for sample, file_name in zip(samples, file_names, strict=True):
            class_id_map = {}
            for label_id, configured_name in sample.label_names.items():
                if configured_name is None:
                    class_id_map[label_id] = None
                else:
                    class_id_map[label_id] = final_ids[final_names[configured_name]]

            link_or_copy(sample.image, images_dir / file_name)
            label_text = normalize_yolo_label(sample.label, class_id_map)
            (labels_dir / file_name).with_suffix(".txt").write_text(label_text, encoding="utf-8")

            source_classes = []
            for label_id in label_class_ids(sample.label):
                if sample.label_names.get(label_id) is not None:
                    source_classes.append(sample.label_names[label_id])
            manifest_rows.append({
                "split": split,
                "file": file_name,
                "source": sample.source,
                "source_image": sample.image.relative_to(workspace).as_posix(),
                "source_classes": ";".join(source_classes),
            })

    write_dataset_yaml(temporary_dir / "dataset.yaml", class_names)
    with (temporary_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(manifest_rows)

    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    temporary_dir.rename(dataset_dir)


def prepared_file_names(samples: list[Sample]) -> list[str]:
    """Unique file names ``<stem>__scene_<source>.<suffix>`` in the order of ``samples``."""
    names = []
    used = set()
    for sample in samples:
        base = f"{sample.image.stem}__scene_{sample.source}"
        name = base + sample.image.suffix
        duplicate_number = 0
        while Path(name).stem in used:
            duplicate_number += 1
            name = f"{base}__dup{duplicate_number}{sample.image.suffix}"
        used.add(Path(name).stem)
        names.append(name)
    return names


def write_dataset_yaml(path: Path, class_names: list[str]) -> None:
    content = {
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": dict(enumerate(class_names)),
    }
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(content, handle, sort_keys=False)


def read_dataset_class_names(dataset_dir: Path) -> dict[int, str]:
    with (dataset_dir / "dataset.yaml").open(encoding="utf-8") as handle:
        content = yaml.safe_load(handle)
    return {int(class_id): str(name) for class_id, name in content["names"].items()}


def read_manifest(dataset_dir: Path) -> list[dict[str, str]]:
    with (dataset_dir / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def link_or_copy(source: Path, target: Path) -> None:
    """Hard-link to save space and time; copy when the file system does not allow links."""
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def log_dataset_summary(dataset_dir: Path) -> None:
    """Log images and objects per final class for every split."""
    class_names = list(read_dataset_class_names(dataset_dir).values())
    header = f"  {'split':<6}{'images':>8}" + "".join(f"{name:>16}" for name in class_names)
    log.info("Prepared dataset %s", dataset_dir)
    log.info(header)
    for split in SPLITS:
        label_files = sorted((dataset_dir / split / "labels").glob("*.txt"))
        object_counts = [0] * len(class_names)
        for label_file in label_files:
            for class_id in label_class_ids(label_file):
                object_counts[class_id] += 1
        row = f"  {split:<6}{len(label_files):>8}" + "".join(f"{count:>16}" for count in object_counts)
        log.info(row)
