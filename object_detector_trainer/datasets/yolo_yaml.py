"""YOLO dataset.yaml parsing/writing helpers.

This keeps YAML class-name normalization and write format in one place so
backends/evaluation/dataprep do not each re-implement slightly different rules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import yaml


def load_yolo_dataset_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid dataset YAML at {path}: expected mapping, got {type(payload)}")
    return payload


def normalize_class_names(payload: dict) -> dict[int, str]:
    names = payload.get("names", {})
    if isinstance(names, list):
        return {i: str(name) for i, name in enumerate(names)}
    if isinstance(names, dict):
        parsed: dict[int, str] = {}
        for raw_key, raw_name in names.items():
            try:
                cls_id = int(raw_key)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Invalid class id in dataset.yaml names mapping: {raw_key!r}. "
                    "Expected integer keys (e.g. 0, 1) or numeric strings (e.g. '0', '1')."
                ) from e
            parsed[cls_id] = str(raw_name)
        return {k: parsed[k] for k in sorted(parsed)}
    return {}


def get_class_ids(payload: dict) -> list[int]:
    return list(normalize_class_names(payload).keys())


def get_dataset_classes(path: str | Path) -> tuple[dict[int, str], list[int]]:
    payload = load_yolo_dataset_yaml(path)
    class_names = normalize_class_names(payload)
    return class_names, list(class_names.keys())


def write_yolo_dataset_yaml(
    *,
    output_path: Path,
    dataset_root: Path,
    class_names: Mapping[int, str] | list[str],
    train_rel: str = "train/images",
    val_rel: str = "val/images",
) -> None:
    if isinstance(class_names, Mapping):
        normalized_names = {int(k): str(v) for k, v in class_names.items()}
        normalized_names = {k: normalized_names[k] for k in sorted(normalized_names)}
    else:
        normalized_names = {i: str(name) for i, name in enumerate(class_names)}

    payload = {
        "path": str(dataset_root.resolve()),
        "train": str(train_rel),
        "val": str(val_rel),
        "nc": len(normalized_names),
        "names": normalized_names,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
