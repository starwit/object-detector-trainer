"""Small builders for test inputs: images, label files, sources and params.yaml."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml


def write_image(path: Path, value: int = 100, size: int = 64) -> None:
    """A noise image; different values give images that are not duplicates of each other."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.random.default_rng(value).integers(0, 256, (size, size, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def write_label(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def minimal_params(**sections) -> dict:
    """A valid params.yaml content; keyword arguments replace whole sections."""
    params = {
        "data": {"dataset_name": "sample", "custom_classes": ["waste"]},
        "train": {"model": "yolo-small"},
        "models": {"yolo-small": {"backend": "yolo", "asset_id": "yolov8n.pt"}},
    }
    params.update(sections)
    return params


def write_params(path: Path, params: dict) -> Path:
    path.write_text(yaml.safe_dump(params), encoding="utf-8")
    return path
