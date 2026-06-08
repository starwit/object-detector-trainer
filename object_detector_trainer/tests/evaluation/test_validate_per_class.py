from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from object_detector_trainer.evaluation.validate import _extract_per_class_metrics


def _write_dataset(tmp_path: Path) -> Path:
    dataset_root = tmp_path / "dataset"
    label_dir = dataset_root / "val" / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    (dataset_root / "val" / "images").mkdir(parents=True, exist_ok=True)
    (label_dir / "one.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n1 0.4 0.4 0.3 0.3\n",
        encoding="utf-8",
    )

    dataset_yaml = tmp_path / "dataset.yaml"
    dataset_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(dataset_root),
                "val": "val/images",
                "names": {0: "waste", 1: "cigarette", 2: "leaves_dense"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return dataset_yaml


def test_extract_per_class_filters_classes_without_ground_truth(tmp_path: Path) -> None:
    dataset_yaml = _write_dataset(tmp_path)
    metrics = SimpleNamespace(
        per_class={
            "waste": {
                "precision": 0.8,
                "recall": 0.7,
                "map50": 0.6,
                "map": 0.5,
                "f1_score": 0.7467,
            },
            "leaves_dense": {
                "precision": 0.0,
                "recall": 0.0,
                "map50": 0.0,
                "map": 0.0,
                "f1_score": 0.0,
            },
        }
    )

    per_class = _extract_per_class_metrics(metrics, str(dataset_yaml))

    assert set(per_class) == {"waste", "cigarette"}
    assert per_class["waste"]["precision"] == pytest.approx(0.8)
    assert per_class["cigarette"] == {
        "precision": 0.0,
        "recall": 0.0,
        "map50": 0.0,
        "map": 0.0,
        "f1_score": 0.0,
    }


def test_extract_per_class_normalizes_ultralytics_box_metrics(tmp_path: Path) -> None:
    dataset_yaml = _write_dataset(tmp_path)
    box = SimpleNamespace(
        ap_class_index=np.array([0, 2]),
        p=np.array([0.8, 0.2]),
        r=np.array([0.7, 0.1]),
        ap50=np.array([0.6, 0.05]),
        ap=np.array([0.5, 0.02]),
    )
    metrics = SimpleNamespace(box=box)

    per_class = _extract_per_class_metrics(metrics, str(dataset_yaml))

    assert set(per_class) == {"waste", "cigarette"}
    assert per_class["waste"]["precision"] == pytest.approx(0.8)
    assert per_class["cigarette"]["map50"] == pytest.approx(0.0)
