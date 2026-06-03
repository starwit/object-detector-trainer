from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from object_detector_trainer.backends.training_config import resolve_training_config
from object_detector_trainer.config.loader import load_config


def _write_minimal_config(path: Path) -> None:
    payload = {
        "data": {
            "dataset_name": "sample-ds",
            "custom_classes": ["waste"],
            "use_coco_classes": False,
        },
        "prepare": {
            "folder_subsets": {
                "scene_a": 200,
            }
        },
        "train": {
            "model": "rtmdet-tiny",
        },
        "models_defaults": {
            "rtmdet": {
                "cache_dir": "models/pretrained/rtmdet",
                "allow_download": False,
            }
        },
        "models": {
            "rtmdet-tiny": {
                "backend": "rtmdet",
                "asset_id": "rtmdet_tiny_8xb32-300e_coco",
            }
        },
        "evaluation": {
            "baseline_weights_path": "models/current_best/best.pt",
        },
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_folder_subset_integer_values_are_not_coerced_to_float(tmp_path: Path) -> None:
    params_path = tmp_path / "params.yaml"
    _write_minimal_config(params_path)

    cfg = load_config(params_path)

    value = cfg.prepare.folder_subsets["scene_a"]
    assert value == 200
    assert isinstance(value, int)


def test_resolve_training_config_uses_cli_seed(tmp_path: Path) -> None:
    params_path = tmp_path / "params.yaml"
    _write_minimal_config(params_path)
    cfg = load_config(params_path)

    resolved = resolve_training_config(SimpleNamespace(seed=1337, model=None), cfg)
    assert resolved["seed"] == 1337


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("image_size", 0, "image_size.*> 0"),
        ("epochs", 0, "epochs.*> 0"),
        ("batch_size", 0, "batch_size.*> 0"),
    ],
)
def test_resolve_training_config_rejects_non_positive_training_values(
    tmp_path: Path,
    field: str,
    value: int,
    message: str,
) -> None:
    params_path = tmp_path / "params.yaml"
    _write_minimal_config(params_path)
    payload = yaml.safe_load(params_path.read_text(encoding="utf-8"))
    payload["train"][field] = value
    params_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    cfg = load_config(params_path)

    with pytest.raises(ValueError, match=message):
        resolve_training_config(SimpleNamespace(seed=42, model=None), cfg)


def test_resolve_training_config_requires_explicit_rfdetr_variant(tmp_path: Path) -> None:
    params_path = tmp_path / "params.yaml"
    payload = {
        "data": {
            "dataset_name": "sample-ds",
            "custom_classes": ["waste"],
            "use_coco_classes": False,
        },
        "train": {"model": "rfdetr-nano"},
        "models": {
            "rfdetr-nano": {
                "backend": "rfdetr",
                "asset_id": "rf-detr-nano.pth",
            }
        },
        "evaluation": {
            "baseline_weights_path": "models/current_best/best.pt",
        },
    }
    params_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    cfg = load_config(params_path)

    with pytest.raises(ValueError, match="must define variant explicitly"):
        resolve_training_config(SimpleNamespace(seed=42, model=None), cfg)


def test_resolve_training_config_does_not_accept_rtmdet_variant_alias(tmp_path: Path) -> None:
    params_path = tmp_path / "params.yaml"
    payload = {
        "data": {
            "dataset_name": "sample-ds",
            "custom_classes": ["waste"],
            "use_coco_classes": False,
        },
        "train": {"model": "rtmdet-tiny"},
        "models": {
            "rtmdet-tiny": {
                "backend": "rtmdet",
                "variant": "rtmdet_tiny_8xb32-300e_coco",
            }
        },
        "evaluation": {
            "baseline_weights_path": "models/current_best/best.pt",
        },
    }
    params_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    cfg = load_config(params_path)

    with pytest.raises(ValueError, match="must define asset_id"):
        resolve_training_config(SimpleNamespace(seed=42, model=None), cfg)


def test_load_config_requires_baseline_weights_path(tmp_path: Path) -> None:
    params_path = tmp_path / "params.yaml"
    payload = {
        "data": {
            "dataset_name": "sample-ds",
            "custom_classes": ["waste"],
            "use_coco_classes": False,
        },
        "train": {"model": "yolov8n"},
        "models": {
            "yolov8n": {
                "backend": "yolo",
                "asset_id": "yolov8n.pt",
            }
        },
        "evaluation": {},
    }
    params_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="baseline_weights_path"):
        load_config(params_path)
