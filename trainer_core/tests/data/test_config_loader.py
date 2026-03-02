from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

from trainer_core.backends.training_config import resolve_training_config
from trainer_core.config.loader import load_config


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
        "models": {
            "rtmdet-tiny": {
                "backend": "rtmdet",
                "config_name": "rtmdet_tiny_8xb32-300e_coco",
            }
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
