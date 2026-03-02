from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from trainer_core.config.overrides import apply_set_overrides
from trainer_core.config.schema import AppConfig


def _as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _apply_direct_arg_overrides(raw_config: dict[str, Any], args) -> dict[str, Any]:
    merged = dict(raw_config)
    data_cfg = dict(_as_mapping(merged.get("data", {})))
    train_cfg = dict(_as_mapping(merged.get("train", {})))
    prepare_cfg = dict(_as_mapping(merged.get("prepare", {})))

    dataset_name = getattr(args, "dataset_name", None)
    if dataset_name:
        data_cfg["dataset_name"] = str(dataset_name)
    model_name = getattr(args, "model", None)
    if model_name:
        train_cfg["model"] = str(model_name)

    for attr_name, key in (
        ("val_split", "val_split"),
        ("test_split", "test_split"),
        ("augment_multiplier", "augment_multiplier"),
    ):
        val = getattr(args, attr_name, None)
        if val is not None:
            prepare_cfg[key] = val

    if data_cfg:
        merged["data"] = data_cfg
    if train_cfg:
        merged["train"] = train_cfg
    if prepare_cfg:
        merged["prepare"] = prepare_cfg
    return merged


def load_config(config_path: str | Path = "params.yaml", args=None) -> AppConfig:
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration root must be a mapping in {cfg_path}")

    merged: dict[str, Any] = dict(raw)
    if args is not None:
        merged = _apply_direct_arg_overrides(merged, args)
        merged = apply_set_overrides(merged, getattr(args, "set", None))

    return AppConfig.model_validate(merged)

