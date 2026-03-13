from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from object_detector_trainer.config.overrides import apply_set_overrides
from object_detector_trainer.config.schema import AppConfig


def _as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)  # type: ignore[arg-type]
        else:
            merged[key] = value
    return merged


def _normalize_backend_key(value: Any) -> str:
    return "".join(ch for ch in str(value).strip().lower() if ch.isalnum())


def _apply_models_defaults(raw_config: dict[str, Any]) -> dict[str, Any]:
    defaults_by_backend = _as_mapping(raw_config.get("models_defaults"))
    if not defaults_by_backend:
        return raw_config

    models_cfg = raw_config.get("models")
    if not isinstance(models_cfg, dict) or not models_cfg:
        return raw_config

    global_defaults = _as_mapping(defaults_by_backend.get("*"))

    merged_models: dict[str, Any] = {}
    for model_key, model_cfg_raw in models_cfg.items():
        if not isinstance(model_cfg_raw, dict):
            merged_models[str(model_key)] = model_cfg_raw
            continue

        backend_raw = model_cfg_raw.get("backend")
        if backend_raw is None:
            merged_models[str(model_key)] = model_cfg_raw
            continue

        backend_key = _normalize_backend_key(backend_raw)
        backend_defaults = _as_mapping(defaults_by_backend.get(backend_key))
        combined_defaults = _deep_merge(global_defaults, backend_defaults) if global_defaults else backend_defaults
        merged_models[str(model_key)] = (
            _deep_merge(combined_defaults, model_cfg_raw) if combined_defaults else model_cfg_raw
        )

    merged = dict(raw_config)
    merged["models"] = merged_models
    return merged


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

    merged: dict[str, Any] = _apply_models_defaults(dict(raw))
    if args is not None:
        merged = _apply_direct_arg_overrides(merged, args)
        merged = apply_set_overrides(merged, getattr(args, "set", None))

    return AppConfig.model_validate(merged)
