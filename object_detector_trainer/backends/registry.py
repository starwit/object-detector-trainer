"""Thin backend registry.

Backend-specific behavior belongs in the backend module itself. To add a new
backend, implement the same small surface as ``yolo``, ``rfdetr``, and
``rtmdet``:

- ``resolve_config(...)``
- ``bootstrap_assets(model_key, model_cfg)``
- ``train_backend(...)``
- ``build_reload_metadata(model, resolved_cfg)``
- ``load_model_from_weights(candidate_path, meta, display_name, yolo_loader)``

Then add the backend to ``BACKEND_MODULES`` and ``REQUIRED_RESOLVED_FIELDS``.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping


BACKEND_MODULES = {
    "yolo": "object_detector_trainer.backends.yolo",
    "rfdetr": "object_detector_trainer.backends.rfdetr",
    "rtmdet": "object_detector_trainer.backends.rtmdet",
}

SUPPORTED_BACKEND_NAMES = tuple(BACKEND_MODULES)
REQUIRED_RESOLVED_FIELDS = {
    "yolo": ("checkpoint",),
    "rfdetr": (
        "rfdetr_variant",
        "rfdetr_resolution",
        "rfdetr_batch_size",
        "rfdetr_checkpoint",
    ),
    "rtmdet": ("rtmdet_config_name", "rtmdet_cache_dir"),
}


def normalize_backend_name(model_type: str | None) -> str:
    backend = str(model_type or "").strip().lower()
    if backend in BACKEND_MODULES:
        return backend
    expected = " | ".join(SUPPORTED_BACKEND_NAMES)
    raise ValueError(f"Unsupported backend: {model_type!r}. Expected one of: {expected}")


def _backend_module(backend: str | None) -> ModuleType:
    return import_module(BACKEND_MODULES[normalize_backend_name(backend)])


def resolve_backend_config(
    *,
    backend: str,
    model_key: str,
    model_cfg: Mapping[str, Any],
    shared_image_size: int,
    shared_epochs: int,
    shared_batch_size: int,
    finetune_enabled: bool,
    finetune_epochs: int | None,
) -> dict[str, Any]:
    return _backend_module(backend).resolve_config(
        model_key=model_key,
        model_cfg=model_cfg,
        shared_image_size=shared_image_size,
        shared_epochs=shared_epochs,
        shared_batch_size=shared_batch_size,
        finetune_enabled=finetune_enabled,
        finetune_epochs=finetune_epochs,
    )


def bootstrap_model_assets(model_key: str, model_cfg: Mapping[str, Any]) -> Path:
    backend = normalize_backend_name(model_cfg.get("backend"))
    return _backend_module(backend).bootstrap_assets(model_key, model_cfg)


def train_backend(backend: str, **kwargs: Any):
    return _backend_module(backend).train_backend(**kwargs)


def build_reload_metadata(
    backend: str,
    model: object,
    resolved_cfg: Mapping[str, Any],
) -> dict[str, object]:
    return _backend_module(backend).build_reload_metadata(model, resolved_cfg)


def load_backend_model_from_weights(
    backend: str,
    candidate_path: Path,
    meta: Mapping[str, object],
    display_name: str,
    yolo_loader,
) -> object:
    return _backend_module(backend).load_model_from_weights(
        candidate_path=candidate_path,
        meta=meta,
        display_name=display_name,
        yolo_loader=yolo_loader,
    )


__all__ = [
    "BACKEND_MODULES",
    "REQUIRED_RESOLVED_FIELDS",
    "SUPPORTED_BACKEND_NAMES",
    "bootstrap_model_assets",
    "build_reload_metadata",
    "load_backend_model_from_weights",
    "normalize_backend_name",
    "resolve_backend_config",
    "train_backend",
]
