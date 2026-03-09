from __future__ import annotations

from typing import Any

from object_detector_trainer.backends.registry import (
    normalize_backend_name,
    resolve_backend_config,
)
from object_detector_trainer.config.schema import AppConfig


def _as_mapping(value: Any) -> dict:
    return value if isinstance(value, dict) else {}

def resolve_training_config(args, config: AppConfig) -> dict:
    train_cfg = config.train.model_dump()
    models_cfg = config.models
    eval_cfg = config.evaluation.model_dump()

    selected_model = getattr(args, "model", None) or train_cfg.get("model")
    if not selected_model:
        raise ValueError("Missing train.model in config and no --model override was provided.")
    if selected_model not in models_cfg:
        available = ", ".join(sorted(models_cfg.keys())) or "<none>"
        raise ValueError(f"Unknown model key '{selected_model}'. Available models: {available}")

    model_cfg = _as_mapping(models_cfg.get(selected_model, {}))
    backend_raw = model_cfg.get("backend")
    if backend_raw is None:
        raise ValueError(f"models.{selected_model} must define backend explicitly.")
    backend = normalize_backend_name(backend_raw)
    shared_image_size = int(train_cfg.get("image_size", 640))
    shared_epochs = int(train_cfg.get("epochs", 100))
    shared_batch_size = int(train_cfg.get("batch_size", 8))

    finetune_cfg = _as_mapping(train_cfg.get("finetune", {}))
    finetune_enabled = bool(finetune_cfg.get("enabled", False))
    finetune_weights = finetune_cfg.get("weights")
    finetune_lr = finetune_cfg.get("lr")
    finetune_epochs = finetune_cfg.get("epochs")
    finetune_freeze_backbone = bool(finetune_cfg.get("freeze_backbone", False))

    single_phase_overrides = {}
    for key in ("optimizer", "mosaic", "close_mosaic", "mixup", "cos_lr", "lrf", "patience"):
        value = finetune_cfg.get(key, train_cfg.get(key))
        if value is not None:
            single_phase_overrides[key] = value
    if not single_phase_overrides:
        single_phase_overrides = None

    finetune_epochs_value = int(finetune_epochs) if finetune_epochs is not None else None
    resolved = {
        "model_key": str(selected_model),
        "backend": backend,
        "seed": int(getattr(args, "seed", 42)),
        "image_size": int(model_cfg.get("image_size", shared_image_size)),
        "epochs": int(model_cfg.get("epochs", shared_epochs)),
        "batch_size": int(model_cfg.get("batch_size", shared_batch_size)),
        "baseline_weights_path": eval_cfg.get("baseline_weights_path"),
        "finetune_mode": finetune_enabled,
        "pretrained_model_path": finetune_weights,
        "finetune_lr": finetune_lr,
        "finetune_epochs": finetune_epochs,
        "freeze_backbone": finetune_freeze_backbone,
        "single_phase_overrides": single_phase_overrides,
        "params": config.model_dump(),
    }
    resolved.update(
        resolve_backend_config(
            backend=backend,
            model_key=str(selected_model),
            model_cfg=model_cfg,
            shared_image_size=shared_image_size,
            shared_epochs=shared_epochs,
            shared_batch_size=shared_batch_size,
            finetune_enabled=finetune_enabled,
            finetune_epochs=finetune_epochs_value,
        )
    )
    return resolved
