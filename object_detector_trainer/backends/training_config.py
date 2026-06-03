from __future__ import annotations

from object_detector_trainer.backends.registry import (
    normalize_backend_name,
    resolve_backend_config,
)
from object_detector_trainer.config.schema import AppConfig


def resolve_training_config(args, config: AppConfig) -> dict:
    train_cfg = config.train.model_dump()
    models_cfg = config.models

    selected_model = getattr(args, "model", None) or config.train.model
    if selected_model not in models_cfg:
        available = ", ".join(sorted(models_cfg.keys())) or "<none>"
        raise ValueError(f"Unknown model key '{selected_model}'. Available models: {available}")

    model_cfg = models_cfg[selected_model]
    backend = normalize_backend_name(model_cfg["backend"])
    shared_image_size = int(config.train.image_size)
    shared_epochs = int(config.train.epochs)
    shared_batch_size = int(config.train.batch_size)

    finetune_cfg = config.train.finetune.model_dump()
    finetune_enabled = bool(config.train.finetune.enabled)
    finetune_weights = finetune_cfg.get("weights")
    finetune_lr = finetune_cfg.get("lr")
    finetune_epochs = finetune_cfg.get("epochs")
    finetune_freeze_backbone = bool(config.train.finetune.freeze_backbone)

    single_phase_overrides = {}
    for key in ("optimizer", "mosaic", "close_mosaic", "mixup", "cos_lr", "lrf", "patience"):
        value = finetune_cfg.get(key, train_cfg.get(key))
        if value is not None:
            single_phase_overrides[key] = value
    if not single_phase_overrides:
        single_phase_overrides = None

    finetune_epochs_value = int(finetune_epochs) if finetune_epochs is not None else None
    image_size = int(model_cfg.get("image_size", shared_image_size))
    epochs = int(model_cfg.get("epochs", shared_epochs))
    batch_size = int(model_cfg.get("batch_size", shared_batch_size))

    if image_size <= 0:
        raise ValueError(
            f"models.{selected_model}.image_size/train.image_size must be > 0, got {image_size}."
        )
    if epochs <= 0:
        raise ValueError(
            f"models.{selected_model}.epochs/train.epochs must be > 0, got {epochs}."
        )
    if batch_size <= 0:
        raise ValueError(
            f"models.{selected_model}.batch_size/train.batch_size must be > 0, got {batch_size}."
        )
    if finetune_enabled and finetune_epochs_value is not None and finetune_epochs_value <= 0:
        raise ValueError(
            f"train.finetune.epochs must be > 0 when fine-tuning is enabled, got {finetune_epochs_value}."
        )

    resolved = {
        "model_key": str(selected_model),
        "backend": backend,
        "seed": int(getattr(args, "seed", 42)),
        "image_size": image_size,
        "epochs": epochs,
        "batch_size": batch_size,
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
