from __future__ import annotations

from typing import Any

from trainer_core.config.schema import AppConfig


def _as_mapping(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def normalize_backend_name(model_type: str | None) -> str:
    raw = (model_type or "yolo")
    if not isinstance(raw, str):
        raw = str(raw)
    compact = "".join(ch for ch in raw.strip().lower() if ch.isalnum())
    if compact in {"yolo"}:
        return "yolo"
    if compact in {"rfdetr"}:
        return "rfdetr"
    if compact in {"rtmdet", "mmdet"}:
        return "rtmdet"
    raise ValueError(
        f"Unsupported backend: {model_type!r}. Expected one of: yolo | rfdetr | rtmdet"
    )


def _default_yolo_checkpoint(model_key: str) -> str:
    key = str(model_key or "").strip()
    if key.endswith(".pt"):
        return key
    return f"{key}.pt"


def _first_yolo_fallback_checkpoint(models_cfg: dict) -> str | None:
    for model_key in sorted(models_cfg):
        model_cfg = models_cfg[model_key]
        if not isinstance(model_cfg, dict):
            continue
        backend = normalize_backend_name(model_cfg.get("backend", "yolo"))
        if backend == "yolo":
            checkpoint = model_cfg.get("checkpoint")
            return str(checkpoint) if checkpoint else _default_yolo_checkpoint(str(model_key))
    return None


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
    backend = normalize_backend_name(model_cfg.get("backend", "yolo"))
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

    model_checkpoint = str(model_cfg.get("checkpoint") or _default_yolo_checkpoint(str(selected_model)))
    fallback_checkpoint = model_checkpoint if backend == "yolo" else _first_yolo_fallback_checkpoint(models_cfg)
    if not fallback_checkpoint:
        fallback_checkpoint = "yolov8n.pt"

    resolved = {
        "model_key": str(selected_model),
        "backend": backend,
        "seed": int(getattr(args, "seed", 42)),
        "image_size": int(model_cfg.get("image_size", shared_image_size)),
        "epochs": int(model_cfg.get("epochs", shared_epochs)),
        "batch_size": int(model_cfg.get("batch_size", shared_batch_size)),
        "baseline_weights_path": eval_cfg.get("baseline_weights_path"),
        "fallback_checkpoint": str(fallback_checkpoint),
        "finetune_mode": finetune_enabled,
        "pretrained_model_path": finetune_weights,
        "finetune_lr": finetune_lr,
        "finetune_epochs": finetune_epochs,
        "freeze_backbone": finetune_freeze_backbone,
        "single_phase_overrides": single_phase_overrides,
        "params": config.model_dump(),
    }

    if backend == "yolo":
        resolved["checkpoint"] = model_checkpoint
        if finetune_enabled and finetune_epochs is not None:
            resolved["epochs"] = int(finetune_epochs)
        return resolved

    if backend == "rtmdet":
        config_name = model_cfg.get("config_name", model_cfg.get("variant"))
        config_path = model_cfg.get("config_path")
        if not config_name and not config_path:
            raise ValueError(
                f"models.{selected_model} (backend=rtmdet) must define either config_name or config_path."
            )

        resolved.update(
            {
                "rtmdet_config_name": str(config_name) if config_name else None,
                "rtmdet_config_path": str(config_path) if config_path else None,
                "rtmdet_checkpoint": str(model_cfg["checkpoint"]) if model_cfg.get("checkpoint") else None,
                "rtmdet_cache_dir": str(model_cfg.get("cache_dir", "models/pretrained/rtmdet")),
                "rtmdet_allow_download": bool(model_cfg.get("allow_download", True)),
                "rtmdet_lr": model_cfg.get("lr"),
                "rtmdet_device": model_cfg.get("device"),
                "rtmdet_cleanup_tmp": bool(model_cfg.get("cleanup_tmp", False)),
            }
        )
        return resolved

    from trainer_core.backends import rfdetr as rfdetr_backend

    rfdetr_variant = str(
        model_cfg.get("variant")
        or model_cfg.get("model")
        or rfdetr_backend._infer_rfdetr_variant(str(selected_model))
    )
    rfdetr_batch_size = int(model_cfg.get("batch_size", shared_batch_size))
    explicit_grad_accum = model_cfg.get("grad_accum_steps")
    target_effective_batch = int(model_cfg.get("target_effective_batch", 16))
    if explicit_grad_accum is not None:
        rfdetr_grad_accum = int(explicit_grad_accum)
    else:
        rfdetr_grad_accum = max(1, target_effective_batch // rfdetr_batch_size)

    rfdetr_resolution = rfdetr_backend._normalize_rfdetr_resolution(
        rfdetr_variant,
        model_cfg.get("resolution", None),
        int(model_cfg.get("image_size", shared_image_size)),
    )

    resolved.update(
        {
            "rfdetr_variant": rfdetr_variant,
            "rfdetr_epochs": int(model_cfg.get("epochs", shared_epochs)),
            "rfdetr_batch_size": rfdetr_batch_size,
            "rfdetr_grad_accum": rfdetr_grad_accum,
            "rfdetr_grad_accum_explicit": explicit_grad_accum is not None,
            "rfdetr_target_effective_batch": target_effective_batch,
            "rfdetr_resolution": int(rfdetr_resolution),
            "rfdetr_lr": model_cfg.get("lr"),
            "rfdetr_pretrain": model_cfg.get("pretrain_weights"),
            "rfdetr_grad_ckpt": model_cfg.get("gradient_checkpointing"),
            "rfdetr_extra": model_cfg.get("extra_train_kwargs"),
        }
    )
    return resolved
