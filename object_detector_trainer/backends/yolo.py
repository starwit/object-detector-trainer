from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

# Set matplotlib backend to non-GUI before any imports that might use it
# This prevents "Cannot load backend 'tkagg'" errors on headless systems
import matplotlib

matplotlib.use("Agg")

import torch

from object_detector_trainer.backends.assets import (
    download_yolo_checkpoint,
    is_ready_file,
    require_asset_id,
    require_bootstrapped_file,
    resolve_cache_dir,
)
from object_detector_trainer.utils.path_ops import resolve_unique_run_dir, resolve_workspace_path


def resolve_config(
    *,
    model_key: str,
    model_cfg: Mapping[str, Any],
    shared_image_size: int,
    shared_epochs: int,
    shared_batch_size: int,
    finetune_enabled: bool,
    finetune_epochs: int | None,
) -> dict[str, Any]:
    asset_id = require_asset_id(model_key=model_key, model_cfg=model_cfg)
    checkpoint_path = resolve_cache_dir(backend="yolo", model_cfg=model_cfg) / asset_id
    resolved: dict[str, Any] = {"checkpoint": str(checkpoint_path)}
    if finetune_enabled and finetune_epochs is not None:
        resolved["epochs"] = int(finetune_epochs)
    return resolved


def bootstrap_assets(model_key: str, model_cfg: Mapping[str, Any]) -> Path:
    asset_id = require_asset_id(model_key=model_key, model_cfg=model_cfg)
    checkpoint = resolve_cache_dir(backend="yolo", model_cfg=model_cfg) / asset_id

    if not is_ready_file(checkpoint):
        if not bool(model_cfg.get("allow_download", True)):
            raise FileNotFoundError(f"models.{model_key} YOLO checkpoint is missing: {checkpoint}.")
        download_yolo_checkpoint(checkpoint)
    return require_bootstrapped_file(checkpoint, label=f"models.{model_key}.checkpoint")


def build_reload_metadata(model: object, resolved_cfg: Mapping[str, Any]) -> dict[str, object]:
    return {}


def load_model_from_weights(
    candidate_path: Path,
    meta: Mapping[str, object],
    display_name: str,
    yolo_loader,
) -> object:
    model_instance = yolo_loader(str(candidate_path))
    setattr(model_instance, "model_backend", "yolo")
    setattr(model_instance, "model_name", str(display_name))
    if meta.get("model_variant"):
        setattr(model_instance, "model_variant", str(meta["model_variant"]))
    setattr(model_instance, "resolution", int(meta["image_size"]))
    class_names = meta.get("class_names")
    if isinstance(class_names, dict) and class_names:
        setattr(model_instance, "class_names", {int(k): str(v) for k, v in class_names.items()})
    return model_instance


def YOLO(*args, **kwargs):
    from ultralytics import YOLO as UltralyticsYOLO

    return UltralyticsYOLO(*args, **kwargs)


def _resolve_required_weights(path_like: str | Path, *, label: str) -> Path:
    path = resolve_workspace_path(path_like)
    if path is None or not path.is_file() or path.stat().st_size == 0:
        if label == "train.finetune.weights":
            raise FileNotFoundError(
                "Fine-tuning mode is enabled, but "
                f"{label} must point to an existing non-empty file: {path}"
            )
        raise FileNotFoundError(f"{label} must point to an existing non-empty file: {path}")
    return path


def train_backend(
    *,
    training_path: Path,
    test_path: Path,
    dataset_name: str,
    resolved_cfg: dict,
    experiment_name: str | None,
) -> tuple[object, Path, str, int, int]:
    finetune_mode = bool(resolved_cfg.get("finetune_mode", False))
    run_name = experiment_name

    if finetune_mode:
        pretrained_model_path = resolved_cfg.get("pretrained_model_path")
        if not pretrained_model_path:
            raise ValueError(
                "Fine-tuning mode is enabled, but train.finetune.weights is not set."
            )
        pretrained_model = _resolve_required_weights(
            str(pretrained_model_path),
            label="train.finetune.weights",
        )
        print(f"Fine-tuning mode enabled. Loading pre-trained model: {pretrained_model}")
        model = YOLO(str(pretrained_model))
        run_name = f"{experiment_name}-finetune" if experiment_name else None
    else:
        checkpoint_path = _resolve_required_weights(
            str(resolved_cfg["checkpoint"]),
            label="models.<key>.checkpoint",
        )
        print(f"Using YOLO checkpoint: {checkpoint_path}")
        model = YOLO(str(checkpoint_path))

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    runs_root = Path(str(resolved_cfg.get("runs_root", "runs")))
    runs_root.mkdir(parents=True, exist_ok=True)
    name = resolve_unique_run_dir(runs_root, run_name).name if run_name else None

    train_args = {
        "data": str(training_path / "dataset.yaml"),
        "epochs": int(resolved_cfg["epochs"]),
        "imgsz": int(resolved_cfg["image_size"]),
        "batch": int(resolved_cfg["batch_size"]),
        "device": device,
        "workers": 4,
        # Keep runtime offline/strict: Ultralytics' AMP checks auto-download
        # YOLO11n if it's missing. Users can explicitly re-enable AMP via
        # single_phase_overrides={"amp": True} once assets are provisioned.
        "amp": False,
        "project": str(runs_root),
    }
    if name:
        train_args["name"] = name

    if finetune_mode:
        finetune_lr = resolved_cfg.get("finetune_lr")
        if finetune_lr is not None:
            train_args["lr0"] = finetune_lr
            print(f"Using fine-tuning learning rate: {finetune_lr}")

        if bool(resolved_cfg.get("freeze_backbone", False)):
            # Freeze backbone layers (layers 0-9 typically for YOLOv8)
            train_args["freeze"] = list(range(10))
            print("Freezing backbone layers for fine-tuning")

        # Fine-tune overrides stay explicit so normal training keeps the
        # default Ultralytics schedule.
        single_phase_overrides = resolved_cfg.get("single_phase_overrides")
        if single_phase_overrides:
            sp = {k: v for k, v in single_phase_overrides.items() if v is not None}
            if sp:
                print(f"Applying single-phase overrides: {sorted(sp.keys())}")
                train_args.update(sp)

    results = model.train(**train_args)
    model.model_backend = "yolo"
    model.model_name = str(resolved_cfg["model_key"])
    model.model_variant = str(resolved_cfg["model_key"])
    model.resolution = int(resolved_cfg["image_size"])
    return (
        model,
        Path(results.save_dir),
        run_name or resolved_cfg["model_key"],
        int(resolved_cfg["image_size"]),
        int(resolved_cfg["epochs"]),
    )


__all__ = [
    "bootstrap_assets",
    "build_reload_metadata",
    "load_model_from_weights",
    "resolve_config",
    "train_backend",
]
