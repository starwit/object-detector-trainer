from __future__ import annotations

from pathlib import Path

# Set matplotlib backend to non-GUI before any imports that might use it
# This prevents "Cannot load backend 'tkagg'" errors on headless systems
import matplotlib

matplotlib.use("Agg")

import torch


def YOLO(*args, **kwargs):
    from ultralytics import YOLO as UltralyticsYOLO

    return UltralyticsYOLO(*args, **kwargs)


def _resolve_save_dir(model, results, default: Path) -> Path:
    """Return Ultralytics' actual save_dir if available, else default.

    Keeps train_model concise while handling version differences
    (trainer.save_dir vs. results.save_dir).
    """
    trainer = getattr(model, "trainer", None)
    candidate = getattr(trainer, "save_dir", None) if trainer is not None else None
    if isinstance(candidate, (str, Path)):
        return Path(candidate)

    candidate = getattr(results, "save_dir", None)
    if isinstance(candidate, (str, Path)):
        return Path(candidate)

    return default


def train_model(
    dataset_path,
    checkpoint,
    image_size,
    batch_size,
    experiment_name,
    epochs=100,
    finetune_mode=False,
    pretrained_model_path=None,
    finetune_lr=None,
    freeze_backbone=False,
    single_phase_overrides: dict | None = None,
):
    """
    Train the YOLO model on the specified dataset.

    Args:
        dataset_path (Path): Path to the dataset directory.
        checkpoint (str): YOLO checkpoint to train from (e.g., 'yolov8m.pt').
        image_size (int): Size of images for training.
        batch_size (int): Batch size for training.
        experiment_name (str): Name for the experiment.
        epochs (int): Number of training epochs.
        finetune_mode (bool): Whether to use fine-tuning mode.
        pretrained_model_path (str): Path to pre-trained model for fine-tuning.
        finetune_lr (float): Learning rate for fine-tuning.
        freeze_backbone (bool): Whether to freeze backbone layers during fine-tuning.

    Returns:
        model (YOLO): The trained YOLO model.
        results: Training results.
        Path: Directory path of the training output.
    """
    # Choose model based on fine-tuning mode
    pretrained_model = None
    if pretrained_model_path:
        candidate = Path(pretrained_model_path)
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        if candidate.exists() and candidate.stat().st_size > 0:
            pretrained_model = candidate

    if finetune_mode:
        if not pretrained_model_path:
            raise ValueError(
                "Fine-tuning mode is enabled, but train.finetune.weights is not set."
            )
        if pretrained_model is None:
            raise FileNotFoundError(
                "Fine-tuning mode is enabled, but the configured weights are missing or still "
                f"the 0-byte DVC placeholder: {pretrained_model_path}"
            )
        print(f"Fine-tuning mode enabled. Loading pre-trained model: {pretrained_model_path}")
        model = YOLO(str(pretrained_model))
        experiment_name = f"{experiment_name}-finetune"
    else:
        checkpoint_name = str(checkpoint)
        print(f"Using YOLO checkpoint: {checkpoint_name}")
        model = YOLO(checkpoint_name)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Define project name and ensure unique run name
    project = "runs"
    name = None
    if experiment_name:
        base_name = experiment_name
        name = base_name
        run_number = 1
        while (Path(project) / name).exists():
            name = f"{base_name}_{run_number}"
            run_number += 1

    # Prepare training arguments
    train_args = {
        "data": str(dataset_path / "dataset.yaml"),
        "epochs": epochs,
        "imgsz": image_size,
        "batch": batch_size,
        "device": device,
        "workers": 4,
        "amp": True,
        "project": project,
    }
    if name:
        train_args["name"] = name

    # Two-phase path removed for simplicity; single-phase fine-tune only

    # Single-phase (original) path
    if finetune_mode:
        if finetune_lr is not None:
            train_args["lr0"] = finetune_lr
            print(f"Using fine-tuning learning rate: {finetune_lr}")

        if freeze_backbone:
            # Freeze backbone layers (layers 0-9 typically for YOLOv8)
            train_args["freeze"] = list(range(10))
            print("Freezing backbone layers for fine-tuning")

        # Allow minimal, explicit overrides (optimizer, mosaic, mixup, close_mosaic, cos_lr, lrf, patience)
        if single_phase_overrides:
            sp = {k: v for k, v in single_phase_overrides.items() if v is not None}
            if sp:
                print(f"Applying single-phase overrides: {sorted(sp.keys())}")
                train_args.update(sp)

    results = model.train(**train_args)

    default_dir = Path(project) / (name if name else "train")
    output_dir = _resolve_save_dir(model, results, default_dir)

    # Keep return signature compatible with tests: (model, results, output_dir)
    return model, results, output_dir


def train_yolo(
    *,
    training_path: Path,
    resolved_cfg: dict,
    experiment_name: str | None,
) -> tuple[object, object, Path, str | None]:
    """Backend wrapper used by ``train_pipeline.py``."""

    model, results, train_output_dir = train_model(
        training_path,
        resolved_cfg["checkpoint"],
        int(resolved_cfg["image_size"]),
        int(resolved_cfg["batch_size"]),
        experiment_name,
        epochs=int(resolved_cfg["epochs"]),
        finetune_mode=bool(resolved_cfg.get("finetune_mode", False)),
        pretrained_model_path=resolved_cfg.get("pretrained_model_path"),
        finetune_lr=resolved_cfg.get("finetune_lr"),
        freeze_backbone=bool(resolved_cfg.get("freeze_backbone", False)),
        single_phase_overrides=resolved_cfg.get("single_phase_overrides"),
    )

    final_experiment_name = (
        (
            f"{experiment_name}-finetune"
            if experiment_name
            and bool(resolved_cfg.get("finetune_mode", False))
            and resolved_cfg.get("pretrained_model_path")
            and Path(str(resolved_cfg.get("pretrained_model_path"))).exists()
            else experiment_name
        )
    )
    return model, results, train_output_dir, final_experiment_name


def train_backend(
    *,
    training_path: Path,
    test_path: Path,
    dataset_name: str,
    resolved_cfg: dict,
    experiment_name: str | None,
) -> tuple[object, Path, str, int, int]:
    model, _results, train_output_dir, final_experiment_name = train_yolo(
        training_path=training_path,
        resolved_cfg=resolved_cfg,
        experiment_name=experiment_name,
    )
    return (
        model,
        train_output_dir,
        final_experiment_name or resolved_cfg["model_key"],
        int(resolved_cfg["image_size"]),
        int(resolved_cfg["epochs"]),
    )


__all__ = ["train_backend", "train_model", "train_yolo"]
