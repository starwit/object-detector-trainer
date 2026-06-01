from __future__ import annotations

import contextlib
import shutil
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from object_detector_trainer.backends.assets import (
    is_ready_file,
    require_asset_id,
    require_bootstrapped_file,
    resolve_cache_dir,
)
from object_detector_trainer.utils.path_ops import (
    resolve_unique_run_dir,
    resolve_workspace_path,
    safe_dataset_dirname,
)


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
    variant = str(model_cfg.get("variant") or "").strip()
    if not variant:
        raise ValueError(
            f"models.{model_key} (backend=rfdetr) must define variant explicitly."
        )

    asset_id = require_asset_id(model_key=model_key, model_cfg=model_cfg)
    checkpoint_path = resolve_cache_dir(backend="rfdetr", model_cfg=model_cfg) / asset_id

    batch_size = int(model_cfg.get("batch_size", shared_batch_size))
    explicit_grad_accum = model_cfg.get("grad_accum_steps")
    target_effective_batch = int(model_cfg.get("target_effective_batch", 16))
    grad_accum = (
        int(explicit_grad_accum)
        if explicit_grad_accum is not None
        else max(1, target_effective_batch // batch_size)
    )

    resolution = _normalize_rfdetr_resolution(
        variant,
        model_cfg.get("resolution", None),
        int(model_cfg.get("image_size", shared_image_size)),
    )

    return {
        "rfdetr_variant": variant,
        "rfdetr_epochs": int(model_cfg.get("epochs", shared_epochs)),
        "rfdetr_batch_size": batch_size,
        "rfdetr_grad_accum": grad_accum,
        "rfdetr_grad_accum_explicit": explicit_grad_accum is not None,
        "rfdetr_target_effective_batch": target_effective_batch,
        "rfdetr_resolution": int(resolution),
        "rfdetr_lr": model_cfg.get("lr"),
        "rfdetr_checkpoint": str(checkpoint_path),
        "rfdetr_grad_ckpt": model_cfg.get("gradient_checkpointing"),
        "rfdetr_extra": model_cfg.get("extra_train_kwargs"),
    }


def bootstrap_assets(model_key: str, model_cfg: Mapping[str, Any]) -> Path:
    asset_id = require_asset_id(model_key=model_key, model_cfg=model_cfg)
    checkpoint_path = resolve_cache_dir(backend="rfdetr", model_cfg=model_cfg) / asset_id

    if not is_ready_file(checkpoint_path):
        if not bool(model_cfg.get("allow_download", True)):
            raise FileNotFoundError(
                f"models.{model_key} RF-DETR checkpoint is missing: {checkpoint_path}."
            )

        from rfdetr.assets.model_weights import download_pretrain_weights

        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.chdir(checkpoint_path.parent):
            download_pretrain_weights(checkpoint_path.name)

    return require_bootstrapped_file(
        checkpoint_path,
        label=f"models.{model_key}.checkpoint",
    )


def build_reload_metadata(model: object, resolved_cfg: Mapping[str, Any]) -> dict[str, object]:
    return {
        "model_variant": str(
            getattr(model, "model_variant", None) or resolved_cfg["rfdetr_variant"]
        )
    }


def load_model_from_weights(
    candidate_path: Path,
    meta: Mapping[str, object],
    display_name: str,
    yolo_loader=None,
) -> object:
    from object_detector_trainer.wrappers.rfdetr import RFDETRModelAdapter

    model_variant = str(meta.get("model_variant") or "").strip().lower()
    if not model_variant:
        raise ValueError("RF-DETR model metadata must include model_variant.")
    resolution = int(meta["image_size"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rfdetr_model = _get_rfdetr_model(
        model_variant=model_variant,
        checkpoint_path=str(candidate_path),
        device=device,
        resolution=int(resolution),
    )
    return RFDETRModelAdapter(
        rfdetr_model,
        model_name=str(display_name),
        resolution=int(resolution),
        model_variant=model_variant,
    )


@contextlib.contextmanager
def _patched_rfdetr_best_metric_holder(*, init_res: float) -> None:
    """Patch RF-DETR to always emit a canonical best checkpoint.

    Upstream RF-DETR only writes checkpoint_best_regular.pth when mAP strictly
    improves over the initial best (default 0.0). On tiny smoke fixtures, mAP
    can remain at 0.0 which leaves no best checkpoint and crashes later when
    RF-DETR tries to copy it into checkpoint_best_total.pth.

    We patch rfdetr.main.BestMetricHolder for the duration of training so the
    first evaluation is always treated as "best" (init_res = -inf). This is not a
    fallback to a different model; it makes the canonical output artifact
    deterministic and prevents silent substitution.
    """

    import rfdetr.main as rfdetr_main
    from rfdetr.util.utils import BestMetricHolder as OriginalBestMetricHolder

    original_symbol = rfdetr_main.BestMetricHolder
    patched_init_res = float(init_res)

    class PatchedBestMetricHolder(OriginalBestMetricHolder):  # type: ignore[misc]
        def __init__(
            self,
            init_res: float = patched_init_res,
            better: str = "large",
            use_ema: bool = False,
        ) -> None:
            super().__init__(init_res=init_res, better=better, use_ema=use_ema)

    rfdetr_main.BestMetricHolder = PatchedBestMetricHolder  # type: ignore[assignment]
    try:
        yield
    finally:
        rfdetr_main.BestMetricHolder = original_symbol  # type: ignore[assignment]


def _resolve_required_checkpoint(path_like: str | Path | None) -> Path:
    if not path_like:
        raise ValueError(
            "RF-DETR training requires a resolved local checkpoint path. "
            "Run bootstrap first or check models.<key>.asset_id / cache_dir."
        )
    checkpoint = resolve_workspace_path(path_like)
    return require_bootstrapped_file(checkpoint, label="Resolved RF-DETR checkpoint")


def _normalize_rfdetr_resolution(
    model_variant: str,
    resolution: int | None,
    shared_image_size: int,
) -> int:
    """Adjust resolution to RF-DETR's official patch/window divisors."""
    divisors = {
        "nano": 32,
        "small": 32,
        "medium": 32,
        "base": 56,
        "large": 56,
    }
    variant = model_variant.lower()
    if variant not in divisors:
        raise ValueError(f"Unsupported RF-DETR model variant: {model_variant}")

    if resolution is None:
        resolution = int(shared_image_size)
    divisor = divisors[variant]
    if resolution % divisor != 0:
        adjusted = (resolution // divisor) * divisor
        if adjusted < divisor:
            adjusted = divisor
        print(
            f"Warning: RF-DETR resolution {resolution} is not divisible by {divisor}. "
            f"Using {adjusted}."
        )
        resolution = adjusted
    return int(resolution)


def _get_rfdetr_model(
    model_variant: str,
    checkpoint_path: str | None = None,
    device: str | None = None,
    resolution: int | None = None,
    gradient_checkpointing: bool | None = None,
):
    """Return an initialized RF-DETR model based on a variant name."""
    import rfdetr

    variant = model_variant.lower()
    class_name_by_variant = {
        "nano": "RFDETRNano",
        "small": "RFDETRSmall",
        "medium": "RFDETRMedium",
        "base": "RFDETRBase",
        "large": "RFDETRLarge",
    }
    class_name = class_name_by_variant.get(variant)
    if class_name is None:
        raise ValueError(f"Unsupported RF-DETR model variant: {model_variant}")

    model_cls = getattr(rfdetr, class_name)

    init_kwargs: dict[str, object] = {}
    if checkpoint_path:
        init_kwargs["pretrain_weights"] = checkpoint_path
    if device:
        init_kwargs["device"] = device
    if resolution is not None:
        init_kwargs["resolution"] = int(resolution)
    if gradient_checkpointing is not None:
        init_kwargs["gradient_checkpointing"] = bool(gradient_checkpointing)
    return model_cls(**init_kwargs) if init_kwargs else model_cls()


def train_rfdetr(
    dataset_dir: Path,
    output_root: Path,
    experiment_name: str | None,
    model_variant: str,
    epochs: int,
    batch_size: int,
    grad_accum_steps: int,
    lr: float | None,
    resolution: int,
    checkpoint_path: str | None = None,
    gradient_checkpointing: bool | None = None,
    extra_train_kwargs: dict | None = None,
):
    """Train an RF-DETR model using the Roboflow dataset loader.

    RF-DETR's ``dataset_file='roboflow'`` mode auto-detects either COCO-style
    or YOLO-style exports under ``dataset_dir``. Our pipeline supplies a YOLO
    layout bridge (data.yaml + train/valid/test image+label dirs).
    """
    run_name = experiment_name or "rfdetr-train"
    output_dir = resolve_unique_run_dir(output_root, run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = _resolve_required_checkpoint(checkpoint_path)

    model = _get_rfdetr_model(
        model_variant,
        checkpoint_path=str(checkpoint),
        device=device,
        resolution=resolution,
        gradient_checkpointing=gradient_checkpointing,
    )

    train_kwargs: dict[str, object] = {
        "dataset_dir": str(dataset_dir),
        "dataset_file": "roboflow",
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "grad_accum_steps": int(grad_accum_steps),
        "output_dir": str(output_dir),
        "resolution": int(resolution),
        "run_test": True,
    }
    if lr is not None:
        train_kwargs["lr"] = float(lr)
    if extra_train_kwargs:
        train_kwargs.update(extra_train_kwargs)

    with _patched_rfdetr_best_metric_holder(init_res=float("-inf")):
        model.train(**train_kwargs)
    return model, output_dir


def _save_rfdetr_weights(output_dir: Path) -> None:
    """Copy the best RF-DETR checkpoint into ``output_dir/weights/best.pt``.

    This mirrors the YOLO convention so that ``export_baseline.py`` (which looks
    for ``<run>/weights/best.pt``) works identically for both model families.
    """
    weights_dir = output_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    source = output_dir / "checkpoint_best_total.pth"
    if not source.exists():
        raise FileNotFoundError(
            f"RF-DETR training did not produce the canonical best checkpoint: {source}"
        )
    if source.stat().st_size == 0:
        raise FileNotFoundError(f"RF-DETR best checkpoint is empty: {source}")

    dest = weights_dir / "best.pt"
    shutil.copy2(source, dest)
    print(f"RF-DETR best weights saved to {dest}")


def _prepare_rfdetr_yolo_layout(training_path: Path, test_path: Path, dataset_name: str) -> Path:
    """Create a lightweight YOLO-format directory for RF-DETR.

    RF-DETR 1.4+ auto-detects YOLO datasets when it finds ``data.yaml`` +
    ``train/images/`` at the dataset root.  Our prepare stage produces a
    slightly different layout (``val/`` instead of ``valid/``, separate
    train/test roots, ``dataset.yaml`` instead of ``data.yaml``), so this
    helper bridges the gap with three symlinks and one tiny YAML file.
    """
    base_dir = Path(".tmp") / "rfdetr_datasets"
    output_dir = base_dir / safe_dataset_dirname(str(dataset_name))
    # Defensive check: ensure output_dir cannot escape base_dir.
    if not output_dir.resolve(strict=False).is_relative_to(base_dir.resolve(strict=False)):
        raise ValueError(f"Unsafe dataset_name for RF-DETR export dir: {dataset_name!r}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # RF-DETR expects train/, valid/, test/ under one root
    (output_dir / "train").symlink_to((training_path / "train").resolve())
    (output_dir / "valid").symlink_to((training_path / "val").resolve())
    (output_dir / "test").symlink_to((test_path / "val").resolve())

    # RF-DETR looks for data.yaml (not dataset.yaml)
    src_yaml = training_path / "dataset.yaml"
    with open(src_yaml, "r", encoding="utf-8") as f:
        ds_cfg = yaml.safe_load(f) or {}

    names_raw = ds_cfg.get("names", [])
    if isinstance(names_raw, dict):
        names_list = [names_raw[k] for k in sorted(names_raw.keys(), key=lambda x: int(x))]
    else:
        names_list = list(names_raw)

    data_yaml = {
        "names": names_list,
        "nc": len(names_list),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
    }
    with open(output_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    return output_dir


def train_backend(
    *,
    training_path: Path,
    test_path: Path,
    dataset_name: str,
    resolved_cfg: dict,
    experiment_name: str | None,
) -> tuple[object, Path, str, int, int]:
    """Train RF-DETR and return an Ultralytics-compatible model adapter."""

    from object_detector_trainer.evaluation.validate import get_dataset_classes
    from object_detector_trainer.wrappers.rfdetr import RFDETRModelAdapter

    rfdetr_variant = resolved_cfg["rfdetr_variant"]
    rfdetr_epochs = int(resolved_cfg["rfdetr_epochs"])
    rfdetr_batch_size = int(resolved_cfg["rfdetr_batch_size"])
    rfdetr_grad_accum = int(resolved_cfg["rfdetr_grad_accum"])
    if not resolved_cfg.get("rfdetr_grad_accum_explicit", False):
        print(
            f"RF-DETR: auto-computed grad_accum_steps={rfdetr_grad_accum} "
            f"(target_effective_batch={resolved_cfg['rfdetr_target_effective_batch']} / "
            f"batch_size={rfdetr_batch_size})"
        )

    rfdetr_lr = resolved_cfg.get("rfdetr_lr")
    rfdetr_resolution = int(resolved_cfg["rfdetr_resolution"])
    rfdetr_checkpoint = resolved_cfg.get("rfdetr_checkpoint")
    rfdetr_grad_ckpt = resolved_cfg.get("rfdetr_grad_ckpt")
    rfdetr_extra = resolved_cfg.get("rfdetr_extra")

    # Prepare a lightweight YOLO-format directory layout for RF-DETR.
    # RF-DETR 1.4+ auto-detects YOLO format (data.yaml + train/images/).
    rfdetr_export_dir = _prepare_rfdetr_yolo_layout(
        training_path=training_path,
        test_path=test_path,
        dataset_name=str(dataset_name),
    )
    rfdetr_dataset_dir = rfdetr_export_dir

    runs_root = Path("runs")
    runs_root.mkdir(parents=True, exist_ok=True)
    display_name = f"{(experiment_name or resolved_cfg['model_key'])}-rfdetr-{rfdetr_variant}"

    rfdetr_model, train_output_dir = train_rfdetr(
        dataset_dir=rfdetr_dataset_dir,
        output_root=runs_root,
        experiment_name=display_name,
        model_variant=rfdetr_variant,
        epochs=rfdetr_epochs,
        batch_size=rfdetr_batch_size,
        grad_accum_steps=rfdetr_grad_accum,
        lr=float(rfdetr_lr) if rfdetr_lr is not None else None,
        resolution=rfdetr_resolution,
        checkpoint_path=str(rfdetr_checkpoint) if rfdetr_checkpoint is not None else None,
        gradient_checkpointing=rfdetr_grad_ckpt,
        extra_train_kwargs=rfdetr_extra if isinstance(rfdetr_extra, dict) else None,
    )

    _save_rfdetr_weights(train_output_dir)

    test_yaml = test_path / "dataset.yaml"
    class_names_map, _ = get_dataset_classes(test_yaml)

    model = RFDETRModelAdapter(
        rfdetr_model,
        model_name=display_name,
        resolution=rfdetr_resolution,
        class_names=class_names_map,
        model_variant=rfdetr_variant,
    )

    shutil.rmtree(rfdetr_export_dir, ignore_errors=True)

    return model, train_output_dir, display_name, int(rfdetr_resolution), int(rfdetr_epochs)


__all__ = [
    "bootstrap_assets",
    "build_reload_metadata",
    "load_model_from_weights",
    "resolve_config",
    "train_backend",
    "_get_rfdetr_model",
    "_normalize_rfdetr_resolution",
]
