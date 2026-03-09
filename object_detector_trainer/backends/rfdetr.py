from __future__ import annotations

import contextlib
import errno
import logging
import shutil
from pathlib import Path

import torch
import yaml

from object_detector_trainer.utils.path_ops import resolve_unique_run_dir, safe_dataset_dirname

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _patched_rfdetr_best_metric_holder(*, init_res: float) -> None:
    """Patch RF-DETR to always emit a canonical best checkpoint.

    Upstream RF-DETR only writes checkpoint_best_regular.pth when mAP strictly
    improves over the initial best (default 0.0). On tiny smoke fixtures, mAP
    can remain at 0.0 which leaves no best checkpoint and crashes later when
    RF-DETR tries to copy it into checkpoint_best_total.pth.

    We patch rfdetr.main.BestMetricHolder for the duration of training so the
    first evaluation is always treated as "best" (init_res < 0.0). This is not a
    fallback to a different model; it makes the canonical output artifact
    deterministic and prevents silent substitution.
    """

    import rfdetr.main as rfdetr_main
    from rfdetr.util.utils import BestMetricHolder as OriginalBestMetricHolder

    original_symbol = getattr(rfdetr_main, "BestMetricHolder", None)
    patched_init_res = float(init_res)

    class PatchedBestMetricHolder(OriginalBestMetricHolder):  # type: ignore[misc]
        def __init__(self, init_res: float = patched_init_res, better: str = "large", use_ema: bool = False) -> None:
            super().__init__(init_res=init_res, better=better, use_ema=use_ema)

    rfdetr_main.BestMetricHolder = PatchedBestMetricHolder  # type: ignore[assignment]
    try:
        yield
    finally:
        if original_symbol is not None:
            rfdetr_main.BestMetricHolder = original_symbol  # type: ignore[assignment]


def _resolve_required_pretrain_weights(path_like: str | Path | None) -> Path:
    if not path_like:
        raise ValueError(
            "RF-DETR models must define models.<key>.pretrain_weights explicitly. "
            "Automatic downloads are not allowed."
        )
    candidate = Path(path_like).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    if not candidate.exists():
        raise FileNotFoundError(f"models.<key>.pretrain_weights does not exist: {candidate}")
    if candidate.stat().st_size == 0:
        raise FileNotFoundError(f"models.<key>.pretrain_weights is empty: {candidate}")
    return candidate


def _rfdetr_resolution_divisor(model_variant: str) -> int:
    """
    RF-DETR requires resolution divisible by (patch_size * num_windows).
    For the official variants:
      - nano/small/medium: 16 * 2 = 32
      - base/large: 14 * 4 = 56
    """
    variant = (model_variant or "").lower()
    if variant in {"nano", "small", "medium"}:
        return 32
    if variant in {"base", "large"}:
        return 56
    return 56


def _normalize_rfdetr_resolution(model_variant: str, resolution: int | None, fallback: int) -> int:
    """Adjust resolution to satisfy RF-DETR constraints."""
    if resolution is None:
        resolution = int(fallback)
    divisor = _rfdetr_resolution_divisor(model_variant)
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
    pretrain_weights: str | None = None,
    device: str | None = None,
    resolution: int | None = None,
    gradient_checkpointing: bool | None = None,
):
    """Return an initialized RF-DETR model based on a variant name."""
    import rfdetr

    variant = (model_variant or "base").lower()
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

    model_cls = getattr(rfdetr, class_name, None)
    if model_cls is None:
        raise RuntimeError(
            f"Your installed rfdetr package does not provide {class_name}. "
            "Either choose another train.rfdetr.model or upgrade rfdetr."
        )

    init_kwargs: dict[str, object] = {}
    if pretrain_weights:
        init_kwargs["pretrain_weights"] = pretrain_weights
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
    pretrain_weights: str | None = None,
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
    pretrain_path = _resolve_required_pretrain_weights(pretrain_weights)

    model = _get_rfdetr_model(
        model_variant,
        pretrain_weights=str(pretrain_path),
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

    with _patched_rfdetr_best_metric_holder(init_res=-1.0):
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


def train_rfdetr_backend(
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
            f"(target_effective_batch={resolved_cfg['rfdetr_target_effective_batch']} / batch_size={rfdetr_batch_size})"
        )

    rfdetr_lr = resolved_cfg.get("rfdetr_lr")
    rfdetr_resolution = int(resolved_cfg["rfdetr_resolution"])
    rfdetr_pretrain = resolved_cfg.get("rfdetr_pretrain")
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
        pretrain_weights=str(rfdetr_pretrain) if rfdetr_pretrain is not None else None,
        gradient_checkpointing=rfdetr_grad_ckpt,
        extra_train_kwargs=rfdetr_extra if isinstance(rfdetr_extra, dict) else None,
    )

    # ── Save weights in the same layout as YOLO (weights/best.pt) ──
    _save_rfdetr_weights(train_output_dir)

    # ── Read class names from the dataset for the adapter ──
    test_yaml = test_path / "dataset.yaml"
    class_names_map, _ = get_dataset_classes(test_yaml)

    # ── Wrap in adapter so evaluate.py treats it like a YOLO model ──
    model = RFDETRModelAdapter(
        rfdetr_model,
        model_name=display_name,
        resolution=rfdetr_resolution,
        class_names=class_names_map,
        model_variant=rfdetr_variant,
    )

    # ── Clean up temporary YOLO layout (not needed after training) ──
    tmp_root = Path(".tmp")
    if rfdetr_export_dir.exists():
        shutil.rmtree(rfdetr_export_dir, ignore_errors=True)
    # Remove empty parent dirs (.tmp/rfdetr_datasets/, .tmp/) if nothing else uses them
    for parent in (rfdetr_export_dir.parent, tmp_root):
        try:
            parent.rmdir()  # only succeeds if empty
        except OSError as exc:
            # Only suppress the expected "not empty" / "already gone" cases.
            if exc.errno in {errno.ENOTEMPTY, errno.ENOENT}:
                logger.debug("Skipping temp dir cleanup for %s: %s", parent, exc)
                continue
            raise

    return model, train_output_dir, display_name, int(rfdetr_resolution), int(rfdetr_epochs)


def train_backend(
    *,
    training_path: Path,
    test_path: Path,
    dataset_name: str,
    resolved_cfg: dict,
    experiment_name: str | None,
) -> tuple[object, Path, str, int, int]:
    return train_rfdetr_backend(
        training_path=training_path,
        test_path=test_path,
        dataset_name=dataset_name,
        resolved_cfg=resolved_cfg,
        experiment_name=experiment_name,
    )


__all__ = [
    "train_backend",
    "train_rfdetr_backend",
    "_get_rfdetr_model",
    "_normalize_rfdetr_resolution",
]
