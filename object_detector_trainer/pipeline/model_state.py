"""Shared train/evaluate state and model-loading logic.

This module exists to keep one source of truth for:
1) persisted train-result schema (runs/.last_train_result.json),
2) loading a trained model from saved weights/metadata,
3) baseline resolution fallback order.

Both train_stage and evaluate_stage use these functions to avoid duplicated
state/weight-loading behavior and drift.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from object_detector_trainer.backends.training_config import normalize_backend_name
from object_detector_trainer.wrappers.rfdetr import RFDETRModelAdapter


@dataclass
class PersistedTrainResult:
    train_output_dir: Path
    experiment_name: str
    image_size: int
    train_epochs: int
    training_path: Path
    test_path: Path
    baseline_weights_path: str | None
    fallback_checkpoint: str
    finetune_weights_path: str | None
    best_weights_path: Path
    reload_metadata: dict[str, Any]


_PERSISTED_RESULT_VERSION = 1


def _runs_root() -> Path:
    return Path("runs")


def _last_train_result_path() -> Path:
    return _runs_root() / ".last_train_result.json"


def _load_yolo_model(*args: Any, **kwargs: Any) -> Any:
    # Lazy import so non-YOLO workflows don't import Ultralytics at module import time.
    from ultralytics import YOLO as UltralyticsYOLO

    return UltralyticsYOLO(*args, **kwargs)


def persist_train_result(
    *,
    train_output_dir: Path,
    experiment_name: str,
    image_size: int,
    train_epochs: int,
    training_path: Path,
    test_path: Path,
    baseline_weights_path: str | None,
    fallback_checkpoint: str,
    finetune_weights_path: str | None,
    reload_metadata: dict[str, Any],
) -> None:
    payload = {
        "version": _PERSISTED_RESULT_VERSION,
        "train_output_dir": str(train_output_dir),
        "experiment_name": experiment_name,
        "image_size": int(image_size),
        "train_epochs": int(train_epochs),
        "training_path": str(training_path),
        "test_path": str(test_path),
        "baseline_weights_path": baseline_weights_path,
        "fallback_checkpoint": fallback_checkpoint,
        "finetune_weights_path": finetune_weights_path,
        "best_weights_path": str(train_output_dir / "weights" / "best.pt"),
        "reload_metadata": reload_metadata,
    }
    runs_dir = _runs_root()
    runs_dir.mkdir(parents=True, exist_ok=True)
    with _last_train_result_path().open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _normalize_candidate_path(path_candidate: str | Path | None) -> Path | None:
    if not path_candidate:
        return None
    candidate_path = Path(path_candidate).expanduser()
    if not candidate_path.is_absolute():
        candidate_path = Path.cwd() / candidate_path
    return candidate_path


def _weight_candidate_status(path_candidate: str | Path | None) -> tuple[Path | None, str]:
    candidate_path = _normalize_candidate_path(path_candidate)
    if candidate_path is None:
        return None, "not configured"
    if not candidate_path.exists():
        return candidate_path, f"missing file: {candidate_path}"
    if candidate_path.stat().st_size == 0:
        return candidate_path, f"empty file: {candidate_path}"
    return candidate_path, "ready"


def _normalize_persisted_payload(path: Path, payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(
            f"Invalid persisted train result at {path}: expected JSON object, got {type(payload)}."
        )

    version = payload.get("version", 0)
    if version not in {0, _PERSISTED_RESULT_VERSION}:
        raise ValueError(f"Unsupported persisted train result version {version!r} at {path}.")

    required_keys = (
        "train_output_dir",
        "experiment_name",
        "image_size",
        "train_epochs",
        "training_path",
        "test_path",
        "fallback_checkpoint",
    )
    missing = [k for k in required_keys if k not in payload]
    if missing:
        raise ValueError(
            f"Invalid persisted train result at {path}: missing keys {', '.join(missing)}."
        )

    for key in ("train_output_dir", "experiment_name", "training_path", "test_path", "fallback_checkpoint"):
        if not isinstance(payload.get(key), str):
            raise ValueError(
                f"Invalid persisted train result at {path}: key '{key}' must be a string."
            )

    for key in ("image_size", "train_epochs"):
        if not isinstance(payload.get(key), int):
            raise ValueError(
                f"Invalid persisted train result at {path}: key '{key}' must be an integer."
            )

    baseline_weights_path = payload.get("baseline_weights_path")
    if baseline_weights_path is not None and not isinstance(baseline_weights_path, str):
        raise ValueError(
            f"Invalid persisted train result at {path}: key 'baseline_weights_path' must be a string or null."
        )
    finetune_weights_path = payload.get("finetune_weights_path")
    if finetune_weights_path is not None and not isinstance(finetune_weights_path, str):
        raise ValueError(
            f"Invalid persisted train result at {path}: key 'finetune_weights_path' must be a string or null."
        )

    reload_metadata = payload.get("reload_metadata", {})
    if not isinstance(reload_metadata, dict):
        reload_metadata = {}

    best_weights_path = payload.get("best_weights_path")
    if version == 0 and best_weights_path is None:
        best_weights_path = str(Path(payload["train_output_dir"]) / "weights" / "best.pt")
    if not isinstance(best_weights_path, str):
        raise ValueError(
            f"Invalid persisted train result at {path}: key 'best_weights_path' must be a string."
        )

    return {
        "train_output_dir": payload["train_output_dir"],
        "experiment_name": payload["experiment_name"],
        "image_size": payload["image_size"],
        "train_epochs": payload["train_epochs"],
        "training_path": payload["training_path"],
        "test_path": payload["test_path"],
        "baseline_weights_path": baseline_weights_path,
        "fallback_checkpoint": payload["fallback_checkpoint"],
        "finetune_weights_path": finetune_weights_path,
        "best_weights_path": best_weights_path,
        "reload_metadata": reload_metadata,
    }


def load_persisted_train_result() -> PersistedTrainResult:
    path = _last_train_result_path()
    if not path.exists():
        raise FileNotFoundError(
            f"No persisted train result found at {path}. "
            "Run the train stage first or provide train_result explicitly."
        )
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    normalized = _normalize_persisted_payload(path, payload)
    return PersistedTrainResult(
        train_output_dir=Path(normalized["train_output_dir"]),
        experiment_name=str(normalized["experiment_name"]),
        image_size=int(normalized["image_size"]),
        train_epochs=int(normalized["train_epochs"]),
        training_path=Path(normalized["training_path"]),
        test_path=Path(normalized["test_path"]),
        baseline_weights_path=normalized["baseline_weights_path"],
        fallback_checkpoint=str(normalized["fallback_checkpoint"]),
        finetune_weights_path=normalized["finetune_weights_path"],
        best_weights_path=Path(normalized["best_weights_path"]),
        reload_metadata=normalized["reload_metadata"],
    )


def load_model_from_weights(
    path_candidate: str | Path | None,
    metadata_override: dict[str, object] | None = None,
) -> tuple[object | None, str | None]:
    candidate_path, status = _weight_candidate_status(path_candidate)
    if status != "ready" or candidate_path is None:
        return None, None

    meta: dict[str, object] = {}
    for meta_path in (
        candidate_path.parent / "metadata.yaml",
        candidate_path.parent.parent / "metadata.yaml",
    ):
        if not meta_path.exists():
            continue
        with meta_path.open("r", encoding="utf-8") as mf:
            parsed = yaml.safe_load(mf) or {}
        if isinstance(parsed, dict):
            meta.update(parsed)
            break
    if isinstance(metadata_override, dict):
        meta.update(metadata_override)

    display_name = (
        meta.get("experiment_name")
        or meta.get("baseline_display_name")
        or meta.get("run_name")
        or candidate_path.stem
    )

    backend_raw = str(meta.get("model_backend", "yolo")).strip().lower()
    try:
        backend = normalize_backend_name(backend_raw)
    except ValueError:
        backend = backend_raw
    if backend == "rfdetr":
        from object_detector_trainer.backends import rfdetr as core_rfdetr

        model_variant = str(meta.get("model_variant", "base")).strip().lower() or "base"
        resolution = int(meta.get("image_size", 640) or 640)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        rfdetr_model = core_rfdetr._get_rfdetr_model(
            model_variant=model_variant,
            pretrain_weights=str(candidate_path),
            device=device,
            resolution=int(resolution),
        )
        adapter = RFDETRModelAdapter(
            rfdetr_model,
            model_name=str(display_name),
            resolution=int(resolution),
            model_variant=model_variant,
        )
        return adapter, str(display_name)

    if backend == "rtmdet":
        from object_detector_trainer.backends import rtmdet as core_rtmdet

        adapter = core_rtmdet.load_rtmdet_baseline(
            weights_path=candidate_path,
            metadata=meta,
            display_name=str(display_name),
        )
        return adapter, str(display_name)

    model_instance = _load_yolo_model(str(candidate_path))
    return model_instance, str(display_name)


def resolve_baseline_model(
    baseline_weights_path: str | None,
    fallback_checkpoint: str,
    finetune_weights_path: str | None = None,
) -> tuple[object, str]:
    baseline_candidate, baseline_status = _weight_candidate_status(baseline_weights_path)
    baseline_model, baseline_display_name = load_model_from_weights(baseline_candidate)
    if baseline_model is not None:
        return baseline_model, (baseline_display_name or "baseline")

    reason_lines = [f"baseline candidate: {baseline_status}"]

    if finetune_weights_path:
        secondary_path, secondary_status = _weight_candidate_status(finetune_weights_path)
        if baseline_candidate is None or str(baseline_candidate) != str(secondary_path):
            baseline_model, baseline_display_name = load_model_from_weights(secondary_path)
            if baseline_model is not None:
                return baseline_model, (baseline_display_name or Path(finetune_weights_path).stem)
            reason_lines.append(f"finetune candidate: {secondary_status}")

    baseline_checkpoint = str(fallback_checkpoint)
    try:
        baseline_model = _load_yolo_model(baseline_checkpoint)
    except Exception as e:
        raise RuntimeError(
            "Failed to load a baseline model. "
            + " ; ".join(reason_lines)
            + f" ; fallback checkpoint load failed: {baseline_checkpoint}"
        ) from e
    return baseline_model, f"{Path(baseline_checkpoint).stem}-coco"


__all__ = [
    "PersistedTrainResult",
    "load_model_from_weights",
    "load_persisted_train_result",
    "persist_train_result",
    "resolve_baseline_model",
]
