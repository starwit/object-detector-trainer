"""Shared train/evaluate state and model-loading logic.

This module exists to keep one source of truth for:
1) persisted train-result schema (runs/.last_train_result.json),
2) loading a trained model from saved weights/metadata,
3) strict baseline/model artifact loading.

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
    best_weights_path: Path
    reload_metadata: dict[str, Any]


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
    reload_metadata: dict[str, Any],
) -> None:
    payload = {
        "train_output_dir": str(train_output_dir),
        "experiment_name": experiment_name,
        "image_size": int(image_size),
        "train_epochs": int(train_epochs),
        "training_path": str(training_path),
        "test_path": str(test_path),
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


def _require_ready_weight(path_candidate: str | Path | None, *, label: str) -> Path:
    candidate_path, status = _weight_candidate_status(path_candidate)
    if status == "ready" and candidate_path is not None:
        return candidate_path
    raise FileNotFoundError(f"{label} must point to an existing non-empty file ({status}).")


def _normalize_persisted_payload(path: Path, payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(
            f"Invalid persisted train result at {path}: expected JSON object, got {type(payload)}."
        )

    required_keys = (
        "train_output_dir",
        "experiment_name",
        "image_size",
        "train_epochs",
        "training_path",
        "test_path",
        "best_weights_path",
    )
    missing = [k for k in required_keys if k not in payload]
    if missing:
        raise ValueError(
            f"Invalid persisted train result at {path}: missing keys {', '.join(missing)}."
        )

    for key in ("train_output_dir", "experiment_name", "training_path", "test_path", "best_weights_path"):
        if not isinstance(payload.get(key), str):
            raise ValueError(
                f"Invalid persisted train result at {path}: key '{key}' must be a string."
            )

    for key in ("image_size", "train_epochs"):
        if not isinstance(payload.get(key), int):
            raise ValueError(
                f"Invalid persisted train result at {path}: key '{key}' must be an integer."
            )

    reload_metadata = payload.get("reload_metadata", {})
    if not isinstance(reload_metadata, dict):
        reload_metadata = {}

    return {
        "train_output_dir": payload["train_output_dir"],
        "experiment_name": payload["experiment_name"],
        "image_size": payload["image_size"],
        "train_epochs": payload["train_epochs"],
        "training_path": payload["training_path"],
        "test_path": payload["test_path"],
        "best_weights_path": payload["best_weights_path"],
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
        best_weights_path=Path(normalized["best_weights_path"]),
        reload_metadata=normalized["reload_metadata"],
    )


def load_model_from_weights(
    path_candidate: str | Path | None,
    metadata_override: dict[str, object] | None = None,
) -> tuple[object, str]:
    candidate_path = _require_ready_weight(path_candidate, label="Model weights")

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
    if not meta:
        raise FileNotFoundError(
            f"Missing metadata.yaml for model weights: {candidate_path}. "
            "Baseline and reload artifacts must include model metadata."
        )

    display_name = (
        meta.get("experiment_name")
        or meta.get("baseline_display_name")
        or meta.get("run_name")
        or candidate_path.stem
    )

    backend_raw = str(meta.get("model_backend", "")).strip().lower()
    if not backend_raw:
        raise ValueError(
            f"Model metadata for {candidate_path} must define 'model_backend'."
        )
    backend = normalize_backend_name(backend_raw)
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
    setattr(model_instance, "model_backend", backend)
    setattr(model_instance, "model_name", str(display_name))
    if meta.get("model_variant"):
        setattr(model_instance, "model_variant", str(meta["model_variant"]))
    if meta.get("image_size") is not None:
        setattr(model_instance, "resolution", int(meta["image_size"]))
    class_names = meta.get("class_names")
    if isinstance(class_names, dict) and class_names:
        setattr(model_instance, "class_names", {int(k): str(v) for k, v in class_names.items()})
    return model_instance, str(display_name)


def resolve_baseline_model(
    baseline_weights_path: str | None,
) -> tuple[object, str]:
    baseline_candidate = _require_ready_weight(
        baseline_weights_path,
        label="evaluation.baseline_weights_path",
    )
    baseline_model, baseline_display_name = load_model_from_weights(baseline_candidate)
    return baseline_model, (baseline_display_name or "baseline")


__all__ = [
    "PersistedTrainResult",
    "load_model_from_weights",
    "load_persisted_train_result",
    "persist_train_result",
    "resolve_baseline_model",
]
