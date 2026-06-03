"""Shared train/evaluate state and model-loading logic.

This module exists to keep one source of truth for:
1) persisted train-result schema (.dvc_artifacts/last_train_result.json),
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

import yaml

from object_detector_trainer.backends.registry import (
    load_backend_model_from_weights,
    normalize_backend_name,
)
from object_detector_trainer.utils.path_ops import resolve_workspace_path


TRAIN_RUNS_ROOT = Path(".dvc_artifacts") / "train_runs"
TRAIN_RESULT_PATH = Path(".dvc_artifacts") / "last_train_result.json"
PUBLISHED_RUNS_ROOT = Path("runs")
PUBLISHED_TRAIN_RESULT_PATH = PUBLISHED_RUNS_ROOT / ".last_train_result.json"


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
    marker_path: Path = TRAIN_RESULT_PATH,
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
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    with marker_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_persisted_train_result() -> PersistedTrainResult:
    path = TRAIN_RESULT_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"No persisted train result found at {path}. "
            "Run the train stage first or provide train_result explicitly."
        )
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    return PersistedTrainResult(
        train_output_dir=Path(payload["train_output_dir"]),
        experiment_name=str(payload["experiment_name"]),
        image_size=int(payload["image_size"]),
        train_epochs=int(payload["train_epochs"]),
        training_path=Path(payload["training_path"]),
        test_path=Path(payload["test_path"]),
        best_weights_path=Path(payload["best_weights_path"]),
        reload_metadata=dict(payload["reload_metadata"]),
    )


def load_model_from_weights(
    path_candidate: str | Path | None,
    metadata_override: dict[str, object] | None = None,
) -> tuple[object, str]:
    candidate_path = resolve_workspace_path(path_candidate)
    if candidate_path is None:
        raise FileNotFoundError("Model weights are not configured.")
    if not candidate_path.is_file() or candidate_path.stat().st_size == 0:
        raise FileNotFoundError(
            f"Model weights must point to an existing non-empty file: {candidate_path}"
        )

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
    if metadata_override:
        meta.update(metadata_override)
    if not meta:
        raise FileNotFoundError(
            f"Missing metadata.yaml for model weights: {candidate_path}. "
            "Baseline and reload artifacts must include model metadata."
        )

    display_name = str(meta["experiment_name"])

    backend_raw = str(meta.get("model_backend", "")).strip().lower()
    if not backend_raw:
        raise ValueError(
            f"Model metadata for {candidate_path} must define 'model_backend'."
        )
    backend = normalize_backend_name(backend_raw)
    model_instance = load_backend_model_from_weights(
        backend,
        candidate_path,
        meta,
        str(display_name),
        _load_yolo_model,
    )
    return model_instance, str(display_name)


__all__ = [
    "PUBLISHED_RUNS_ROOT",
    "PUBLISHED_TRAIN_RESULT_PATH",
    "PersistedTrainResult",
    "TRAIN_RESULT_PATH",
    "TRAIN_RUNS_ROOT",
    "load_model_from_weights",
    "load_persisted_train_result",
    "persist_train_result",
]
