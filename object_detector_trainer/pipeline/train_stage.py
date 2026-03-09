from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from object_detector_trainer.backends import rtmdet, rfdetr, yolo
from object_detector_trainer.backends.training_config import resolve_training_config
from object_detector_trainer.config.loader import load_config
from object_detector_trainer.pipeline.model_state import persist_train_result
from object_detector_trainer.plugins.replay import build_or_update_replay_set


@dataclass
class TrainResult:
    model: Any
    train_output_dir: Path
    experiment_name: str
    image_size: int
    train_epochs: int
    training_path: Path
    test_path: Path
    reload_metadata: dict[str, Any]


def run_train_stage(args, config=None) -> TrainResult:
    cfg = config or load_config(getattr(args, "config", "params.yaml"), args=args)
    resolved_cfg = resolve_training_config(args, cfg)
    experiment_name = os.getenv("DVC_EXP_NAME")

    dataset_name = Path(getattr(args, "dataset_name", None) or cfg.data.dataset_name)
    dataset_path = Path("datasets") / dataset_name
    training_path = dataset_path / "train"
    test_path = dataset_path / "test"

    backend = resolved_cfg["backend"]
    trainer_by_backend = {
        "rfdetr": rfdetr.train_backend,
        "rtmdet": rtmdet.train_backend,
        "yolo": yolo.train_backend,
    }
    train_backend = trainer_by_backend.get(backend)
    if train_backend is None:
        raise ValueError(f"Unsupported backend: {backend!r}")

    model, train_output_dir, experiment_display_name, image_size, train_epochs = train_backend(
        training_path=training_path,
        test_path=test_path,
        dataset_name=str(dataset_name),
        resolved_cfg=resolved_cfg,
        experiment_name=experiment_name,
    )

    reload_metadata: dict[str, object] = {
        "experiment_name": str(experiment_display_name),
        "model_backend": str(backend),
        "image_size": int(image_size),
    }
    variant_by_backend = {
        "rfdetr": resolved_cfg.get("rfdetr_variant"),
        "rtmdet": resolved_cfg.get("rtmdet_config_name"),
    }
    model_variant = getattr(model, "model_variant", None) or variant_by_backend.get(backend)
    if model_variant:
        reload_metadata["model_variant"] = str(model_variant)

    if backend == "rtmdet":
        for key, attr_name, cfg_key in (
            ("model_config_path", "model_config_path", "rtmdet_config_path"),
            ("rtmdet_config_name", "rtmdet_config_name", "rtmdet_config_name"),
            ("rtmdet_cache_dir", "rtmdet_cache_dir", "rtmdet_cache_dir"),
        ):
            value = getattr(model, attr_name, None) or resolved_cfg.get(cfg_key)
            if value:
                reload_metadata[key] = str(value)

        rtmdet_allow_download = getattr(model, "rtmdet_allow_download", None)
        if rtmdet_allow_download is None:
            rtmdet_allow_download = resolved_cfg.get("rtmdet_allow_download")
        if rtmdet_allow_download is not None:
            reload_metadata["rtmdet_allow_download"] = bool(rtmdet_allow_download)

    class_names = getattr(model, "class_names", None)
    if isinstance(class_names, dict) and class_names:
        reload_metadata["class_names"] = {int(k): str(v) for k, v in class_names.items()}

    result = TrainResult(
        model=model,
        train_output_dir=train_output_dir,
        experiment_name=experiment_display_name,
        image_size=int(image_size),
        train_epochs=int(train_epochs),
        training_path=training_path,
        test_path=test_path,
        reload_metadata=reload_metadata,
    )

    auto_replay_cfg = cfg.prepare.auto_replay
    if auto_replay_cfg and auto_replay_cfg.get("enabled", False):
        build_or_update_replay_set(
            model=model,
            training_path=training_path,
            train_output_dir=train_output_dir,
            config=auto_replay_cfg,
        )

    persist_train_result(
        train_output_dir=result.train_output_dir,
        experiment_name=result.experiment_name,
        image_size=result.image_size,
        train_epochs=result.train_epochs,
        training_path=result.training_path,
        test_path=result.test_path,
        reload_metadata=result.reload_metadata,
    )
    return result
