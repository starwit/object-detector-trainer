"""Ultralytics YOLO backend."""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Ultralytics draws training plots; this avoids GUI backends on headless machines.
matplotlib.use("Agg")

import numpy as np  # noqa: E402
from pydantic import PositiveInt  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.utils.downloads import attempt_download_asset  # noqa: E402

from object_detector_trainer.backends.common import (  # noqa: E402
    Detections,
    ModelOptions,
    TrainJob,
)

# Training arguments that come from the trainer config; extra_train_kwargs cannot set them.
TRAINER_OWNED_ARGUMENTS = {"data", "epochs", "imgsz", "batch", "nbs", "device", "seed", "project", "name", "exist_ok"}


class Options(ModelOptions):
    # Ultralytics' nominal batch target; accumulation ramps up during warmup.
    effective_batch_size: PositiveInt = 64
    # Freeze the backbone layers, typically when fine-tuning from init_weights.
    freeze_backbone: bool = False


def download_weights(asset_id: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    attempt_download_asset(path, repo="ultralytics/assets")


def train(job: TrainJob) -> Path:
    model = YOLO(str(job.start_weights))
    train_kwargs = {
        "data": str(job.dataset_dir / "dataset.yaml"),
        "epochs": job.epochs,
        "imgsz": job.image_size,
        "batch": job.batch_size,
        "nbs": job.options.effective_batch_size,
        "device": job.device,
        "seed": job.seed,
        "project": str(job.output_dir.parent),
        "name": job.output_dir.name,
        "exist_ok": True,
        # Defaults that extra_train_kwargs may change:
        "workers": 4,
        "amp": True,
        # Ultralytics 8.4.137 MuSGD reshapes gradients with view(), which needs this layout.
        "channels_last": False,
    }
    if job.options.freeze_backbone:
        # Ultralytics freezes the first N layers; the backbone comes first.
        train_kwargs["freeze"] = len(model.model.yaml["backbone"])
    train_kwargs.update(job.options.extra_train_kwargs)

    results = model.train(**train_kwargs)
    if Path(results.save_dir).resolve() != job.output_dir.resolve():
        raise RuntimeError(f"Ultralytics wrote to {results.save_dir} instead of {job.output_dir}.")
    return job.output_dir / "weights" / "best.pt"


class YoloModel:
    def __init__(self, weights_path: Path, image_size: int, device: str) -> None:
        self.model = YOLO(str(weights_path))
        self.image_size = image_size
        self.device = device
        self.class_names = {int(class_id): str(name) for class_id, name in self.model.names.items()}

    def predict(self, image: np.ndarray, conf: float, max_det: int) -> Detections:
        result = self.model.predict(
            image, conf=conf, imgsz=self.image_size, max_det=max_det, device=self.device, verbose=False
        )[0]
        boxes = result.boxes.cpu().numpy()
        return Detections(
            xyxy=boxes.xyxy.astype(np.float32),
            scores=boxes.conf.astype(np.float32),
            class_ids=boxes.cls.astype(np.int64),
        )


def load(weights_path: Path, metadata: dict, *, device: str) -> YoloModel:
    return YoloModel(weights_path, int(metadata["image_size"]), device)
