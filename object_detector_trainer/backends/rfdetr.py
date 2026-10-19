"""RF-DETR backend.

The native library enables TF32 globally on import. Keep its native math
policy around RF operations, then restore the caller's policy so loading an
RF baseline does not change another backend's predictions.
"""

from __future__ import annotations

import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import yaml
from pydantic import PositiveFloat, PositiveInt, model_validator

from object_detector_trainer.backends.common import (
    Detections,
    ModelOptions,
    TrainJob,
)

MODEL_CLASS_NAMES = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "base": "RFDETRBase",
    "large": "RFDETRLarge",
}

# Training arguments that come from the trainer config; extra_train_kwargs cannot set them.
TRAINER_OWNED_ARGUMENTS = {
    "dataset_dir", "dataset_file", "epochs", "batch_size", "grad_accum_steps",
    "output_dir", "resolution", "device", "accelerator", "devices", "seed", "lr", "checkpoint_interval",
}


class Options(ModelOptions):
    variant: Literal["nano", "small", "medium", "base", "large"]
    # RF-DETR's name for the input image size; same meaning as image_size.
    resolution: PositiveInt | None = None
    # Target per device; RF-DETR receives the derived accumulation count.
    effective_batch_size: PositiveInt = 16
    lr: PositiveFloat | None = None
    gradient_checkpointing: bool | None = None

    @model_validator(mode="after")
    def use_resolution_as_image_size(self) -> Options:
        if self.resolution is None:
            return self
        if self.image_size is not None and self.image_size != self.resolution:
            raise ValueError("Set resolution or image_size for RF-DETR, not two different values.")
        self.image_size = self.resolution
        return self


def download_weights(asset_id: str, path: Path) -> None:
    # RF-DETR picks the download from the file name, so the path must end in asset_id.
    with _native_precision():
        from rfdetr.assets.model_weights import download_pretrain_weights

        path.parent.mkdir(parents=True, exist_ok=True)
        download_pretrain_weights(str(path))


def train(job: TrainJob) -> Path:
    import torch

    device = torch.device(job.device)
    options = job.options
    grad_accum_steps = max(1, options.effective_batch_size // job.batch_size)

    model = create_model(
        options.variant,
        weights_path=job.start_weights,
        resolution=job.image_size,
        device=job.device,
        gradient_checkpointing=options.gradient_checkpointing,
    )

    rfdetr_dataset_dir = job.output_dir / "rfdetr_dataset"
    write_rfdetr_dataset(job.dataset_dir, rfdetr_dataset_dir, job.class_names)

    train_kwargs = {
        "dataset_dir": str(rfdetr_dataset_dir),
        "dataset_file": "roboflow",
        "epochs": job.epochs,
        "batch_size": job.batch_size,
        "grad_accum_steps": grad_accum_steps,
        "output_dir": str(job.output_dir),
        "resolution": job.image_size,
        # RF-DETR 1.9.4 does not inherit the constructor device. Its device=CUDA
        # shortcut emits a list that its strategy parser rejects; use native fields.
        "accelerator": "gpu" if device.type == "cuda" else device.type,
        # A trailing comma means one GPU ID, rather than a count of GPUs.
        "devices": f"{device.index or 0}," if device.type == "cuda" else 1,
        "seed": job.seed,
        # RF-DETR saves a full training checkpoint (about 0.5 GB) every 10 epochs by
        # default. Only the best checkpoint is used, so never reach the interval.
        "checkpoint_interval": job.epochs + 1,
    }
    if options.lr is not None:
        train_kwargs["lr"] = options.lr
    train_kwargs.update(options.extra_train_kwargs)

    with _native_precision():
        model.train(**train_kwargs)
    shutil.rmtree(rfdetr_dataset_dir)

    # RF-DETR's best checkpoint by total validation score, stored where every
    # backend stores its best weights.
    best_checkpoint = job.output_dir / "checkpoint_best_total.pth"
    if not best_checkpoint.is_file() or best_checkpoint.stat().st_size == 0:
        raise FileNotFoundError(f"RF-DETR did not write its best checkpoint: {best_checkpoint}")
    weights_path = job.output_dir / "weights" / "best.pt"
    weights_path.parent.mkdir(exist_ok=True)
    shutil.copy2(best_checkpoint, weights_path)
    return weights_path


def write_rfdetr_dataset(dataset_dir: Path, target_dir: Path, class_names: list[str]) -> None:
    """Link the prepared dataset into the ``train/`` + ``valid/`` layout RF-DETR reads.

    The split folders are real directories (RF-DETR checks that paths stay
    inside its dataset folder); the image and label files are symlinks.
    """
    for split, rfdetr_split in (("train", "train"), ("val", "valid")):
        for kind in ("images", "labels"):
            target = target_dir / rfdetr_split / kind
            target.mkdir(parents=True)
            for path in sorted((dataset_dir / split / kind).iterdir()):
                (target / path.name).symlink_to(path.resolve())

    content = {
        "names": list(class_names),
        "nc": len(class_names),
        "train": "train/images",
        "val": "valid/images",
    }
    with (target_dir / "data.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(content, handle, sort_keys=False)


@contextmanager
def _native_precision():
    import torch

    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


def create_model(
    variant: str,
    weights_path: Path,
    resolution: int,
    device: str,
    gradient_checkpointing: bool | None = None,
):
    with _native_precision():
        import rfdetr

        model_class = getattr(rfdetr, MODEL_CLASS_NAMES[variant])
        init_kwargs = {
            "pretrain_weights": str(weights_path),
            "resolution": resolution,
            "device": device,
        }
        if gradient_checkpointing is not None:
            init_kwargs["gradient_checkpointing"] = gradient_checkpointing
        return model_class(**init_kwargs)


class RfdetrModel:
    def __init__(self, model) -> None:
        self.model = model
        self.class_names = dict(enumerate(model.class_names))

    def predict(self, image: np.ndarray, conf: float, max_det: int) -> Detections:
        # OpenCV images are BGR; RF-DETR expects RGB.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with _native_precision():
            detections = self.model.predict(rgb_image, threshold=conf)

        xyxy = np.asarray(detections.xyxy, dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(detections.confidence, dtype=np.float32).reshape(-1)
        class_ids = np.asarray(detections.class_id, dtype=np.int64).reshape(-1)

        # RF-DETR 1.9.4 names its native no-object output explicitly.
        # Any other identity must agree with the checkpoint, never disappear silently.
        names = detections.data["class_name"]
        foreground = names != "__background__"
        for class_id, name in zip(class_ids[foreground], names[foreground], strict=True):
            if self.class_names.get(class_id) != name:
                raise ValueError(f"Unexpected RF-DETR prediction identity: class_id={class_id}, class_name={name!r}.")
        valid = np.flatnonzero(foreground)
        best_first = valid[np.argsort(-scores[valid], kind="stable")[:max_det]]
        return Detections(xyxy=xyxy[best_first], scores=scores[best_first], class_ids=class_ids[best_first])


def load(weights_path: Path, metadata: dict, *, device: str) -> RfdetrModel:
    if "model_options" in metadata:
        variant = metadata["model_options"].get("variant")
    else:
        # Models saved by object-detector-trainer 1.x.
        variant = str(metadata.get("model_variant", "")).lower()
    if variant not in MODEL_CLASS_NAMES:
        raise ValueError(
            f"RF-DETR metadata for {weights_path} needs a variant, one of {list(MODEL_CLASS_NAMES)}: "
            "model_options.variant, or model_variant for models saved by trainer 1.x."
        )
    model = create_model(
        variant,
        weights_path=weights_path,
        resolution=int(metadata["image_size"]),
        device=device,
    )
    return RfdetrModel(model)
