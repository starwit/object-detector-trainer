"""Run folders, saved-model metadata and loading saved models.

A run folder looks like this::

    runs/<run-name>/train/weights/best.pt   best checkpoint
    runs/<run-name>/train/metadata.yaml     what is needed to load and describe the model
    runs/<run-name>/train/config.yaml       the full config the run was trained with
    runs/<run-name>/train/...               the backend's own logs and plots
    runs/<run-name>/eval/                   evaluation reports
    runs/<run-name>/metrics.json            final test metrics of the trained model

``metadata.yaml`` is the contract for saved models, including project
baselines. It needs ``model_backend`` and ``image_size``. It also holds
``class_names`` and ``model_options`` (the model's entry from params.yaml),
which a backend's ``load`` can use to recreate its model.
"""

from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import yaml

from object_detector_trainer.backends import get_backend
from object_detector_trainer.backends.common import resolve_device


def run_dir(workspace: Path, run_name: str) -> Path:
    return workspace / "runs" / run_name


def default_run_name(model_key: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{model_key}-{timestamp}"


def best_weights_path(workspace: Path, run_name: str) -> Path:
    return run_dir(workspace, run_name) / "train" / "weights" / "best.pt"


def write_metadata(train_dir: Path, metadata: dict) -> None:
    metadata = dict(metadata)
    metadata["trainer_version"] = version("object-detector-trainer")
    metadata["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with (train_dir / "metadata.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(metadata, handle, sort_keys=False)


def read_metadata(weights_path: Path) -> dict:
    """Read the metadata.yaml next to the weights, or one folder up (``weights/best.pt``)."""
    for metadata_path in (weights_path.parent / "metadata.yaml", weights_path.parent.parent / "metadata.yaml"):
        if metadata_path.is_file():
            with metadata_path.open(encoding="utf-8") as handle:
                metadata = yaml.safe_load(handle) or {}
            for key in ("model_backend", "image_size"):
                if key not in metadata:
                    raise ValueError(f"{metadata_path} must define {key}.")
            return metadata
    raise FileNotFoundError(
        f"No metadata.yaml found for {weights_path}. Saved models need their metadata.yaml "
        "next to the weights or one folder above."
    )


def load_model(weights_path: Path, *, device: str = "auto"):
    """Load a checkpoint on the requested device and check its saved class mapping."""
    if not weights_path.is_file() or weights_path.stat().st_size == 0:
        raise FileNotFoundError(f"Model weights are missing or empty: {weights_path}")
    metadata = read_metadata(weights_path)
    backend = get_backend(metadata["model_backend"])
    saved_names = None
    if "class_names" in metadata:
        if not isinstance(metadata["class_names"], dict) or not metadata["class_names"]:
            raise ValueError(f"Model metadata for {weights_path} needs a non-empty class_names mapping.")
        saved_names = {int(class_id): str(name) for class_id, name in metadata["class_names"].items()}
    elif metadata["model_backend"] == "rfdetr":
        raise ValueError(f"RF-DETR metadata for {weights_path} needs class_names.")

    model = backend.load(weights_path, metadata, device=resolve_device(device))
    if saved_names is not None and saved_names != model.class_names:
        raise ValueError(
            f"Checkpoint classes do not match metadata for {weights_path}.\n"
            f"  checkpoint: {model.class_names}\n"
            f"  metadata:   {saved_names}"
        )
    return model
