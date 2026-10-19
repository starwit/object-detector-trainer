"""Types and helpers shared by all backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, PositiveInt


class ModelOptions(BaseModel):
    """Settings every model entry under ``models.<key>`` can use.

    Backends extend this class with their own settings. Unknown keys are
    rejected, so a typo in params.yaml fails instead of being ignored.
    """

    model_config = ConfigDict(extra="forbid")

    backend: str
    asset_id: str = Field(min_length=1)
    cache_dir: str | None = None
    allow_download: bool = True
    init_weights: str | None = None
    epochs: PositiveInt | None = None
    image_size: PositiveInt | None = None
    batch_size: PositiveInt = 8
    extra_train_kwargs: dict[str, Any] = Field(default_factory=dict)


@dataclass
class TrainJob:
    """Everything a backend needs to train one model."""

    dataset_dir: Path
    class_names: list[str]
    output_dir: Path
    start_weights: Path
    epochs: int
    image_size: int
    batch_size: int
    seed: int
    device: str
    options: ModelOptions


@dataclass
class Detections:
    """Predictions for one image.

    Boxes are ``x1, y1, x2, y2`` in pixels of the original image. Class IDs
    are the final dataset class IDs.
    """

    xyxy: np.ndarray
    scores: np.ndarray
    class_ids: np.ndarray


def resolve_device(device: str) -> str:
    """Resolve auto once; explicit device selection belongs to the native backend."""
    return default_device() if device == "auto" else device


def default_device() -> str:
    """The first GPU when one is available, otherwise the CPU."""
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"
