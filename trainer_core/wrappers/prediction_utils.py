from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from trainer_core.wrappers.prediction_types import Boxes, PredictionResult


def load_image(source: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(source, (str, Path)):
        img = cv2.imread(str(source))
        if img is None:
            raise FileNotFoundError(f"Could not read image: {source}")
        return img
    return source


def build_prediction_result(
    *,
    xyxy: Any,
    scores: Any,
    labels: Any,
    img: np.ndarray,
    class_names: dict[int, str],
) -> PredictionResult:
    xyxy_arr = np.asarray(xyxy, dtype=np.float32)
    if xyxy_arr.size == 0:
        xyxy_arr = np.empty((0, 4), dtype=np.float32)
    elif xyxy_arr.ndim == 1:
        xyxy_arr = xyxy_arr.reshape(-1, 4)

    scores_arr = np.asarray(scores, dtype=np.float32)
    labels_arr = np.asarray(labels, dtype=np.float32)

    if labels_arr.size == 0:
        empty = np.empty((0, 4), dtype=np.float32)
        return PredictionResult(
            boxes=Boxes(
                xyxy=empty,
                xywhn=empty,
                conf=np.empty(0, dtype=np.float32),
                cls=np.empty(0, dtype=np.float32),
            ),
            orig_img=img,
            names=class_names,
        )

    height, width = img.shape[:2]
    cx = ((xyxy_arr[:, 0] + xyxy_arr[:, 2]) / 2.0) / float(width)
    cy = ((xyxy_arr[:, 1] + xyxy_arr[:, 3]) / 2.0) / float(height)
    bw = (xyxy_arr[:, 2] - xyxy_arr[:, 0]) / float(width)
    bh = (xyxy_arr[:, 3] - xyxy_arr[:, 1]) / float(height)
    xywhn = np.stack([cx, cy, bw, bh], axis=1).astype(np.float32)

    return PredictionResult(
        boxes=Boxes(
            xyxy=xyxy_arr,
            xywhn=xywhn,
            conf=scores_arr,
            cls=labels_arr,
        ),
        orig_img=img,
        names=class_names,
    )


__all__ = ["build_prediction_result", "load_image"]
