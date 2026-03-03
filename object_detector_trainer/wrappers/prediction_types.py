from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class SingleBox:
    xyxy: np.ndarray
    xywhn: np.ndarray
    conf: float
    cls: float


@dataclass
class Boxes:
    xyxy: np.ndarray
    xywhn: np.ndarray
    conf: np.ndarray
    cls: np.ndarray

    def __len__(self) -> int:
        return len(self.cls)

    def __iter__(self):
        for i in range(len(self)):
            yield SingleBox(
                xyxy=self.xyxy[i : i + 1],
                xywhn=self.xywhn[i : i + 1],
                conf=float(self.conf[i]),
                cls=float(self.cls[i]),
            )


def _class_color(cls_id: int) -> tuple[int, int, int]:
    rng = np.random.RandomState(cls_id + 1)
    return tuple(int(c) for c in rng.randint(60, 255, size=3))


@dataclass
class PredictionResult:
    boxes: Boxes
    orig_img: np.ndarray
    names: dict[int, str]

    def plot(
        self,
        line_width: int = 2,
        font_scale: float = 0.5,
        font_thickness: int = 1,
    ) -> np.ndarray:
        img = self.orig_img.copy()
        for i in range(len(self.boxes)):
            x1, y1, x2, y2 = self.boxes.xyxy[i].astype(int)
            cls_id = int(self.boxes.cls[i])
            conf = float(self.boxes.conf[i])
            label = self.names.get(cls_id, str(cls_id))
            text = f"{label} {conf:.2f}"
            color = _class_color(cls_id)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, line_width)
            (tw, th), _ = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                font_thickness,
            )
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
            cv2.putText(
                img,
                text,
                (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                font_thickness,
            )
        return img


@dataclass
class ValMetrics:
    results_dict: dict[str, float]
    fitness: float
    speed: dict[str, float]
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class CocoEvalResults:
    precision: float
    recall: float
    map50: float
    map50_95: float
    per_class: dict[str, dict[str, float]]
    macro_f1: float = 0.0

    def __iter__(self):
        yield self.precision
        yield self.recall
        yield self.map50
        yield self.map50_95
        yield self.per_class


__all__ = [
    "Boxes",
    "CocoEvalResults",
    "PredictionResult",
    "SingleBox",
    "ValMetrics",
]
