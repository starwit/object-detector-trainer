"""Test annotations and predictions, and baseline comparisons."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from object_detector_trainer.backends.common import Detections
from object_detector_trainer.dataset import is_image
from object_detector_trainer.evaluation.coco import MAX_DETECTIONS, METRIC_CONFIDENCE


def write_comparison_images(baseline_model, trained_model, images_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for image_path in sorted(images_dir.iterdir()):
        if not is_image(image_path):
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        baseline = baseline_model.predict(image, conf=METRIC_CONFIDENCE, max_det=MAX_DETECTIONS)
        trained = trained_model.predict(image, conf=METRIC_CONFIDENCE, max_det=MAX_DETECTIONS)
        left = draw_detections(image, baseline, baseline_model.class_names)
        right = draw_detections(image, trained, trained_model.class_names)
        left = caption(left, "Baseline (confidence >= 0.25)")
        right = caption(right, "Trained model (confidence >= 0.25)")
        write_image(output_dir / f"comparison_{image_path.stem}.jpg", np.hstack((left, right)))


def write_prediction_image(
    image: np.ndarray, annotations: list[dict], prediction: Detections,
    class_names: dict[int, str], output_path: Path,
) -> None:
    """Annotations on the left, model predictions on the right; reuse evaluation predictions."""
    boxes = []
    for annotation in annotations:
        x, y, width, height = annotation["bbox"]
        boxes.append([x, y, x + width, y + height])
    ground_truth = Detections(
        xyxy=np.asarray(boxes).reshape(-1, 4),
        scores=np.ones(len(boxes)),
        class_ids=np.asarray([annotation["category_id"] for annotation in annotations]),
    )
    left = caption(draw_detections(image, ground_truth, class_names, show_scores=False), "Annotations")
    count = int(np.count_nonzero(prediction.scores >= METRIC_CONFIDENCE))
    right = caption(
        draw_detections(image, prediction, class_names),
        f"Predictions (confidence >= {METRIC_CONFIDENCE:g}, {count} boxes)",
    )
    write_image(output_path, np.hstack((left, right)))


def caption(image: np.ndarray, text: str) -> np.ndarray:
    scale = max(1.0, image.shape[1] / 640)
    banner = np.zeros((round(32 * scale), image.shape[1], 3), dtype=image.dtype)
    cv2.putText(
        banner, text, (round(8 * scale), round(22 * scale)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (255, 255, 255), max(1, round(scale)),
    )
    return np.vstack((banner, image))


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not save prediction image: {path}")


def draw_detections(
    image: np.ndarray, detections: Detections, class_names: dict[int, str],
    *, show_scores: bool = True,
) -> np.ndarray:
    drawn = image.copy()
    scale = max(1.0, image.shape[1] / 1280)
    font_scale = 0.5 * scale
    thickness = max(1, round(scale))
    for box, score, class_id in zip(detections.xyxy, detections.scores, detections.class_ids, strict=True):
        if score < METRIC_CONFIDENCE:
            continue
        x1, y1, x2, y2 = (int(value) for value in box)
        color = class_color(int(class_id))
        text = str(class_names.get(int(class_id), int(class_id)))
        if show_scores:
            text += f" {score:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(drawn, (x1, y1), (x2, y2), color, 2 * thickness)
        cv2.rectangle(drawn, (x1, y1 - text_height - 6 * thickness), (x1 + text_width, y1), color, -1)
        cv2.putText(
            drawn, text, (x1, y1 - 4 * thickness), cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, (255, 255, 255), thickness,
        )
    return drawn


def class_color(class_id: int) -> tuple[int, int, int]:
    """A fixed color per class, so a class looks the same in every image."""
    rng = np.random.RandomState(class_id + 1)
    blue, green, red = rng.randint(60, 255, size=3)
    return int(blue), int(green), int(red)
