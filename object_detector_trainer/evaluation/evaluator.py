"""Score a model on the test split of a prepared dataset.

The model predicts every test image once. From these predictions we compute:

- overall and per-class metrics (see ``coco.py`` for the definitions),
- the fitness of every scene (the source an image came from),
- metrics on the images that contain a merged class (``data.class_mapping``).

Every backend is scored by this same code, so results are comparable.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from object_detector_trainer.dataset import is_image, read_dataset_class_names, read_manifest
from object_detector_trainer.evaluation.coco import DETECTION_CONFIDENCE, MAX_DETECTIONS, coco_metrics
from object_detector_trainer.evaluation.comparison import write_prediction_image


def evaluate_model(
    model, dataset_dir: Path, merged_classes: dict[str, str], predictions_dir: Path | None = None,
) -> dict:
    """Return the metrics of ``model`` on ``dataset_dir/test``.

    ``merged_classes`` maps configured class names to the final class they
    were merged into; each gets its own metrics on the images that contain it.
    When predictions_dir is provided, save annotations beside predictions for every image.
    """
    class_names = read_dataset_class_names(dataset_dir)
    check_class_names(model.class_names, class_names)
    categories = [{"id": class_id, "name": name} for class_id, name in class_names.items()]

    images = []
    annotations = []
    detections = []
    prediction_seconds = 0.0
    test_dir = dataset_dir / "test"
    image_paths = sorted(path for path in (test_dir / "images").iterdir() if is_image(path))
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        height, width = image.shape[:2]
        image_id = len(images) + 1
        images.append({"id": image_id, "file_name": image_path.name, "width": width, "height": height})

        label_path = test_dir / "labels" / f"{image_path.stem}.txt"
        image_annotations = read_ground_truth(
            label_path, image_id, width, height, len(annotations) + 1, class_names
        )
        annotations += image_annotations

        if image_id == 1:
            # The first prediction also sets the model up on its device; keep that out of the timing.
            model.predict(image, conf=DETECTION_CONFIDENCE, max_det=MAX_DETECTIONS)
        started = perf_counter()
        prediction = model.predict(image, conf=DETECTION_CONFIDENCE, max_det=MAX_DETECTIONS)
        prediction_seconds += perf_counter() - started
        detections += coco_detections(prediction, image_id)
        if predictions_dir is not None:
            write_prediction_image(
                image, image_annotations, prediction, class_names, predictions_dir / f"{image_path.stem}.jpg"
            )

    if not images:
        raise ValueError(f"The test split has no images: {test_dir / 'images'}")

    result = coco_metrics(images, annotations, detections, categories)
    result["ms_per_frame"] = prediction_seconds * 1000 / len(images)

    test_rows = {}
    for row in read_manifest(dataset_dir):
        if row["split"] == "test":
            test_rows[row["file"]] = row
    image_rows = {image["id"]: test_rows[image["file_name"]] for image in images}

    result["scenes"] = scene_fitness(images, annotations, detections, categories, image_rows)
    result["merged_classes"] = merged_class_metrics(
        images, annotations, detections, categories, image_rows, merged_classes
    )
    return result


def check_class_names(model_class_names: dict[int, str], dataset_class_names: dict[int, str]) -> None:
    """Stop when model and dataset disagree on classes; the scores would compare different things."""
    if model_class_names != dataset_class_names:
        raise ValueError(
            "The model's classes do not match the dataset's classes.\n"
            f"  model:   {model_class_names}\n"
            f"  dataset: {dataset_class_names}\n"
            "Evaluate only models trained with the same classes (data.custom_classes and data.class_mapping)."
        )


def read_ground_truth(
    label_path: Path,
    image_id: int,
    width: int,
    height: int,
    first_annotation_id: int,
    class_names: dict[int, str],
) -> list[dict]:
    """Convert one prepared YOLO label file to COCO annotations in pixels."""
    annotations = []
    if not label_path.exists():
        return annotations
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        class_id = int(parts[0])
        if class_id not in class_names:
            raise ValueError(f"{label_path} uses class id {class_id}, which is not in dataset.yaml.")
        x_center, y_center, box_width, box_height = (float(value) for value in parts[1:5])
        pixel_width = box_width * width
        pixel_height = box_height * height
        annotations.append({
            "id": first_annotation_id + len(annotations),
            "image_id": image_id,
            "category_id": class_id,
            "bbox": [
                (x_center - box_width / 2) * width,
                (y_center - box_height / 2) * height,
                pixel_width,
                pixel_height,
            ],
            "area": pixel_width * pixel_height,
            "iscrowd": 0,
        })
    return annotations


def coco_detections(prediction, image_id: int) -> list[dict]:
    """Convert predictions to COCO results, highest score first.

    The confidence floor and detection limit are applied here again, so every
    backend is scored on the same rules regardless of its own filtering.
    """
    kept = np.flatnonzero(prediction.scores >= DETECTION_CONFIDENCE)
    best_first = kept[np.argsort(-prediction.scores[kept], kind="stable")][:MAX_DETECTIONS]

    detections = []
    for index in best_first:
        x1, y1, x2, y2 = (float(value) for value in prediction.xyxy[index])
        detections.append({
            "image_id": image_id,
            "category_id": int(prediction.class_ids[index]),
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "score": float(prediction.scores[index]),
        })
    return detections


def scene_fitness(images, annotations, detections, categories, image_rows) -> dict[str, float]:
    image_ids_by_scene: dict[str, set[int]] = {}
    for image_id, row in image_rows.items():
        image_ids_by_scene.setdefault(row["source"], set()).add(image_id)

    fitness = {}
    for scene in sorted(image_ids_by_scene):
        subset = metrics_on_images(images, annotations, detections, categories, image_ids_by_scene[scene])
        fitness[scene] = subset["fitness"]
    return fitness


def merged_class_metrics(images, annotations, detections, categories, image_rows, merged_classes) -> dict:
    """Metrics on the images that contain each merged class, before it was merged."""
    results = {}
    for configured_name, final_name in merged_classes.items():
        image_ids = set()
        object_count = 0
        for image_id, row in image_rows.items():
            source_classes = row["source_classes"].split(";") if row["source_classes"] else []
            if configured_name in source_classes:
                image_ids.add(image_id)
                object_count += source_classes.count(configured_name)
        if not image_ids:
            continue

        subset = metrics_on_images(images, annotations, detections, categories, image_ids)
        results[configured_name] = {
            "merged_into": final_name,
            "images": len(image_ids),
            "objects": object_count,
            "precision": subset["precision"],
            "recall": subset["recall"],
            "f1_score": subset["f1_score"],
            "map50": subset["map50"],
            "map": subset["map"],
        }
    return results


def metrics_on_images(images, annotations, detections, categories, image_ids: set[int]) -> dict:
    return coco_metrics(
        [image for image in images if image["id"] in image_ids],
        [annotation for annotation in annotations if annotation["image_id"] in image_ids],
        [detection for detection in detections if detection["image_id"] in image_ids],
        categories,
    )
