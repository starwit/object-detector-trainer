from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from trainer_core.datasets.yolo_yaml import (
    load_yolo_dataset_yaml as _load_yolo_dataset_yaml,
    normalize_class_names as _normalize_class_names,
)
from trainer_core.wrappers.prediction_types import CocoEvalResults


def compute_coco_metrics(
    images: list[dict],
    annotations: list[dict],
    detections: list[dict],
    categories: list[dict],
    *,
    eval_fn: Callable[[list[dict], list[dict], list[dict], list[dict]], CocoEvalResults],
) -> CocoEvalResults:
    """Compute COCO metrics with empty-dataset guard."""
    if len(images) == 0:
        return CocoEvalResults(0.0, 0.0, 0.0, 0.0, {}, macro_f1=0.0)
    return eval_fn(images, annotations, detections, categories)


def load_dataset_yaml(path: str | Path) -> dict:
    return _load_yolo_dataset_yaml(path)


def parse_class_names(ds_cfg: dict) -> dict[int, str]:
    return _normalize_class_names(ds_cfg)


def _iter_dataset_images(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        return []
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )


def _parse_yolo_label_file(
    *,
    label_path: Path,
    image_id: int,
    width: int,
    height: int,
    classes_filter: set[int] | None,
    ann_id_start: int,
) -> tuple[list[dict], int]:
    annotations: list[dict] = []
    ann_id = ann_id_start
    if not label_path.exists():
        return annotations, ann_id

    with label_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls_id = int(float(parts[0]))
                cx = float(parts[1])
                cy = float(parts[2])
                bw = float(parts[3])
                bh = float(parts[4])
            except ValueError:
                continue
            if classes_filter and cls_id not in classes_filter:
                continue
            x = (cx - bw / 2.0) * width
            y = (cy - bh / 2.0) * height
            box_w = bw * width
            box_h = bh * height
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cls_id,
                    "bbox": [x, y, box_w, box_h],
                    "area": box_w * box_h,
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    return annotations, ann_id


def evaluate_yolo_dataset(
    *,
    data: str | Path,
    conf_threshold: float,
    infer_fn: Callable[[np.ndarray, float], tuple[np.ndarray, np.ndarray, np.ndarray]],
    classes_filter: set[int] | None = None,
    valid_class_ids: set[int] | None = None,
) -> tuple[list[dict], list[dict], list[dict], dict[int, str], float]:
    """Evaluate a YOLO dataset with a backend-provided inference callback."""
    ds_cfg = load_dataset_yaml(data)
    if "path" not in ds_cfg:
        raise ValueError(f"Dataset YAML at {data} is missing required 'path'.")
    dataset_path = Path(ds_cfg["path"])
    val_rel = ds_cfg.get("val", "val/images")
    images_dir = dataset_path / val_rel
    labels_dir = images_dir.parent / "labels"

    class_names = parse_class_names(ds_cfg)
    keep_ids = valid_class_ids if valid_class_ids is not None else set(class_names.keys())

    gt_images: list[dict] = []
    gt_annotations: list[dict] = []
    dt_results: list[dict] = []
    ann_id = 1
    image_id = 0
    total_inference_ms = 0.0

    for img_path in _iter_dataset_images(images_dir):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        image_id += 1
        height, width = img.shape[:2]
        gt_images.append({"id": image_id, "file_name": img_path.name, "width": width, "height": height})

        parsed_annotations, ann_id = _parse_yolo_label_file(
            label_path=labels_dir / f"{img_path.stem}.txt",
            image_id=image_id,
            width=width,
            height=height,
            classes_filter=classes_filter,
            ann_id_start=ann_id,
        )
        gt_annotations.extend(parsed_annotations)

        t0 = cv2.getTickCount()
        bboxes, scores, labels = infer_fn(img, conf_threshold)
        t1 = cv2.getTickCount()
        total_inference_ms += ((t1 - t0) / cv2.getTickFrequency()) * 1000.0

        boxes_arr = np.asarray(bboxes, dtype=np.float32)
        if boxes_arr.size == 0:
            boxes_arr = np.empty((0, 4), dtype=np.float32)
        elif boxes_arr.ndim == 1:
            boxes_arr = boxes_arr.reshape(-1, 4)
        scores_arr = np.asarray(scores, dtype=np.float32)
        labels_arr = np.asarray(labels, dtype=np.int64)

        keep = scores_arr >= float(conf_threshold)
        if classes_filter is not None:
            keep = keep & np.array([int(c) in classes_filter for c in labels_arr], dtype=bool)
        else:
            keep = keep & np.array([int(c) in keep_ids for c in labels_arr], dtype=bool)

        for box, score, cls_id in zip(boxes_arr[keep], scores_arr[keep], labels_arr[keep]):
            x1, y1, x2, y2 = box.tolist()
            dt_results.append(
                {
                    "image_id": image_id,
                    "category_id": int(cls_id),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(score),
                }
            )

    avg_inference_ms = total_inference_ms / max(image_id, 1)
    return gt_images, gt_annotations, dt_results, class_names, avg_inference_ms


__all__ = [
    "compute_coco_metrics",
    "evaluate_yolo_dataset",
    "load_dataset_yaml",
    "parse_class_names",
]
