"""One metric definition for every backend and evaluation subset."""

from __future__ import annotations

import contextlib
import io

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# AP keeps low-confidence detections; operating-point metrics use a useful
# fixed threshold instead of selecting a different threshold for each model.
DETECTION_CONFIDENCE = 0.001
METRIC_CONFIDENCE = 0.25
MAX_DETECTIONS = 500


def _mean_valid(values: np.ndarray) -> float:
    valid = values[values > -1]
    return float(np.mean(valid)) if valid.size else 0.0


def coco_metrics(images, annotations, detections, categories) -> dict:
    """COCO AP and macro P/R/F1 at confidence 0.25 and IoU 0.5."""
    if not images:
        raise ValueError("Validation dataset contains no images.")

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = {"images": images, "annotations": annotations, "categories": categories}
        gt.createIndex()
        if detections:
            dt = gt.loadRes(detections)
        else:
            dt = COCO()
            dt.dataset = {"images": images, "annotations": [], "categories": categories}
            dt.createIndex()
        evaluation = COCOeval(gt, dt, "bbox")
        evaluation.params.maxDets = [1, 10, MAX_DETECTIONS]
        evaluation.evaluate()
        evaluation.accumulate()

    # COCO dimensions: IoU, recall, category, area, max detections.
    ap = evaluation.eval["precision"]
    map50 = _mean_valid(ap[0, :, :, 0, -1])
    mean_ap = _mean_valid(ap[:, :, :, 0, -1])
    names = {category["id"]: category["name"] for category in categories}
    per_class = {}
    for class_index, class_id in enumerate(evaluation.params.catIds):
        rows = [
            row for row in evaluation.evalImgs
            if row is not None and row["category_id"] == class_id
            and row["aRng"] == evaluation.params.areaRng[0]
        ]
        gt_count = sum(np.count_nonzero(~row["gtIgnore"].astype(bool)) for row in rows)
        if not gt_count:
            continue
        tp = fp = 0
        for row in rows:
            included = (
                ~row["dtIgnore"][0].astype(bool)
                & (np.asarray(row["dtScores"]) >= METRIC_CONFIDENCE)
            )
            matched = row["dtMatches"][0] > 0
            tp += int(np.count_nonzero(matched & included))
            fp += int(np.count_nonzero(~matched & included))
        per_class[names[class_id]] = {
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": float(tp / gt_count),
            "f1_score": float(2 * tp / (gt_count + tp + fp)),
            "map50": _mean_valid(ap[0, :, class_index, 0, -1]),
            "map": _mean_valid(ap[:, :, class_index, 0, -1]),
        }

    return {
        "precision": float(np.mean([row["precision"] for row in per_class.values()])) if per_class else 0.0,
        "recall": float(np.mean([row["recall"] for row in per_class.values()])) if per_class else 0.0,
        "f1_score": float(np.mean([row["f1_score"] for row in per_class.values()])) if per_class else 0.0,
        "map50": map50,
        "map": mean_ap,
        "fitness": 0.1 * map50 + 0.9 * mean_ap,
        "per_class": per_class,
    }
