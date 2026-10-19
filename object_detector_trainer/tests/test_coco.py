from __future__ import annotations

import pytest

from object_detector_trainer.evaluation.coco import coco_metrics

IMAGES = [{"id": 1, "width": 100, "height": 100}]
CATEGORIES = [{"id": 0, "name": "waste"}, {"id": 1, "name": "absent"}]
ANNOTATIONS = [
    {"id": 1, "image_id": 1, "category_id": 0, "bbox": [0, 0, 10, 10], "area": 100, "iscrowd": 0},
    {"id": 2, "image_id": 1, "category_id": 0, "bbox": [50, 50, 10, 10], "area": 100, "iscrowd": 0},
]


@pytest.mark.parametrize("confidence", [0.9, 0.1])
def test_ap_and_operating_point_precision_are_distinct(confidence):
    result = coco_metrics(IMAGES, ANNOTATIONS, [
        {"image_id": 1, "category_id": 0, "bbox": [0, 0, 10, 10], "score": confidence},
        {"image_id": 1, "category_id": 0, "bbox": [80, 80, 10, 10], "score": confidence / 2},
    ], CATEGORIES)
    # One of two GT objects is found. A later false positive affects ordinary
    # precision, but does not reduce interpolated AP for the reached recall.
    assert result["map50"] == pytest.approx(51 / 101)
    assert result["map"] == pytest.approx(51 / 101)
    expected = 0.5 if confidence >= 0.25 else 0.0
    for key in ("precision", "recall", "f1_score"):
        assert result[key] == pytest.approx(expected)
    assert set(result["per_class"]) == {"waste"}


def test_iou_thresholds_apply_consistently():
    result = coco_metrics(IMAGES, ANNOTATIONS[:1], [
        {"image_id": 1, "category_id": 0, "bbox": [2, 0, 10, 10], "score": 0.9},
    ], CATEGORIES)
    assert result["recall"] == result["per_class"]["waste"]["recall"] == 1
    assert result["map50"] == pytest.approx(1)
    assert result["map"] == pytest.approx(0.4)


def test_no_detections_keeps_present_classes_with_zero_metrics():
    result = coco_metrics(IMAGES, ANNOTATIONS, [], CATEGORIES)
    assert result["map"] == result["recall"] == result["precision"] == result["f1_score"] == 0
    assert set(result["per_class"]) == {"waste"}
    assert all(value == 0 for value in result["per_class"]["waste"].values())
