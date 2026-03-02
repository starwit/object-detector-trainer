from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

import trainer_core.wrappers.rtmdet as adapter_mod


def test_predict_converts_detections_to_yolo_like_result() -> None:
    def _infer(_model, _img):
        return {
            "bboxes": np.array([[20.0, 10.0, 60.0, 30.0]], dtype=np.float32),
            "scores": np.array([0.9], dtype=np.float32),
            "labels": np.array([1], dtype=np.int64),
        }

    adapter = adapter_mod.RTMDetModelAdapter(
        object(), class_names={1: "waste"}, infer_fn=_infer
    )
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    results = adapter.predict(img, conf=0.37)

    assert len(results) == 1
    boxes = results[0].boxes
    assert len(boxes) == 1
    assert boxes.xyxy[0].tolist() == pytest.approx([20.0, 10.0, 60.0, 30.0])
    assert boxes.xywhn[0].tolist() == pytest.approx([0.2, 0.2, 0.2, 0.2])
    assert boxes.conf.tolist() == pytest.approx([0.9])
    assert boxes.cls.tolist() == pytest.approx([1.0])
    plotted = results[0].plot()
    assert plotted.shape == img.shape


def test_predict_filters_unknown_class_ids_when_class_names_provided() -> None:
    def _infer(_model, _img):
        return {
            "bboxes": np.array([[0, 0, 10, 10], [10, 0, 20, 10]], dtype=np.float32),
            "scores": np.array([0.9, 0.8], dtype=np.float32),
            "labels": np.array([0, 99], dtype=np.int64),
        }

    adapter = adapter_mod.RTMDetModelAdapter(
        object(), class_names={0: "waste"}, infer_fn=_infer
    )
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    result = adapter.predict(img, conf=0.01)[0]
    assert len(result.boxes) == 1
    assert result.boxes.cls.tolist() == pytest.approx([0.0])


def test_val_filters_classes_and_returns_expected_metric_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_root = tmp_path / "dataset"
    images_dir = dataset_root / "val" / "images"
    labels_dir = dataset_root / "val" / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    img = np.zeros((100, 200, 3), dtype=np.uint8)
    cv2.imwrite(str(images_dir / "img1.jpg"), img)
    (labels_dir / "img1.txt").write_text(
        "0 0.5 0.5 0.4 0.4\n1 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )

    dataset_yaml = tmp_path / "dataset.yaml"
    with open(dataset_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "path": str(dataset_root),
                "val": "val/images",
                "names": ["waste", "other"],
            },
            f,
            sort_keys=False,
        )

    def _infer(_model, _img):
        return {
            "bboxes": np.array([[20, 10, 60, 30], [30, 20, 40, 35]], dtype=np.float32),
            "scores": np.array([0.8, 0.7], dtype=np.float32),
            "labels": np.array([0, 1], dtype=np.int64),
        }

    captured: dict = {}

    def _fake_compute_metrics(images, annotations, detections, categories):
        captured["images"] = images
        captured["annotations"] = annotations
        captured["detections"] = detections
        captured["categories"] = categories
        return (
            0.8,
            0.6,
            0.5,
            0.4,
            {"waste": {"precision": 0.8, "recall": 0.6, "map50": 0.5, "map": 0.4, "f1_score": 0.6857}},
        )

    monkeypatch.setattr(adapter_mod, "_compute_coco_metrics", _fake_compute_metrics)
    adapter = adapter_mod.RTMDetModelAdapter(
        object(), class_names={0: "waste", 1: "other"}, infer_fn=_infer
    )
    metrics = adapter.val(data=str(dataset_yaml), classes=[0], conf=0.42)

    assert len(captured["images"]) == 1
    assert all(a["category_id"] == 0 for a in captured["annotations"])
    assert all(d["category_id"] == 0 for d in captured["detections"])
    assert captured["categories"] == [{"id": 0, "name": "waste"}, {"id": 1, "name": "other"}]
    assert metrics.results_dict["metrics/precision(B)"] == pytest.approx(0.8)
    assert metrics.results_dict["metrics/recall(B)"] == pytest.approx(0.6)
    assert metrics.results_dict["metrics/mAP50(B)"] == pytest.approx(0.5)
    assert metrics.results_dict["metrics/mAP50-95(B)"] == pytest.approx(0.4)
    assert metrics.results_dict["metrics/f1(B)"] == pytest.approx(0.0)
    assert metrics.fitness == pytest.approx(0.41)
    assert "waste" in metrics.per_class
    assert metrics.speed["inference"] >= 0.0


def test_val_real_coco_metrics_path_with_perfect_detection(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset_real_metrics"
    images_dir = dataset_root / "val" / "images"
    labels_dir = dataset_root / "val" / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    img = np.zeros((100, 200, 3), dtype=np.uint8)
    cv2.imwrite(str(images_dir / "img1.jpg"), img)
    (labels_dir / "img1.txt").write_text("0 0.2 0.2 0.2 0.2\n", encoding="utf-8")

    dataset_yaml = tmp_path / "dataset_real_metrics.yaml"
    with open(dataset_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "path": str(dataset_root),
                "val": "val/images",
                "names": ["waste"],
            },
            f,
            sort_keys=False,
        )

    def _infer(_model, _img):
        return {
            "bboxes": np.array([[20.0, 10.0, 60.0, 30.0]], dtype=np.float32),
            "scores": np.array([0.99], dtype=np.float32),
            "labels": np.array([0], dtype=np.int64),
        }

    adapter = adapter_mod.RTMDetModelAdapter(object(), class_names={0: "waste"}, infer_fn=_infer)
    metrics = adapter.val(data=str(dataset_yaml), conf=0.25)

    assert metrics.results_dict["metrics/mAP50(B)"] > 0.99
    assert metrics.results_dict["metrics/mAP50-95(B)"] > 0.99
    assert metrics.results_dict["metrics/recall(B)"] > 0.99
    assert metrics.results_dict["metrics/precision(B)"] > 0.99
    assert metrics.results_dict["metrics/f1(B)"] > 0.99


def test_val_requires_dataset_yaml() -> None:
    adapter = adapter_mod.RTMDetModelAdapter(object(), infer_fn=lambda *_a, **_k: {})
    with pytest.raises(ValueError, match="data .* required"):
        adapter.val()
