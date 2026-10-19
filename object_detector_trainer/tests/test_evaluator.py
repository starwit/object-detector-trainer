from __future__ import annotations

import csv
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from object_detector_trainer.backends.common import Detections
from object_detector_trainer.dataset import MANIFEST_COLUMNS, write_dataset_yaml
from object_detector_trainer.evaluation.comparison import draw_detections, write_comparison_images
from object_detector_trainer.evaluation.evaluator import evaluate_model
from object_detector_trainer.tests.helpers import write_image, write_label


class FixedModel:
    """Predicts the same single 'waste' box on every image."""

    def __init__(self, class_names: dict[int, str]):
        self.class_names = class_names
        self.calls = 0

    def predict(self, image, conf, max_det):
        self.calls += 1
        return Detections(
            xyxy=np.array([[20, 10, 60, 30]], dtype=np.float32),
            scores=np.array([0.9], dtype=np.float32),
            class_ids=np.array([0]),
        )


class SlowFirstCallModel(FixedModel):
    """Like a real model, the first prediction is slow because it sets the model up."""

    def predict(self, image, conf, max_det):
        if self.calls == 0:
            time.sleep(0.5)
        return super().predict(image, conf, max_det)


def write_test_dataset(dataset_dir: Path) -> None:
    """Two test images: one with a waste box and a cigarette, one background."""
    images = dataset_dir / "test/images"
    labels = dataset_dir / "test/labels"
    write_image(images / "one__scene_positive.jpg")
    write_image(images / "two__scene_background.jpg")
    # Images are 64x64; the predicted box (20,10)-(60,30) is not the label box.
    write_label(labels / "one__scene_positive.txt", "0 0.625 0.3125 0.625 0.3125\n1 0.8 0.8 0.1 0.1\n")
    write_label(labels / "two__scene_background.txt", "")
    write_dataset_yaml(dataset_dir / "dataset.yaml", ["waste", "cigarette", "absent"])
    with (dataset_dir / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerow({"split": "test", "file": "one__scene_positive.jpg", "source": "positive",
                         "source_image": "a.jpg", "source_classes": "bottle;cigarette"})
        writer.writerow({"split": "test", "file": "two__scene_background.jpg", "source": "background",
                         "source_image": "b.jpg", "source_classes": ""})


def test_evaluation_scores_classes_scenes_and_merged_classes_from_one_prediction_pass(tmp_path: Path) -> None:
    write_test_dataset(tmp_path)
    model = FixedModel({0: "waste", 1: "cigarette", 2: "absent"})

    output = tmp_path / "predictions"
    result = evaluate_model(model, tmp_path, merged_classes={"bottle": "waste"}, predictions_dir=output)

    assert model.calls == 3  # One untimed warm-up, then one prediction per image.
    assert len(list(output.glob("*.jpg"))) == 2
    image = cv2.imread(str(output / "one__scene_positive.jpg"))
    assert image.shape == (96, 128, 3)
    # waste: 1 of 1 found, 1 false positive on the background image. cigarette: missed.
    assert set(result["per_class"]) == {"waste", "cigarette"}
    assert result["recall"] == pytest.approx(0.5)
    assert result["precision"] == pytest.approx(0.25)
    assert result["f1_score"] == pytest.approx(1 / 3)
    assert result["map"] == pytest.approx(0.5)
    assert result["scenes"] == {"background": 0.0, "positive": pytest.approx(0.5)}
    assert result["merged_classes"]["bottle"]["images"] == 1
    assert result["merged_classes"]["bottle"]["objects"] == 1
    assert result["merged_classes"]["bottle"]["merged_into"] == "waste"


def test_model_with_other_classes_is_rejected(tmp_path: Path) -> None:
    write_test_dataset(tmp_path)
    model = FixedModel({0: "waste", 1: "cigarette"})

    with pytest.raises(ValueError, match="classes do not match"):
        evaluate_model(model, tmp_path, merged_classes={})


def test_prediction_images_filter_low_confidence_without_changing_the_image() -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    prediction = Detections(np.array([[20, 10, 60, 30]]), np.array([0.1]), np.array([0]))

    assert np.array_equal(draw_detections(image, prediction, {0: "waste"}), image)
    prediction.scores[0] = 0.9
    assert not np.array_equal(draw_detections(image, prediction, {0: "waste"}), image)
    assert not image.any()


def test_baseline_comparison_images_still_show_both_models(tmp_path: Path) -> None:
    write_test_dataset(tmp_path)
    baseline = FixedModel({0: "waste"})
    trained = FixedModel({0: "waste"})
    output = tmp_path / "comparisons"

    write_comparison_images(baseline, trained, tmp_path / "test/images", output)

    assert baseline.calls == trained.calls == 2
    images = list(output.glob("*.jpg"))
    assert len(images) == 2
    assert cv2.imread(str(images[0])).shape == (96, 128, 3)


def test_setup_time_of_the_first_prediction_is_not_counted(tmp_path: Path) -> None:
    write_test_dataset(tmp_path)
    model = SlowFirstCallModel({0: "waste", 1: "cigarette", 2: "absent"})

    result = evaluate_model(model, tmp_path, merged_classes={})

    assert result["ms_per_frame"] < 100
