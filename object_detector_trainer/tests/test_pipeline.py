"""When the baseline is compared, and why not."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from object_detector_trainer import pipeline
from object_detector_trainer.config import load_config
from object_detector_trainer.tests.helpers import minimal_params, write_params


def baseline_check(tmp_path: Path, monkeypatch, baseline_classes: dict | None) -> tuple:
    params = minimal_params(evaluation={"baseline_weights_path": "models/current_best/best.pt"})
    config = load_config(write_params(tmp_path / "params.yaml", params))
    dataset_dir = tmp_path / "datasets/sample"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "dataset.yaml").write_text(yaml.safe_dump({"names": {0: "waste"}}))
    if baseline_classes is not None:
        weights = tmp_path / "models/current_best/best.pt"
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b"weights")
        monkeypatch.setattr(
            pipeline, "load_model", lambda path, *, device: SimpleNamespace(class_names=baseline_classes),
        )
    return pipeline.load_baseline(config, tmp_path, dataset_dir)


def test_baseline_with_the_same_classes_is_compared(tmp_path: Path, monkeypatch) -> None:
    model, note = baseline_check(tmp_path, monkeypatch, {0: "waste"})
    assert model is not None and note is None


def test_baseline_with_other_classes_is_not_compared(tmp_path: Path, monkeypatch) -> None:
    model, note = baseline_check(tmp_path, monkeypatch, {0: "waste", 1: "cigarette"})
    assert model is None
    assert "trained with other classes" in note


def test_missing_baseline_is_explained(tmp_path: Path, monkeypatch) -> None:
    model, note = baseline_check(tmp_path, monkeypatch, None)
    assert model is None and "no baseline at models/current_best/best.pt yet" in note

    (tmp_path / "models/current_best").mkdir(parents=True)
    (tmp_path / "models/current_best/best.pt.dvc").write_text("outs: []\n")
    model, note = pipeline.load_baseline(
        load_config(tmp_path / "params.yaml"), tmp_path, tmp_path / "datasets/sample"
    )
    assert model is None and "dvc pull models/current_best/best.pt.dvc" in note


@pytest.mark.parametrize(
    "metadata", [None, "names: [", "model_backend: yolo\n", "model_backend: yolo\nimage_size: 640\n"],
)
def test_unreadable_baseline_does_not_prevent_current_report(tmp_path: Path, monkeypatch, metadata: str | None) -> None:
    import csv

    import numpy as np

    from object_detector_trainer.backends.common import Detections
    from object_detector_trainer.dataset import MANIFEST_COLUMNS
    from object_detector_trainer.tests.helpers import write_image, write_label

    config = load_config(write_params(tmp_path / "params.yaml", minimal_params(
        evaluation={"baseline_weights_path": "models/current_best/best.pt"},
    )))
    dataset = tmp_path / "datasets/sample"
    write_image(dataset / "test/images/test.jpg")
    write_label(dataset / "test/labels/test.txt", "0 0.5 0.5 0.5 0.5\n")
    (dataset / "dataset.yaml").write_text("names: {0: waste}\n")
    with (dataset / "manifest.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerow({"split": "test", "file": "test.jpg", "source": "camera", "source_classes": "waste"})

    weights = tmp_path / "runs/current/train/weights/best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"current checkpoint")
    (weights.parent.parent / "metadata.yaml").write_text("model_backend: yolo\nimage_size: 640\n")
    baseline = tmp_path / "models/current_best/best.pt"
    baseline.parent.mkdir(parents=True)
    baseline.write_bytes(b"invalid checkpoint")
    if metadata is not None:
        (baseline.parent / "metadata.yaml").write_text(metadata)

    class CurrentModel:
        class_names = {0: "waste"}

        def predict(self, image, conf, max_det):
            height, width = image.shape[:2]
            return Detections(
                xyxy=np.array([[width * 0.25, height * 0.25, width * 0.75, height * 0.75]]),
                scores=np.array([0.9]), class_ids=np.array([0]),
            )

    original_load = pipeline.load_model
    monkeypatch.setattr(
        pipeline, "load_model",
        lambda path, *, device: CurrentModel() if path == weights else original_load(path, device=device),
    )
    metrics = pipeline.evaluate(config, tmp_path, "current")

    assert metrics["map50"] > 0.99
    assert "No baseline comparison: could not load" in (tmp_path / "runs/current/eval/report.txt").read_text()
    assert (tmp_path / "runs/current/eval/predictions/test.jpg").is_file()
    assert (tmp_path / "runs/current/metrics.json").is_file()


@pytest.mark.parametrize("backend_name", ["yolo", "rfdetr"])
def test_saved_class_mapping_must_match_native_checkpoint(tmp_path, monkeypatch, backend_name) -> None:
    from object_detector_trainer.backends import get_backend
    from object_detector_trainer.runs import load_model

    weights = tmp_path / "best.pt"
    weights.write_bytes(b"checkpoint")
    metadata = {
        "model_backend": backend_name, "image_size": 128,
        "class_names": {"1": "Bus", "0": "Pkw"},
    }
    received = {}

    def native_load(path, metadata, *, device):
        received["device"] = device
        return SimpleNamespace(class_names={0: "Pkw", 1: "Bus"})

    monkeypatch.setattr(get_backend(backend_name), "load", native_load)
    (tmp_path / "metadata.yaml").write_text(yaml.safe_dump(metadata))
    assert load_model(weights, device="cpu").class_names == {0: "Pkw", 1: "Bus"}
    assert received["device"] == "cpu"

    metadata["class_names"] = {0: "Bus", 1: "Pkw"}
    (tmp_path / "metadata.yaml").write_text(yaml.safe_dump(metadata))
    with pytest.raises(ValueError, match="Checkpoint classes do not match metadata"):
        load_model(weights, device="cpu")


@pytest.mark.parametrize("variant_metadata", [{"model_options": {"variant": "nano"}}, {"model_variant": "NANO"}])
def test_rf_current_and_legacy_metadata_use_native_names(tmp_path, monkeypatch, variant_metadata) -> None:
    from object_detector_trainer.backends import rfdetr
    from object_detector_trainer.runs import load_model

    weights = tmp_path / "best.pt"
    weights.write_bytes(b"checkpoint")
    metadata = {
        "model_backend": "rfdetr", "image_size": 128,
        "class_names": {0: "Pkw", 1: "Bus"}, **variant_metadata,
    }
    (tmp_path / "metadata.yaml").write_text(yaml.safe_dump(metadata))
    received = {}

    def create_model(variant, **kwargs):
        received.update(variant=variant, **kwargs)
        return SimpleNamespace(class_names=["Pkw", "Bus"])

    monkeypatch.setattr(rfdetr, "create_model", create_model)
    model = load_model(weights, device="cpu")
    assert model.class_names == {0: "Pkw", 1: "Bus"}
    assert received["variant"] == "nano" and received["device"] == "cpu"

    if "model_options" in metadata:
        metadata["model_options"] = {}
        metadata["model_variant"] = "nano"
        (tmp_path / "metadata.yaml").write_text(yaml.safe_dump(metadata))
        with pytest.raises(ValueError, match="needs a variant"):
            load_model(weights, device="cpu")


def test_yolo_metadata_without_names_remains_supported(tmp_path, monkeypatch) -> None:
    from object_detector_trainer.backends import yolo
    from object_detector_trainer.runs import load_model

    weights = tmp_path / "best.pt"
    weights.write_bytes(b"checkpoint")
    (tmp_path / "metadata.yaml").write_text("model_backend: yolo\nimage_size: 128\n")
    native = SimpleNamespace(class_names={0: "Pkw"})
    monkeypatch.setattr(yolo, "load", lambda path, metadata, *, device: native)
    assert load_model(weights, device="cpu") is native


def test_inconsistent_baseline_is_skipped_but_direct_load_fails(tmp_path, monkeypatch) -> None:
    from object_detector_trainer.backends import yolo
    from object_detector_trainer.runs import load_model

    model, note = baseline_check(tmp_path, monkeypatch, None)
    weights = tmp_path / "models/current_best/best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"checkpoint")
    (weights.parent / "metadata.yaml").write_text(
        "model_backend: yolo\nimage_size: 128\nclass_names: {0: Bus}\n"
    )
    monkeypatch.setattr(yolo, "load", lambda path, metadata, *, device: SimpleNamespace(class_names={0: "waste"}))
    with pytest.raises(ValueError, match="Checkpoint classes do not match metadata"):
        load_model(weights, device="cpu")
    model, note = pipeline.load_baseline(
        load_config(tmp_path / "params.yaml"), tmp_path, tmp_path / "datasets/sample",
    )
    assert model is None and "could not load" in note and "Checkpoint classes do not match metadata" in note


def test_auto_device_resolution_preserves_explicit_choices(monkeypatch) -> None:
    from object_detector_trainer.backends import common

    monkeypatch.setattr(common, "default_device", lambda: "cuda:0")
    assert common.resolve_device("auto") == "cuda:0"
    assert common.resolve_device("cpu") == "cpu"
    assert common.resolve_device("cuda:1") == "cuda:1"
