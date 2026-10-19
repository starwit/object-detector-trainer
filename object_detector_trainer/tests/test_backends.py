from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from object_detector_trainer.backends.rfdetr import RfdetrModel, write_rfdetr_dataset
from object_detector_trainer.tests.helpers import write_image, write_label


class FakeRfdetr:
    """Stands in for the RF-DETR library model; records what it receives."""

    def __init__(self, detections: SimpleNamespace, class_names=("waste", "cigarette")):
        self.detections = detections
        self.class_names = list(class_names)
        self.images = []
        if not hasattr(detections, "data"):
            detections.data = {
                "class_name": np.array([self.class_names[i] for i in detections.class_id], dtype=object)
            }

    def predict(self, image, threshold):
        self.images.append(image)
        return self.detections


def test_rfdetr_predictions_are_rgb_input_sorted_and_limited() -> None:
    fake = FakeRfdetr(SimpleNamespace(
        xyxy=np.array([[0, 0, 5, 5], [10, 10, 20, 20], [1, 1, 2, 2]], dtype=np.float32),
        confidence=np.array([0.2, 0.9, 0.5], dtype=np.float32),
        class_id=np.array([0, 1, 0]),
    ))
    model = RfdetrModel(fake)
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[0, 0] = [1, 2, 3]

    detections = model.predict(image, conf=0.1, max_det=2)

    assert fake.images[0][0, 0].tolist() == [3, 2, 1]
    assert detections.scores.tolist() == pytest.approx([0.9, 0.5])
    assert detections.class_ids.tolist() == [1, 0]
    assert detections.xyxy.tolist() == [[10, 10, 20, 20], [1, 1, 2, 2]]


def test_rfdetr_handles_no_detections() -> None:
    fake = FakeRfdetr(SimpleNamespace(xyxy=np.empty((0, 4)), confidence=np.empty(0), class_id=np.empty(0)))
    detections = RfdetrModel(fake).predict(np.zeros((4, 4, 3), dtype=np.uint8), conf=0.1, max_det=5)

    assert detections.xyxy.shape == (0, 4)
    assert detections.scores.shape == (0,)


def test_rfdetr_dataset_links_train_and_validation(tmp_path: Path) -> None:
    for split in ("train", "val", "test"):
        write_image(tmp_path / f"dataset/{split}/images/{split}.jpg")
        write_label(tmp_path / f"dataset/{split}/labels/{split}.txt", "")

    write_rfdetr_dataset(tmp_path / "dataset", tmp_path / "rfdetr", ["waste"])

    assert (tmp_path / "rfdetr/train/images/train.jpg").is_symlink()
    assert (tmp_path / "rfdetr/valid/labels/val.txt").is_symlink()
    assert not (tmp_path / "rfdetr/test").exists()
    assert "names:\n- waste" in (tmp_path / "rfdetr/data.yaml").read_text()


def test_importing_the_trainer_keeps_full_float32_precision() -> None:
    # Importing the rfdetr library enables TF32 for the whole process, which
    # changes YOLO results; it must only be imported when RF-DETR is used.
    code = "import torch, object_detector_trainer; print(torch.get_float32_matmul_precision())"
    output = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True).stdout

    assert output.strip() == "highest"


@pytest.mark.parametrize("device,accelerator,devices", [
    ("cuda:0", "gpu", "0,"), ("cuda:1", "gpu", "1,"),
    ("cuda", "gpu", "0,"), ("cpu", "cpu", 1), ("mps", "mps", 1),
])
def test_rfdetr_training_uses_the_configured_device(
    tmp_path: Path, monkeypatch, device: str, accelerator: str, devices: str | int,
) -> None:
    from object_detector_trainer.backends import rfdetr

    received = {}
    output_dir = tmp_path / "run" / "train"

    def fake_train(**kwargs):
        received["train_accelerator"] = kwargs.get("accelerator")
        received["train_devices"] = kwargs.get("devices")
        assert "device" not in kwargs
        (output_dir / "checkpoint_best_total.pth").write_bytes(b"trained checkpoint")

    def fake_create_model(variant, weights_path, resolution, device, gradient_checkpointing):
        received["constructor_device"] = device
        return SimpleNamespace(train=fake_train)

    def fake_write_dataset(dataset_dir, target_dir, class_names):
        target_dir.mkdir(parents=True)

    monkeypatch.setattr(rfdetr, "create_model", fake_create_model)
    monkeypatch.setattr(rfdetr, "write_rfdetr_dataset", fake_write_dataset)
    job = SimpleNamespace(
        options=rfdetr.Options(
            backend="rfdetr", variant="medium", asset_id="rf-detr-medium.pth",
            effective_batch_size=16, gradient_checkpointing=True,
        ),
        batch_size=1, start_weights=tmp_path / "pretrained.pth",
        image_size=1280, device=device, output_dir=output_dir,
        dataset_dir=tmp_path / "dataset", class_names=["Pkw"], epochs=1, seed=42,
    )

    saved = rfdetr.train(job)

    assert received == {
        "constructor_device": device, "train_accelerator": accelerator, "train_devices": devices,
    }
    assert saved.read_bytes() == b"trained checkpoint"


def test_rfdetr_device_settings_are_accepted_by_the_installed_training_api() -> None:
    # Exercise RF-DETR's real config/strategy parser without requiring a GPU.
    code = """
from types import SimpleNamespace
from rfdetr.config import RFDETRMediumConfig, TrainConfig
from rfdetr.detr import _prepare_run_config
from rfdetr.training.trainer import _requests_multiple_devices
for accelerator, devices in [("gpu", "0,"), ("gpu", "1,"), ("cpu", 1), ("mps", 1)]:
    detector = SimpleNamespace(
        model_config=RFDETRMediumConfig(), model=None,
        get_train_config=lambda **kwargs: TrainConfig(dataset_dir="/tmp", **kwargs),
    )
    config, override_accelerator, override_devices = _prepare_run_config(
        detector, accelerator=accelerator, devices=devices,
    )
    assert (config.accelerator, config.devices) == (accelerator, devices)
    assert override_accelerator is None and override_devices is None
    assert not _requests_multiple_devices(config.devices, config.accelerator)
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("max_det,expected_ids", [(1, [1]), (2, [1, 0])])
def test_rfdetr_removes_native_background_before_detection_limit(max_det, expected_ids) -> None:
    fake = FakeRfdetr(SimpleNamespace(
        xyxy=np.array([[0, 0, 5, 5], [10, 10, 20, 20], [1, 1, 2, 2]], dtype=np.float32),
        confidence=np.array([0.99, 0.6, 0.8], dtype=np.float32),
        class_id=np.array([2, 0, 1]),
        data={"class_name": np.array(["__background__", "Pkw", "Bus"], dtype=object)},
    ), class_names=("Pkw", "Bus"))
    result = RfdetrModel(fake).predict(np.zeros((4, 4, 3), dtype=np.uint8), conf=0.001, max_det=max_det)
    assert result.class_ids.tolist() == expected_ids
    assert result.scores.tolist() == pytest.approx([0.8, 0.6][:max_det])
    assert result.xyxy.tolist() == [[1, 1, 2, 2], [10, 10, 20, 20]][:max_det]


@pytest.mark.parametrize("class_id,name", [(17, ""), (0, "Bus")])
def test_rfdetr_rejects_unexpected_native_identity(class_id, name) -> None:
    fake = FakeRfdetr(SimpleNamespace(
        xyxy=np.array([[0, 0, 5, 5]], dtype=np.float32),
        confidence=np.array([0.8], dtype=np.float32),
        class_id=np.array([class_id]),
        data={"class_name": np.array([name], dtype=object)},
    ), class_names=("Pkw", "Bus"))
    with pytest.raises(ValueError, match="Unexpected RF-DETR prediction identity"):
        RfdetrModel(fake).predict(np.zeros((4, 4, 3), dtype=np.uint8), conf=0.001, max_det=5)


def test_rfdetr_background_only_returns_empty_foreground() -> None:
    fake = FakeRfdetr(SimpleNamespace(
        xyxy=np.array([[0, 0, 5, 5]], dtype=np.float32),
        confidence=np.array([0.8], dtype=np.float32),
        class_id=np.array([2]),
        data={"class_name": np.array(["__background__"], dtype=object)},
    ), class_names=("Pkw", "Bus"))
    result = RfdetrModel(fake).predict(np.zeros((4, 4, 3), dtype=np.uint8), conf=0.001, max_det=5)
    assert result.xyxy.shape == (0, 4)
    assert result.class_ids.shape == result.scores.shape == (0,)


def test_yolo_passes_explicit_device_to_native_prediction(tmp_path, monkeypatch) -> None:
    from object_detector_trainer.backends import yolo

    received = {}
    boxes = SimpleNamespace(
        xyxy=np.empty((0, 4)), conf=np.empty(0), cls=np.empty(0),
    )
    boxes.cpu = lambda: boxes
    boxes.numpy = lambda: boxes

    def predict(image, **kwargs):
        received.update(kwargs)
        return [SimpleNamespace(boxes=boxes)]

    native = SimpleNamespace(names={0: "Pkw"}, predict=predict)
    monkeypatch.setattr(yolo, "YOLO", lambda path: native)
    model = yolo.load(tmp_path / "weights.pt", {"image_size": 128}, device="cpu")
    model.predict(np.zeros((80, 160, 3), dtype=np.uint8), conf=0.25, max_det=7)
    assert received == {"conf": 0.25, "imgsz": 128, "max_det": 7, "device": "cpu", "verbose": False}
