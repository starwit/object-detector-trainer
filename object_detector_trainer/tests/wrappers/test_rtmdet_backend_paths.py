from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from object_detector_trainer.backends.rtmdet import _prepare_rtmdet_coco_layout, _save_rtmdet_weights


def test_prepare_rtmdet_coco_layout_sanitizes_dataset_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    tmp_root = tmp_path / ".tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    sentinel = tmp_root / "sentinel.txt"
    sentinel.write_text("do-not-delete", encoding="utf-8")

    training_path = tmp_path / "datasets" / "ds" / "train"
    (training_path / "train" / "images").mkdir(parents=True, exist_ok=True)
    (training_path / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (training_path / "val" / "images").mkdir(parents=True, exist_ok=True)
    (training_path / "val" / "labels").mkdir(parents=True, exist_ok=True)

    img = np.zeros((16, 16, 3), dtype=np.uint8)
    cv2.imwrite(str(training_path / "train" / "images" / "img1.jpg"), img)
    (training_path / "train" / "labels" / "img1.txt").write_text(
        "0 0.5 0.5 0.5 0.5\n", encoding="utf-8"
    )
    cv2.imwrite(str(training_path / "val" / "images" / "img2.jpg"), img)
    (training_path / "val" / "labels" / "img2.txt").write_text(
        "0 0.5 0.5 0.5 0.5\n", encoding="utf-8"
    )
    (training_path / "dataset.yaml").write_text("names:\n  0: waste\n", encoding="utf-8")

    export_dir, class_names = _prepare_rtmdet_coco_layout(training_path, dataset_name="..")

    assert sentinel.exists(), "dataset_name must not allow deleting .tmp/"
    assert export_dir.exists()
    assert export_dir.resolve().is_relative_to((tmp_root / "rtmdet_datasets").resolve())
    assert class_names == {0: "waste"}
    train_ann = export_dir / "annotations" / "instances_train.json"
    val_ann = export_dir / "annotations" / "instances_val.json"
    assert train_ann.exists()
    assert val_ann.exists()

    train_payload = json.loads(train_ann.read_text(encoding="utf-8"))
    assert train_payload["images"][0]["file_name"] == "img1.jpg"


def test_save_rtmdet_weights_writes_weights_only_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "rtmdet" / "stub-run"
    run_dir.mkdir(parents=True, exist_ok=True)
    source_ckpt = run_dir / "epoch_1.pth"
    torch.save(
        {
            "state_dict": {"backbone.stem.weight": torch.zeros((1,), dtype=torch.float32)},
            "history": {"dummy": True},
        },
        source_ckpt,
    )

    best_pt = _save_rtmdet_weights(run_dir, source_ckpt)

    payload = torch.load(best_pt, map_location="cpu", weights_only=True)
    assert set(payload.keys()) == {"state_dict"}
