from __future__ import annotations

from pathlib import Path

import pytest

from trainer_core.backends.rfdetr import _prepare_rfdetr_yolo_layout


def test_prepare_rfdetr_yolo_layout_sanitizes_dataset_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    # Sentinel in .tmp must never be deleted by dataset_name traversal tricks.
    tmp_root = tmp_path / ".tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    sentinel = tmp_root / "sentinel.txt"
    sentinel.write_text("do-not-delete", encoding="utf-8")

    # Minimal directory structure required by the helper.
    training_path = tmp_path / "datasets" / "ds" / "train"
    test_path = tmp_path / "datasets" / "ds" / "test"
    (training_path / "train").mkdir(parents=True, exist_ok=True)
    (training_path / "val").mkdir(parents=True, exist_ok=True)
    (test_path / "val").mkdir(parents=True, exist_ok=True)

    # The helper only needs the names mapping from dataset.yaml.
    (training_path / "dataset.yaml").write_text("names:\n  0: waste\n", encoding="utf-8")

    export_dir = _prepare_rfdetr_yolo_layout(training_path, test_path, dataset_name="..")

    assert sentinel.exists(), "dataset_name must not allow deleting .tmp/"
    assert export_dir.exists()
    assert export_dir.resolve().is_relative_to((tmp_root / "rfdetr_datasets").resolve())
