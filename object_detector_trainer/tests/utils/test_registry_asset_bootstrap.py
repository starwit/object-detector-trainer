from __future__ import annotations

import os
from pathlib import Path

from object_detector_trainer.backends import assets, registry


def test_download_yolo_checkpoint_copies_into_requested_cache_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    downloaded = tmp_path / "downloads" / "yolov8n.pt"
    downloaded.parent.mkdir(parents=True, exist_ok=True)
    downloaded.write_bytes(b"downloaded-yolo-weights")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "ultralytics.utils.downloads.attempt_download_asset",
        lambda *args, **kwargs: str(downloaded),
    )

    target = tmp_path / "models" / "pretrained" / "yolo" / "yolov8n.pt"
    result = assets.download_yolo_checkpoint(target)

    assert result == target
    assert target.exists()
    assert target.read_bytes() == b"downloaded-yolo-weights"


def test_download_yolo_checkpoint_uses_ultralytics_default_release_resolution(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    downloaded = tmp_path / "downloads" / "yolo11m.pt"
    downloaded.parent.mkdir(parents=True, exist_ok=True)
    downloaded.write_bytes(b"downloaded-yolo-weights")

    captured: dict[str, object] = {}

    def _fake_attempt_download_asset(file: str, **kwargs) -> str:
        captured["file"] = file
        captured["kwargs"] = dict(kwargs)
        return str(downloaded)

    monkeypatch.setattr(
        "ultralytics.utils.downloads.attempt_download_asset",
        _fake_attempt_download_asset,
    )

    target = tmp_path / "models" / "pretrained" / "yolo" / "yolo11m.pt"
    result = assets.download_yolo_checkpoint(target)

    assert result == target
    assert target.exists()
    assert target.read_bytes() == b"downloaded-yolo-weights"
    assert captured == {
        "file": "yolo11m.pt",
        "kwargs": {"repo": "ultralytics/assets"},
    }


def test_bootstrap_rfdetr_checkpoint_downloads_into_requested_cache_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    captured: dict[str, object] = {}

    def _fake_download_pretrain_weights(filename: str) -> None:
        captured["filename"] = filename
        captured["cwd"] = os.getcwd()
        Path(filename).write_bytes(b"downloaded-rfdetr-weights")

    monkeypatch.setattr(
        "rfdetr.assets.model_weights.download_pretrain_weights",
        _fake_download_pretrain_weights,
    )

    result = registry.bootstrap_model_assets(
        "rfdetr-nano",
        {
            "backend": "rfdetr",
            "asset_id": "rf-detr-nano.pth",
            "cache_dir": "models/pretrained/rfdetr",
            "allow_download": True,
        },
    )

    target = tmp_path / "models" / "pretrained" / "rfdetr" / "rf-detr-nano.pth"
    assert result == target
    assert target.exists()
    assert target.read_bytes() == b"downloaded-rfdetr-weights"
    assert captured == {
        "filename": "rf-detr-nano.pth",
        "cwd": str(target.parent),
    }
    assert Path.cwd() == tmp_path


def test_bootstrap_rfdetr_checkpoint_raises_when_download_disabled_and_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    try:
        registry.bootstrap_model_assets(
            "rfdetr-nano",
            {
                "backend": "rfdetr",
                "asset_id": "rf-detr-nano.pth",
                "cache_dir": "models/pretrained/rfdetr",
                "allow_download": False,
            },
        )
    except FileNotFoundError as exc:
        assert "RF-DETR checkpoint is missing" in str(exc)
    else:
        raise AssertionError("Expected missing RF-DETR checkpoint bootstrap to fail.")
