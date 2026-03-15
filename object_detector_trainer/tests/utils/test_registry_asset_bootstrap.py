from __future__ import annotations

from pathlib import Path

from object_detector_trainer.backends import registry


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
    result = registry._download_yolo_checkpoint(target)

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
    result = registry._download_yolo_checkpoint(target)

    assert result == target
    assert target.exists()
    assert target.read_bytes() == b"downloaded-yolo-weights"
    assert captured == {
        "file": "yolo11m.pt",
        "kwargs": {"repo": "ultralytics/assets"},
    }
