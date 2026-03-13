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
