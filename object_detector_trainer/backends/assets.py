from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from object_detector_trainer.utils.path_ops import link_or_copy, resolve_workspace_path


def resolve_cache_dir(*, backend: str, model_cfg: Mapping[str, Any]) -> Path:
    cache_dir = str(model_cfg.get("cache_dir") or "").strip()
    return resolve_workspace_path(cache_dir) or Path.cwd() / "models" / "pretrained" / backend


def require_asset_id(*, model_key: str, model_cfg: Mapping[str, Any]) -> str:
    asset_id = str(model_cfg.get("asset_id") or "").strip()
    if not asset_id:
        raise ValueError(f"models.{model_key} must define asset_id.")
    return asset_id


def is_ready_file(path: Path | None) -> bool:
    return path is not None and path.is_file() and path.stat().st_size > 0


def require_bootstrapped_file(path: Path | None, *, label: str) -> Path:
    if path is None or not is_ready_file(path):
        raise FileNotFoundError(f"{label} is missing or empty after bootstrap: {path}")
    return path


def download_yolo_checkpoint(checkpoint_path: Path) -> Path:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    from ultralytics.utils.downloads import attempt_download_asset

    downloaded_path = resolve_workspace_path(
        attempt_download_asset(checkpoint_path.name, repo="ultralytics/assets")
    )
    downloaded_path = require_bootstrapped_file(downloaded_path, label="Downloaded YOLO checkpoint")
    if downloaded_path.resolve() != checkpoint_path.resolve():
        link_or_copy(downloaded_path, checkpoint_path, prefer_hardlink=False)
    require_bootstrapped_file(checkpoint_path, label="YOLO checkpoint")
    if (
        downloaded_path.resolve() != checkpoint_path.resolve()
        and downloaded_path.parent == Path.cwd()
        and downloaded_path.exists()
    ):
        downloaded_path.unlink()
    return checkpoint_path
