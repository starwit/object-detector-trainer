from __future__ import annotations

"""Shared helpers for assembling the lightweight datasets/params used in tests."""

import copy
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import yaml

from object_detector_trainer.backends.registry import (
    _require_asset_id,
    normalize_backend_name,
    require_bootstrapped_file,
)
from object_detector_trainer.backends.rtmdet import _resolve_rtmdet_assets
from object_detector_trainer.utils.path_ops import link_or_copy


HEAVY_TEST_MODEL_KEYS = ("yolov8n", "rfdetr-nano", "rtmdet-tiny")


BASE_PARAMS: Dict[str, Any] = {
    "data": {
        "dataset_name": "waste-detection",
        "experiment_name": "waste-detection",
        "custom_classes": ["waste"],
        "use_coco_classes": False,
    },
    "prepare": {
        "val_split": 0.5,
        "test_split": 0.0,
        "augment_multiplier": 1,
        "folder_subsets": {},
    },
    "train": {
        "model": "yolov8n",
        "image_size": 320,
        "epochs": 1,
        "batch_size": 1,
        "finetune": {
            "enabled": False,
            "weights": "models/current_best/best.pt",
            "lr": 0.0001,
            "epochs": 1,
            "freeze_backbone": False,
        },
    },
    "models_defaults": {
        "yolo": {"cache_dir": "models/pretrained/yolo", "allow_download": True},
        "rfdetr": {"cache_dir": "models/pretrained/rfdetr", "allow_download": True},
        "rtmdet": {"cache_dir": "models/pretrained/rtmdet", "allow_download": True},
    },
    "models": {
        "yolov8n": {
            "backend": "yolo",
            "asset_id": "yolov8n.pt",
        },
        "rfdetr-nano": {
            "backend": "rfdetr",
            "variant": "nano",
            "asset_id": "rf-detr-nano.pth",
            "resolution": 320,
            "epochs": 1,
            "batch_size": 1,
            "grad_accum_steps": 1,
        },
        "rtmdet-tiny": {
            "backend": "rtmdet",
            "asset_id": "rtmdet_tiny_8xb32-300e_coco",
            "epochs": 1,
            "batch_size": 1,
            "image_size": 320,
        },
    },
    "evaluation": {
        "baseline_weights_path": "models/current_best/best.pt",
    },
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _merge_dict(target: dict, overrides: dict) -> dict:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            target[key] = _merge_dict(target[key], value)
        else:
            target[key] = value
    return target


def _resolve_workspace_path(workspace: Path, raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    candidate = Path(str(raw_path)).expanduser()
    if not candidate.is_absolute():
        candidate = workspace / candidate
    return candidate


def copy_preprovisioned_model_assets(
    *,
    source_workspace: Path,
    destination_workspace: Path,
    model_key: str,
    model_cfg: dict[str, Any],
) -> list[Path]:
    backend = normalize_backend_name(model_cfg.get("backend"))
    source_cache_dir = source_workspace / "models" / "pretrained" / backend
    destination_cache_dir = _resolve_workspace_path(
        destination_workspace,
        str(model_cfg.get("cache_dir", f"models/pretrained/{backend}")),
    )
    if destination_cache_dir is None:
        raise ValueError(f"Could not resolve destination cache directory for {model_key}.")
    asset_id = _require_asset_id(model_key=model_key, model_cfg=model_cfg)

    copied_paths: list[Path] = []
    if backend in {"yolo", "rfdetr"}:
        source_path = source_cache_dir / asset_id
        require_bootstrapped_file(source_path, label=f"preprovisioned {backend} asset")
        destination_path = destination_cache_dir / asset_id
        link_or_copy(source_path, destination_path, prefer_hardlink=False)
        copied_paths.append(destination_path)
        return copied_paths

    source_cfg, source_ckpt, _ = _resolve_rtmdet_assets(
        config_path=None,
        checkpoint_path=None,
        config_name=asset_id,
        cache_dir=source_cache_dir,
    )
    require_bootstrapped_file(source_cfg, label="preprovisioned rtmdet config")
    require_bootstrapped_file(source_ckpt, label="preprovisioned rtmdet checkpoint")
    destination_cfg = destination_cache_dir / source_cfg.name
    destination_ckpt = destination_cache_dir / source_ckpt.name
    link_or_copy(source_cfg, destination_cfg, prefer_hardlink=False)
    link_or_copy(source_ckpt, destination_ckpt, prefer_hardlink=False)
    copied_paths.extend([destination_cfg, destination_ckpt])
    return copied_paths


def create_local_yolo_checkpoint(
    workspace: Path,
    *,
    checkpoint_path: str = "models/pretrained/yolo/yolov8n.pt",
    payload: bytes = b"stub-yolo-checkpoint",
) -> Path:
    resolved_path = _resolve_workspace_path(workspace, checkpoint_path)
    if resolved_path is None:
        raise ValueError("checkpoint_path must be provided for local YOLO checkpoint creation.")
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_bytes(payload)
    return resolved_path


def create_baseline_artifact(
    workspace: Path,
    *,
    weights_path: str = "models/current_best/best.pt",
    experiment_name: str = "baseline",
    model_backend: str = "yolo",
    image_size: int = 320,
    model_variant: str | None = None,
) -> Path:
    baseline_path = _resolve_workspace_path(workspace, weights_path)
    if baseline_path is None:
        raise ValueError("weights_path must be provided for baseline artifact creation.")

    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_bytes(b"baseline-stub")
    metadata = {
        "experiment_name": experiment_name,
        "model_backend": model_backend,
        "image_size": int(image_size),
    }
    if model_variant:
        metadata["model_variant"] = str(model_variant)
    with open(baseline_path.parent / "metadata.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(metadata, f, sort_keys=False)
    return baseline_path


def write_params_yaml(workspace: Path, overrides: dict | None = None) -> dict:
    params = _merge_dict(copy.deepcopy(BASE_PARAMS), overrides or {})
    with open(workspace / "params.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(params, f, sort_keys=False)
    return params


def create_minimal_dataset(base_dir: Path, *, include_empty_train_sample: bool = False) -> None:
    """Create the small synthetic dataset the pipeline tests rely on."""

    # Training folder 1
    images_dir = base_dir / "raw_data" / "train" / "source1" / "images"
    labels_dir = base_dir / "raw_data" / "train" / "source1" / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    img1 = np.full((128, 128, 3), 100, dtype=np.uint8)
    cv2.rectangle(img1, (32, 32), (96, 96), (255, 255, 255), -1)
    cv2.imwrite(str(images_dir / "img1.jpg"), img1)
    with open(labels_dir / "img1.txt", "w", encoding="utf-8") as f:
        f.write("0 0.5 0.5 0.5 0.5\n")

    img2 = np.full((128, 128, 3), 150, dtype=np.uint8)
    cv2.imwrite(str(images_dir / "img2.jpg"), img2)
    with open(labels_dir / "img2.txt", "w", encoding="utf-8") as f:
        f.write("0 0.5 0.5 0.3 0.3\n")

    if include_empty_train_sample:
        img_empty1 = np.full((128, 128, 3), 25, dtype=np.uint8)
        cv2.rectangle(img_empty1, (20, 20), (108, 108), (70, 70, 70), 2)
        cv2.imwrite(str(images_dir / "img4.jpg"), img_empty1)
        # Background-only training frame: label file exists but stays empty.
        (labels_dir / "img4.txt").write_text("", encoding="utf-8")

    # Training folder 2
    images_dir2 = base_dir / "raw_data" / "train" / "source2" / "images"
    labels_dir2 = base_dir / "raw_data" / "train" / "source2" / "labels"
    images_dir2.mkdir(parents=True, exist_ok=True)
    labels_dir2.mkdir(parents=True, exist_ok=True)
    img3 = np.full((128, 128, 3), 60, dtype=np.uint8)
    cv2.line(img3, (0, 0), (127, 127), (255, 255, 255), 3)
    cv2.imwrite(str(images_dir2 / "img3.jpg"), img3)
    with open(labels_dir2 / "img3.txt", "w", encoding="utf-8") as f:
        f.write("0 0.5 0.5 0.2 0.2\n")

    if include_empty_train_sample:
        img4 = np.full((128, 128, 3), 30, dtype=np.uint8)
        cv2.rectangle(img4, (16, 16), (112, 112), (80, 80, 80), 2)
        cv2.imwrite(str(images_dir2 / "img5.jpg"), img4)
        # Background-only training frame: label file exists but stays empty.
        (labels_dir2 / "img5.txt").write_text("", encoding="utf-8")

    # Held-out test scene
    test_images_dir = base_dir / "raw_data" / "test" / "sourceT" / "images"
    test_labels_dir = base_dir / "raw_data" / "test" / "sourceT" / "labels"
    test_images_dir.mkdir(parents=True, exist_ok=True)
    test_labels_dir.mkdir(parents=True, exist_ok=True)
    imgT = np.full((128, 128, 3), 80, dtype=np.uint8)
    cv2.circle(imgT, (64, 64), 20, (255, 255, 255), -1)
    cv2.imwrite(str(test_images_dir / "test1.jpg"), imgT)
    with open(test_labels_dir / "test1.txt", "w", encoding="utf-8") as f:
        f.write("0 0.5 0.5 0.25 0.25\n")


def build_args(dataset_name: str, overrides: dict | None = None):
    """Mirror the CLI arguments our tests pass into the train/prepare stages."""
    args = {
        "stage": None,
        "seed": 42,
        "dataset_name": dataset_name,
        "model": None,
        "val_split": 0.5,
        "test_split": 0.0,
        "recreate_dataset": True,
        "augment_multiplier": 1,
        "folder_subset": None,
    }
    if overrides:
        args.update(overrides)
    from types import SimpleNamespace

    return SimpleNamespace(**args)
