from __future__ import annotations

"""Shared helpers for assembling the lightweight datasets/params used in tests."""

import copy
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import yaml

from object_detector_trainer.backends.registry import SUPPORTED_BACKEND_NAMES, normalize_backend_name


# Canonical per-backend test models. The heavy/contract suites derive their
# backend coverage from this mapping so adding a new supported backend forces a
# single explicit test-model choice here instead of drifting across files.
_REPRESENTATIVE_MODEL_SPECS_BY_BACKEND: Dict[str, tuple[str, Dict[str, Any]]] = {
    "yolo": (
        "yolov8n",
        {
            "backend": "yolo",
            "asset_id": "yolov8n.pt",
        },
    ),
    "rfdetr": (
        "rfdetr-nano",
        {
            "backend": "rfdetr",
            "variant": "nano",
            "asset_id": "rf-detr-nano.pth",
            "resolution": 320,
            "epochs": 1,
            "batch_size": 1,
            "grad_accum_steps": 1,
        },
    ),
    "rtmdet": (
        "rtmdet-tiny",
        {
            "backend": "rtmdet",
            "asset_id": "rtmdet_tiny_8xb32-300e_coco",
            "epochs": 1,
            "batch_size": 1,
            "image_size": 320,
        },
    ),
}


def _validated_representative_model_specs() -> Dict[str, tuple[str, Dict[str, Any]]]:
    expected_backends = set(SUPPORTED_BACKEND_NAMES)
    configured_backends = set(_REPRESENTATIVE_MODEL_SPECS_BY_BACKEND)
    if expected_backends != configured_backends:
        missing = sorted(expected_backends - configured_backends)
        extra = sorted(configured_backends - expected_backends)
        raise RuntimeError(
            "Representative backend test coverage drifted. "
            f"Missing backends: {missing or 'none'}. Extra backends: {extra or 'none'}."
        )

    for backend, (model_key, model_cfg) in _REPRESENTATIVE_MODEL_SPECS_BY_BACKEND.items():
        resolved_backend = normalize_backend_name(model_cfg.get("backend"))
        if resolved_backend != backend:
            raise RuntimeError(
                f"Representative model {model_key!r} is registered for backend {backend!r}, "
                f"but its config declares {resolved_backend!r}."
            )
    return _REPRESENTATIVE_MODEL_SPECS_BY_BACKEND


REPRESENTATIVE_MODEL_SPECS_BY_BACKEND = _validated_representative_model_specs()
REPRESENTATIVE_MODEL_KEYS_BY_BACKEND = {
    backend: model_key
    for backend, (model_key, _model_cfg) in REPRESENTATIVE_MODEL_SPECS_BY_BACKEND.items()
}


def representative_model_cases() -> list[tuple[str, str]]:
    return [
        (model_key, backend)
        for backend, model_key in sorted(REPRESENTATIVE_MODEL_KEYS_BY_BACKEND.items())
    ]


def representative_model_key_for_backend(backend: str) -> str:
    normalized_backend = normalize_backend_name(backend)
    if normalized_backend not in REPRESENTATIVE_MODEL_KEYS_BY_BACKEND:
        raise RuntimeError(f"No representative test model configured for backend: {backend!r}")
    return REPRESENTATIVE_MODEL_KEYS_BY_BACKEND[normalized_backend]


def representative_model_config_for_backend(backend: str) -> Dict[str, Any]:
    normalized_backend = normalize_backend_name(backend)
    if normalized_backend not in REPRESENTATIVE_MODEL_SPECS_BY_BACKEND:
        raise RuntimeError(f"No representative test model configured for backend: {backend!r}")
    _model_key, model_cfg = REPRESENTATIVE_MODEL_SPECS_BY_BACKEND[normalized_backend]
    return copy.deepcopy(model_cfg)


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
        "model": REPRESENTATIVE_MODEL_KEYS_BY_BACKEND["yolo"],
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
        model_key: copy.deepcopy(model_cfg)
        for model_key, model_cfg in REPRESENTATIVE_MODEL_SPECS_BY_BACKEND.values()
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


def create_local_rfdetr_checkpoint(
    workspace: Path,
    *,
    checkpoint_path: str = "models/pretrained/rfdetr/rf-detr-nano.pth",
    payload: bytes = b"stub-rfdetr-checkpoint",
) -> Path:
    resolved_path = _resolve_workspace_path(workspace, checkpoint_path)
    if resolved_path is None:
        raise ValueError("checkpoint_path must be provided for local RF-DETR checkpoint creation.")
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_bytes(payload)
    return resolved_path


def create_local_rtmdet_assets(
    workspace: Path,
    *,
    cache_dir: str = "models/pretrained/rtmdet",
    config_name: str = "rtmdet_tiny_8xb32-300e_coco",
) -> tuple[Path, Path]:
    resolved_cache_dir = _resolve_workspace_path(workspace, cache_dir)
    if resolved_cache_dir is None:
        raise ValueError("cache_dir must be provided for local RTMDet asset creation.")
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = resolved_cache_dir / f"{config_name}.py"
    ckpt_path = resolved_cache_dir / f"{config_name}_stub.pth"
    cfg_path.write_text("# rtmdet stub\n", encoding="utf-8")
    ckpt_path.write_bytes(b"stub-rtmdet-checkpoint")
    return cfg_path, ckpt_path


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
