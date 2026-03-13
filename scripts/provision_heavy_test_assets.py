from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from types import SimpleNamespace

import yaml

from object_detector_trainer.backends.registry import require_bootstrapped_file
from object_detector_trainer.backends.rtmdet import _resolve_rtmdet_assets
from object_detector_trainer.pipeline.bootstrap_stage import run_bootstrap_stage
from object_detector_trainer.utils.path_ops import link_or_copy


HEAVY_TEST_MODEL_KEYS = ("yolov8n", "rfdetr-nano", "rtmdet-tiny")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _build_params(root: Path, model_keys: list[str]) -> dict:
    params = {
        "data": {
            "dataset_name": "heavy-test-assets",
            "experiment_name": "heavy-test-assets",
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
            "model": model_keys[0],
            "image_size": 320,
            "epochs": 1,
            "batch_size": 1,
            "finetune": {
                "enabled": False,
                "weights": "models/current_best/best.pt",
            },
        },
        "models_defaults": {
            "yolo": {"cache_dir": str(root / "models" / "pretrained" / "yolo"), "allow_download": True},
            "rfdetr": {"cache_dir": str(root / "models" / "pretrained" / "rfdetr"), "allow_download": True},
            "rtmdet": {"cache_dir": str(root / "models" / "pretrained" / "rtmdet"), "allow_download": True},
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
    params["models"] = {
        model_key: copy.deepcopy(params["models"][model_key])
        for model_key in model_keys
    }
    return params


def _seed_local_yolo_cache(root: Path) -> None:
    source = root / "yolov8n.pt"
    if not source.exists() or source.stat().st_size == 0:
        return
    destination = root / "models" / "pretrained" / "yolo" / "yolov8n.pt"
    if destination.exists() and destination.stat().st_size > 0:
        return
    link_or_copy(source, destination, prefer_hardlink=False)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Provision local pretrained assets required by the backend-heavy test suite."
    )
    parser.add_argument(
        "--model",
        action="append",
        choices=HEAVY_TEST_MODEL_KEYS,
        help="Provision only the selected heavy-test model key. Repeat to provision multiple.",
    )
    args = parser.parse_args()

    root = _repo_root()
    model_keys = list(args.model or HEAVY_TEST_MODEL_KEYS)
    params = _build_params(root, model_keys)
    params_path = root / ".tmp" / "heavy_test_assets_params.yaml"
    params_path.parent.mkdir(parents=True, exist_ok=True)
    params_path.write_text(yaml.safe_dump(params, sort_keys=False), encoding="utf-8")

    _seed_local_yolo_cache(root)

    os.chdir(root)
    bootstrap_args = SimpleNamespace(
        config=str(params_path),
        all_models=True,
        model=None,
        set=[],
    )
    run_bootstrap_stage(bootstrap_args)

    for model_key in model_keys:
        model_cfg = params["models"][model_key]
        backend = str(model_cfg["backend"]).strip().lower()
        cache_dir = root / "models" / "pretrained" / backend
        asset_id = str(model_cfg["asset_id"]).strip()
        if backend == "rtmdet":
            config_path, checkpoint_path, _ = _resolve_rtmdet_assets(
                config_path=None,
                checkpoint_path=None,
                config_name=asset_id,
                cache_dir=cache_dir,
            )
            require_bootstrapped_file(config_path, label=f"{model_key} config")
            require_bootstrapped_file(checkpoint_path, label=f"{model_key} checkpoint")
        else:
            require_bootstrapped_file(cache_dir / asset_id, label=f"{model_key} asset")

    print(f"Provisioned heavy-test assets for: {', '.join(model_keys)}")
    print(f"Asset cache root: {root / 'models' / 'pretrained'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
