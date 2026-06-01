from __future__ import annotations

import json
from pathlib import Path

from object_detector_trainer.backends.assets import (
    resolve_cache_dir,
    require_asset_id,
    require_bootstrapped_file,
)
from object_detector_trainer.backends.registry import (
    normalize_backend_name,
    bootstrap_model_assets,
)
from object_detector_trainer.config.loader import load_config


def _fingerprint(path: Path) -> dict[str, object]:
    st = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def _resolve_model_assets(model_key: str, model_cfg: dict) -> dict[str, object]:
    backend = normalize_backend_name(model_cfg.get("backend"))
    cache_dir = resolve_cache_dir(backend=backend, model_cfg=model_cfg)
    asset_id = require_asset_id(model_key=model_key, model_cfg=model_cfg)

    if backend in {"yolo", "rfdetr"}:
        checkpoint = cache_dir / asset_id
        checkpoint = require_bootstrapped_file(
            checkpoint,
            label=f"models.{model_key}.checkpoint",
        )
        return {
            "backend": backend,
            "checkpoint": _fingerprint(checkpoint),
        }

    # RTMDet: use the same resolver as training so the manifest pins the concrete
    # config/checkpoint files in the shared cache.
    from object_detector_trainer.backends import rtmdet as core_rtmdet

    cfg_path, ckpt_path, variant = core_rtmdet._resolve_rtmdet_assets(
        config_path=None,
        checkpoint_path=None,
        config_name=asset_id,
        cache_dir=cache_dir,
    )
    cfg_path = require_bootstrapped_file(cfg_path, label=f"models.{model_key}.config")
    ckpt_path = require_bootstrapped_file(
        ckpt_path,
        label=f"models.{model_key}.checkpoint",
    )
    return {
        "backend": backend,
        "variant": str(variant),
        "config": _fingerprint(cfg_path),
        "checkpoint": _fingerprint(ckpt_path),
    }


def run_bootstrap_stage(args, config=None) -> None:
    cfg = config or load_config(getattr(args, "config", "params.yaml"), args=args)

    if bool(getattr(args, "all_models", False)):
        model_keys = sorted(cfg.models.keys())
    else:
        selected_model = getattr(args, "model", None) or cfg.train.model
        if selected_model not in cfg.models:
            available = ", ".join(sorted(cfg.models.keys())) or "<none>"
            raise ValueError(f"Unknown model key '{selected_model}'. Available models: {available}")
        model_keys = [str(selected_model)]

    assets: dict[str, dict[str, object]] = {}
    for model_key in model_keys:
        model_cfg = cfg.models[model_key]
        bootstrap_model_assets(str(model_key), model_cfg)
        assets[str(model_key)] = _resolve_model_assets(str(model_key), model_cfg)

    out_path = Path(".tmp") / "bootstrap_manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"version": 1, "models": assets}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
