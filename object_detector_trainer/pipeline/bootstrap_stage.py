from __future__ import annotations

import json
from pathlib import Path

from object_detector_trainer.backends.assets import (
    resolve_workspace_path,
    require_bootstrapped_file,
)
from object_detector_trainer.backends.registry import (
    normalize_backend_name,
    bootstrap_model_assets,
)
from object_detector_trainer.config.loader import load_config


def _fingerprint(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    try:
        st = path.stat()
    except FileNotFoundError:
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def _iter_bootstrap_model_keys(args, cfg) -> list[str]:
    if bool(getattr(args, "all_models", False)):
        return sorted(cfg.models.keys())

    selected_model = getattr(args, "model", None) or cfg.train.model
    if selected_model not in cfg.models:
        available = ", ".join(sorted(cfg.models.keys())) or "<none>"
        raise ValueError(f"Unknown model key '{selected_model}'. Available models: {available}")
    return [str(selected_model)]


def _bootstrap_model_assets(model_key: str, model_cfg: dict) -> Path:
    return bootstrap_model_assets(model_key, model_cfg)

def _resolve_model_assets(model_key: str, model_cfg: dict) -> dict[str, object]:
    backend = normalize_backend_name(model_cfg.get("backend"))
    cache_dir = resolve_workspace_path(model_cfg.get("cache_dir")) or (
        Path.cwd() / "models" / "pretrained" / backend
    )
    asset_id = str(model_cfg.get("asset_id") or "").strip()
    if not asset_id:
        raise ValueError(f"models.{model_key} must define asset_id.")
    if backend == "yolo":
        checkpoint = cache_dir / asset_id
        require_bootstrapped_file(checkpoint, label=f"models.{model_key}.checkpoint")
        return {
            "backend": backend,
            "checkpoint": _fingerprint(Path(checkpoint)),
        }
    if backend == "rfdetr":
        checkpoint = cache_dir / asset_id
        require_bootstrapped_file(checkpoint, label=f"models.{model_key}.checkpoint")
        return {
            "backend": backend,
            "checkpoint": _fingerprint(Path(checkpoint)),
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
    require_bootstrapped_file(cfg_path, label=f"models.{model_key}.config")
    require_bootstrapped_file(ckpt_path, label=f"models.{model_key}.checkpoint")
    return {
        "backend": backend,
        "variant": str(variant),
        "config": _fingerprint(cfg_path),
        "checkpoint": _fingerprint(Path(ckpt_path)),
    }


def _write_bootstrap_manifest(
    *,
    model_assets: dict[str, dict[str, object]],
) -> Path:
    manifest = {
        "version": 1,
        "models": model_assets,
    }
    out_path = Path(".tmp") / "bootstrap_manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def run_bootstrap_stage(args, config=None) -> None:
    cfg = config or load_config(getattr(args, "config", "params.yaml"), args=args)

    assets: dict[str, dict[str, object]] = {}
    for model_key in _iter_bootstrap_model_keys(args, cfg):
        model_cfg = cfg.models[model_key]
        if not isinstance(model_cfg, dict):
            raise ValueError(f"models.{model_key} must be a mapping.")
        _bootstrap_model_assets(str(model_key), model_cfg)
        assets[str(model_key)] = _resolve_model_assets(str(model_key), model_cfg)

    _write_bootstrap_manifest(model_assets=assets)
