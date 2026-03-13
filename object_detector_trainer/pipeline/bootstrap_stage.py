from __future__ import annotations

import json
from pathlib import Path

from object_detector_trainer.backends.registry import (
    normalize_backend_name,
    resolve_workspace_path,
    require_bootstrapped_file,
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


def _bootstrap_baseline(weights_path: str | Path | None) -> dict[str, object]:
    baseline_path = resolve_workspace_path(weights_path)
    if baseline_path is None:
        return {
            "configured_weights_path": None,
            "promoted": False,
            "weights": None,
            "metadata": None,
        }

    metadata_candidates = (
        baseline_path.parent / "metadata.yaml",
        baseline_path.parent.parent / "metadata.yaml",
    )
    metadata_dvc_candidates = [p.with_name(f"{p.name}.dvc") for p in metadata_candidates]
    baseline_promoted = any(p.exists() for p in metadata_candidates) or any(
        p.exists() for p in metadata_dvc_candidates
    )
    if not baseline_promoted:
        # Template / first-run repos intentionally have no promoted baseline yet.
        return {
            "configured_weights_path": str(baseline_path),
            "promoted": False,
            "weights": None,
            "metadata": None,
        }

    # Promoted baseline: record whether metadata/weights are present locally.
    # Do not auto-run `dvc pull` here: training does not require baselines, and
    # evaluation emits a clear error if a promoted baseline is missing locally.
    resolved_meta: Path | None = None
    for meta_path in metadata_candidates:
        if meta_path.exists():
            resolved_meta = meta_path
            break

    return {
        "configured_weights_path": str(baseline_path),
        "promoted": True,
        "weights": _fingerprint(baseline_path),
        "metadata": _fingerprint(resolved_meta),
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
    baseline_info: dict[str, object],
    model_assets: dict[str, dict[str, object]],
) -> Path:
    manifest = {
        "version": 1,
        "baseline": baseline_info,
        "models": model_assets,
    }
    out_path = Path(".tmp") / "bootstrap_manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def run_bootstrap_stage(args, config=None) -> None:
    cfg = config or load_config(getattr(args, "config", "params.yaml"), args=args)
    baseline_info = _bootstrap_baseline(cfg.evaluation.baseline_weights_path)

    assets: dict[str, dict[str, object]] = {}
    for model_key in _iter_bootstrap_model_keys(args, cfg):
        model_cfg = cfg.models[model_key]
        if not isinstance(model_cfg, dict):
            raise ValueError(f"models.{model_key} must be a mapping.")
        _bootstrap_model_assets(str(model_key), model_cfg)
        assets[str(model_key)] = _resolve_model_assets(str(model_key), model_cfg)

    _write_bootstrap_manifest(baseline_info=baseline_info, model_assets=assets)
