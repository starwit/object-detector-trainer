from __future__ import annotations

import subprocess
import sys
import json
from pathlib import Path

from object_detector_trainer.backends.registry import (
    bootstrap_model_assets,
    is_ready_file,
    normalize_backend_name,
    resolve_workspace_path,
    require_bootstrapped_file,
)
from object_detector_trainer.config.loader import load_config


def _run_subprocess(cmd: list[str], *, cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def _pull_dvc_path(path: Path) -> None:
    # Avoid relying on PATH for `dvc` (fresh venvs often have DVC installed as a
    # module but don't expose the entrypoint in the current PATH).
    _run_subprocess([sys.executable, "-m", "dvc", "pull", str(path)], cwd=Path.cwd())

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

    # Promoted baseline: metadata must exist (pull it if it is DVC-tracked).
    resolved_meta: Path | None = None
    for meta_path in metadata_candidates:
        if meta_path.exists():
            resolved_meta = meta_path
            break
        meta_dvc = meta_path.with_name(f"{meta_path.name}.dvc")
        if meta_dvc.exists():
            _pull_dvc_path(meta_path)
            if meta_path.exists():
                resolved_meta = meta_path
                break
    else:
        raise FileNotFoundError(
            f"Promoted baseline metadata is missing for {baseline_path}. "
            "Expected metadata.yaml next to the weights."
        )

    # Promoted baseline: weights must exist and be non-empty (pull if DVC-tracked).
    if not is_ready_file(baseline_path):
        dvc_file = baseline_path.with_name(f"{baseline_path.name}.dvc")
        if dvc_file.exists():
            _pull_dvc_path(baseline_path)
    require_bootstrapped_file(baseline_path, label="evaluation.baseline_weights_path")
    if resolved_meta is None:
        raise FileNotFoundError(
            f"Promoted baseline metadata is missing for {baseline_path}. "
            "Expected metadata.yaml next to the weights."
        )
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
    if backend == "yolo":
        checkpoint = resolve_workspace_path(model_cfg.get("checkpoint"))
        require_bootstrapped_file(checkpoint, label=f"models.{model_key}.checkpoint")
        return {
            "backend": backend,
            "checkpoint": _fingerprint(Path(checkpoint)),
        }
    if backend == "rfdetr":
        pretrain = resolve_workspace_path(model_cfg.get("pretrain_weights"))
        require_bootstrapped_file(pretrain, label=f"models.{model_key}.pretrain_weights")
        return {
            "backend": backend,
            "pretrain_weights": _fingerprint(Path(pretrain)),
        }

    # RTMDet: use the same resolver as training so the manifest pins the concrete
    # config/checkpoint files in the shared cache.
    from object_detector_trainer.backends import rtmdet as core_rtmdet

    cfg_path, ckpt_path, variant = core_rtmdet._resolve_rtmdet_assets(
        config_path=resolve_workspace_path(model_cfg.get("config_path")),
        checkpoint_path=resolve_workspace_path(model_cfg.get("checkpoint")),
        config_name=str(model_cfg.get("config_name") or "").strip() or None,
        cache_dir=resolve_workspace_path(model_cfg.get("cache_dir")),
        allow_download=bool(model_cfg.get("allow_download", False)),
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
