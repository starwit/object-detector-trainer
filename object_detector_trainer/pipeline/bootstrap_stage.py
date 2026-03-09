from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from object_detector_trainer.backends.registry import (
    bootstrap_model_assets,
    is_ready_file,
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


def _bootstrap_baseline(weights_path: str | Path | None) -> Path | None:
    baseline_path = resolve_workspace_path(weights_path)
    if baseline_path is None:
        return None

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
        return None

    # Promoted baseline: metadata must exist (pull it if it is DVC-tracked).
    for meta_path in metadata_candidates:
        if meta_path.exists():
            break
        meta_dvc = meta_path.with_name(f"{meta_path.name}.dvc")
        if meta_dvc.exists():
            _pull_dvc_path(meta_path)
            if meta_path.exists():
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
    return baseline_path


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


def run_bootstrap_stage(args, config=None) -> None:
    cfg = config or load_config(getattr(args, "config", "params.yaml"), args=args)
    _bootstrap_baseline(cfg.evaluation.baseline_weights_path)

    for model_key in _iter_bootstrap_model_keys(args, cfg):
        model_cfg = cfg.models[model_key]
        if not isinstance(model_cfg, dict):
            raise ValueError(f"models.{model_key} must be a mapping.")
        _bootstrap_model_assets(str(model_key), model_cfg)
