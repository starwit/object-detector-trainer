from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from object_detector_trainer.backends.training_config import normalize_backend_name
from object_detector_trainer.config.loader import load_config


def _resolve_workspace_path(raw_path: str | Path | None) -> Path | None:
    if not raw_path:
        return None
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate


def _is_ready_file(path: Path | None) -> bool:
    return path is not None and path.exists() and path.stat().st_size > 0


def _run_subprocess(cmd: list[str], *, cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def _pull_dvc_path(path: Path) -> None:
    # Avoid relying on PATH for `dvc` (fresh venvs often have DVC installed as a
    # module but don't expose the entrypoint in the current PATH).
    _run_subprocess([sys.executable, "-m", "dvc", "pull", str(path)], cwd=Path.cwd())


def _download_yolo_asset(target: Path) -> None:
    from ultralytics.utils.downloads import attempt_download_asset

    target.parent.mkdir(parents=True, exist_ok=True)
    # Ultralytics may return a cached path under SETTINGS["weights_dir"] without
    # materializing the file at our configured location. Keep the workspace
    # canonical by copying/moving the asset into `target`.
    # Use Ultralytics' default pinned release tag. "latest" is not a valid tag
    # for ultralytics/assets and results in 404s.
    downloaded = Path(attempt_download_asset(target.name))
    if not downloaded.is_absolute():
        downloaded = (Path.cwd() / downloaded).resolve()
    else:
        downloaded = downloaded.resolve()

    target = target.resolve()
    if downloaded == target:
        return

    cwd = Path.cwd().resolve()
    if downloaded.parent == cwd:
        downloaded.replace(target)
    else:
        shutil.copy2(downloaded, target)


def _download_rtmdet_assets(*, config_name: str, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    _run_subprocess(
        [sys.executable, "-m", "mim", "download", "mmdet", "--config", config_name, "--dest", str(cache_dir)],
        cwd=Path.cwd(),
    )


def _download_rfdetr_asset(target: Path) -> None:
    from rfdetr.main import HOSTED_MODELS
    from rfdetr.util.files import download_file

    url = HOSTED_MODELS.get(target.name)
    if not url:
        raise ValueError(
            f"Unsupported RF-DETR pretrained asset for bootstrap: {target.name}. "
            "Use a hosted RF-DETR checkpoint name or provision it manually."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    download_file(url, str(target))


def _require_bootstrapped_file(path: Path | None, *, label: str) -> Path:
    if _is_ready_file(path):
        return path  # type: ignore[return-value]
    raise FileNotFoundError(f"{label} is missing or empty after bootstrap: {path}")


def _bootstrap_baseline(weights_path: str | Path | None) -> Path | None:
    baseline_path = _resolve_workspace_path(weights_path)
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
    if not _is_ready_file(baseline_path):
        dvc_file = baseline_path.with_name(f"{baseline_path.name}.dvc")
        if dvc_file.exists():
            _pull_dvc_path(baseline_path)
    _require_bootstrapped_file(baseline_path, label="evaluation.baseline_weights_path")
    return baseline_path


def _bootstrap_yolo_model(model_key: str, model_cfg: dict) -> Path:
    checkpoint = _resolve_workspace_path(model_cfg.get("checkpoint"))
    if checkpoint is None:
        raise ValueError(f"models.{model_key} (backend=yolo) must define checkpoint.")
    if not _is_ready_file(checkpoint):
        _download_yolo_asset(checkpoint)
    return _require_bootstrapped_file(checkpoint, label=f"models.{model_key}.checkpoint")


def _bootstrap_rtmdet_model(model_key: str, model_cfg: dict) -> Path:
    from object_detector_trainer.backends import rtmdet as rtmdet_backend

    config_name = str(model_cfg.get("config_name") or model_cfg.get("variant") or "").strip()
    config_path = _resolve_workspace_path(model_cfg.get("config_path"))
    checkpoint_path = _resolve_workspace_path(model_cfg.get("checkpoint"))
    cache_dir = _resolve_workspace_path(model_cfg.get("cache_dir")) or (Path.cwd() / "models" / "pretrained" / "rtmdet")

    if config_path is not None and not _is_ready_file(config_path):
        raise FileNotFoundError(
            f"models.{model_key}.config_path is missing: {config_path}. "
            "Bootstrap only supports config_name-based RTMDet downloads."
        )
    if checkpoint_path is not None and not _is_ready_file(checkpoint_path):
        raise FileNotFoundError(
            f"models.{model_key}.checkpoint is missing: {checkpoint_path}. "
            "Bootstrap only supports cache_dir/config_name-based RTMDet downloads."
        )

    try:
        resolved_cfg, resolved_ckpt, _ = rtmdet_backend._resolve_rtmdet_assets(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            config_name=config_name or None,
            cache_dir=cache_dir,
            allow_download=False,
        )
        if _is_ready_file(resolved_cfg) and _is_ready_file(resolved_ckpt):
            return resolved_cfg
    except FileNotFoundError:
        pass

    if not config_name:
        raise ValueError(
            f"models.{model_key} (backend=rtmdet) must define config_name for bootstrap."
        )

    _download_rtmdet_assets(config_name=config_name, cache_dir=cache_dir)
    resolved_cfg, resolved_ckpt, _ = rtmdet_backend._resolve_rtmdet_assets(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        config_name=config_name,
        cache_dir=cache_dir,
        allow_download=False,
    )
    _require_bootstrapped_file(resolved_cfg, label=f"models.{model_key}.config")
    _require_bootstrapped_file(resolved_ckpt, label=f"models.{model_key}.checkpoint")
    return resolved_cfg


def _bootstrap_rfdetr_model(model_key: str, model_cfg: dict) -> Path:
    pretrain_weights = _resolve_workspace_path(model_cfg.get("pretrain_weights"))
    if pretrain_weights is None:
        raise ValueError(
            f"models.{model_key} (backend=rfdetr) must define pretrain_weights."
        )
    if not _is_ready_file(pretrain_weights):
        _download_rfdetr_asset(pretrain_weights)
    return _require_bootstrapped_file(
        pretrain_weights,
        label=f"models.{model_key}.pretrain_weights",
    )


def _iter_bootstrap_model_keys(args, cfg) -> list[str]:
    if bool(getattr(args, "all_models", False)):
        return sorted(cfg.models.keys())

    selected_model = getattr(args, "model", None) or cfg.train.model
    if selected_model not in cfg.models:
        available = ", ".join(sorted(cfg.models.keys())) or "<none>"
        raise ValueError(f"Unknown model key '{selected_model}'. Available models: {available}")
    return [str(selected_model)]


def _bootstrap_model_assets(model_key: str, model_cfg: dict) -> Path:
    backend = normalize_backend_name(model_cfg.get("backend"))
    if backend == "yolo":
        return _bootstrap_yolo_model(model_key, model_cfg)
    if backend == "rtmdet":
        return _bootstrap_rtmdet_model(model_key, model_cfg)
    if backend == "rfdetr":
        return _bootstrap_rfdetr_model(model_key, model_cfg)
    raise ValueError(f"Unsupported backend for bootstrap: {backend!r}")


def run_bootstrap_stage(args, config=None) -> None:
    cfg = config or load_config(getattr(args, "config", "params.yaml"), args=args)
    _bootstrap_baseline(cfg.evaluation.baseline_weights_path)

    for model_key in _iter_bootstrap_model_keys(args, cfg):
        model_cfg = cfg.models[model_key]
        if not isinstance(model_cfg, dict):
            raise ValueError(f"models.{model_key} must be a mapping.")
        _bootstrap_model_assets(str(model_key), model_cfg)
