from __future__ import annotations

from contextlib import suppress
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

from object_detector_trainer.utils.path_ops import link_or_copy
from object_detector_trainer.wrappers.rfdetr import RFDETRModelAdapter

_BACKEND_NAMES = ("yolo", "rfdetr", "rtmdet")
_BACKEND_ALIASES = {
    "yolo": "yolo",
    "rfdetr": "rfdetr",
    "rtmdet": "rtmdet",
}
_REQUIRED_RESOLVED_FIELDS = {
    "yolo": ("checkpoint",),
    "rfdetr": ("rfdetr_variant", "rfdetr_resolution", "rfdetr_batch_size", "rfdetr_checkpoint"),
    "rtmdet": ("rtmdet_config_name", "rtmdet_cache_dir"),
}


def supported_backend_names() -> tuple[str, ...]:
    return _BACKEND_NAMES


def required_resolved_fields(backend: str) -> tuple[str, ...]:
    return _REQUIRED_RESOLVED_FIELDS[normalize_backend_name(backend)]


def normalize_backend_name(model_type: str | None) -> str:
    if model_type is None:
        raise ValueError("Model backend must be configured explicitly.")
    compact = "".join(ch for ch in str(model_type).strip().lower() if ch.isalnum())
    backend = _BACKEND_ALIASES.get(compact)
    if backend:
        return backend
    expected = " | ".join(_BACKEND_NAMES)
    raise ValueError(f"Unsupported backend: {model_type!r}. Expected one of: {expected}")


def resolve_workspace_path(raw_path: str | Path | None) -> Path | None:
    if not raw_path:
        return None
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate


def _coerce_nonempty_str(value: object | None) -> str | None:
    text = str(value or "").strip()
    return text or None


def _default_cache_dir(backend: str) -> Path:
    backend = normalize_backend_name(backend)
    return Path.cwd() / "models" / "pretrained" / backend


def _resolve_cache_dir(*, backend: str, model_cfg: Mapping[str, Any]) -> Path:
    raw = _coerce_nonempty_str(model_cfg.get("cache_dir"))
    return resolve_workspace_path(raw) or _default_cache_dir(backend)


def _require_asset_id(*, model_key: str, model_cfg: Mapping[str, Any]) -> str:
    asset_id = _coerce_nonempty_str(model_cfg.get("asset_id"))
    if asset_id:
        return asset_id
    raise ValueError(f"models.{model_key} must define asset_id.")


def _download_yolo_checkpoint(checkpoint_path: Path) -> Path:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from ultralytics.utils.downloads import attempt_download_asset
    except (ImportError, ModuleNotFoundError) as e:
        raise RuntimeError(
            "YOLO checkpoint download requested but `ultralytics` is not installed."
        ) from e
    downloaded_path = Path(
        attempt_download_asset(checkpoint_path.name, repo="ultralytics/assets")
    ).expanduser()
    if not downloaded_path.is_absolute():
        downloaded_path = Path.cwd() / downloaded_path
    if not is_ready_file(downloaded_path):
        raise FileNotFoundError(
            f"Ultralytics reported a downloaded checkpoint, but no non-empty file exists at {downloaded_path}."
        )
    if downloaded_path.resolve() != checkpoint_path.resolve():
        link_or_copy(downloaded_path, checkpoint_path, prefer_hardlink=False)
    if not is_ready_file(checkpoint_path):
        raise FileNotFoundError(
            f"YOLO checkpoint download did not create a usable file at {checkpoint_path}."
        )
    if downloaded_path.resolve() != checkpoint_path.resolve() and downloaded_path.parent == Path.cwd():
        with suppress(FileNotFoundError):
            downloaded_path.unlink()
    return checkpoint_path


def is_ready_file(path: Path | None) -> bool:
    return path is not None and path.exists() and path.stat().st_size > 0


def require_bootstrapped_file(path: Path | None, *, label: str) -> Path:
    if is_ready_file(path):
        return path  # type: ignore[return-value]
    raise FileNotFoundError(f"{label} is missing or empty after bootstrap: {path}")


def resolve_backend_config(
    *,
    backend: str,
    model_key: str,
    model_cfg: Mapping[str, Any],
    shared_image_size: int,
    shared_epochs: int,
    shared_batch_size: int,
    finetune_enabled: bool,
    finetune_epochs: int | None,
) -> dict[str, Any]:
    backend = normalize_backend_name(backend)

    if backend == "yolo":
        asset_id = _require_asset_id(model_key=model_key, model_cfg=model_cfg)
        cache_dir = _resolve_cache_dir(backend=backend, model_cfg=model_cfg)
        checkpoint_path = cache_dir / asset_id
        resolved = {"checkpoint": str(checkpoint_path)}
        if finetune_enabled and finetune_epochs is not None:
            resolved["epochs"] = int(finetune_epochs)
        return resolved

    if backend == "rtmdet":
        config_name = _require_asset_id(model_key=model_key, model_cfg=model_cfg)
        cache_dir = _resolve_cache_dir(backend=backend, model_cfg=model_cfg)

        batch_size = int(model_cfg.get("batch_size", shared_batch_size))
        explicit_grad_accum = model_cfg.get("grad_accum_steps")
        target_effective_batch = int(model_cfg.get("target_effective_batch", 32))
        if explicit_grad_accum is not None:
            accum = int(explicit_grad_accum)
        else:
            accum = max(1, target_effective_batch // batch_size)

        explicit_lr = model_cfg.get("lr")
        effective_bs = batch_size * accum
        lr = float(explicit_lr) if explicit_lr is not None else 0.004 * effective_bs / 256

        return {
            "rtmdet_config_name": str(config_name),
            "rtmdet_cache_dir": str(cache_dir),
            "rtmdet_lr": lr,
            "rtmdet_accum": accum,
            "rtmdet_device": model_cfg.get("device"),
            "rtmdet_cleanup_tmp": bool(model_cfg.get("cleanup_tmp", False)),
        }

    from object_detector_trainer.backends import rfdetr as core_rfdetr

    variant = str(model_cfg.get("variant") or "").strip()
    if not variant:
        raise ValueError(
            f"models.{model_key} (backend=rfdetr) must define variant explicitly."
        )
    asset_id = _require_asset_id(model_key=model_key, model_cfg=model_cfg)
    cache_dir = _resolve_cache_dir(backend=backend, model_cfg=model_cfg)
    checkpoint_path = cache_dir / asset_id

    batch_size = int(model_cfg.get("batch_size", shared_batch_size))
    explicit_grad_accum = model_cfg.get("grad_accum_steps")
    target_effective_batch = int(model_cfg.get("target_effective_batch", 16))
    if explicit_grad_accum is not None:
        grad_accum = int(explicit_grad_accum)
    else:
        grad_accum = max(1, target_effective_batch // batch_size)

    resolution = core_rfdetr._normalize_rfdetr_resolution(
        variant,
        model_cfg.get("resolution", None),
        int(model_cfg.get("image_size", shared_image_size)),
    )

    return {
        "rfdetr_variant": variant,
        "rfdetr_epochs": int(model_cfg.get("epochs", shared_epochs)),
        "rfdetr_batch_size": batch_size,
        "rfdetr_grad_accum": grad_accum,
        "rfdetr_grad_accum_explicit": explicit_grad_accum is not None,
        "rfdetr_target_effective_batch": target_effective_batch,
        "rfdetr_resolution": int(resolution),
        "rfdetr_lr": model_cfg.get("lr"),
        "rfdetr_checkpoint": str(checkpoint_path),
        "rfdetr_grad_ckpt": model_cfg.get("gradient_checkpointing"),
        "rfdetr_extra": model_cfg.get("extra_train_kwargs"),
    }


def train_backend(backend: str, **kwargs: Any):
    backend = normalize_backend_name(backend)
    if backend == "yolo":
        from object_detector_trainer.backends import yolo as core_yolo

        return core_yolo.train_backend(**kwargs)
    if backend == "rfdetr":
        from object_detector_trainer.backends import rfdetr as core_rfdetr

        return core_rfdetr.train_backend(**kwargs)

    from object_detector_trainer.backends import rtmdet as core_rtmdet

    return core_rtmdet.train_backend(**kwargs)


def build_reload_metadata(
    backend: str,
    model: object,
    resolved_cfg: Mapping[str, Any],
) -> dict[str, object]:
    backend = normalize_backend_name(backend)
    if backend == "yolo":
        return {}

    if backend == "rfdetr":
        model_variant = getattr(model, "model_variant", None) or resolved_cfg.get("rfdetr_variant")
        return {"model_variant": str(model_variant)} if model_variant else {}

    metadata: dict[str, object] = {}
    model_variant = getattr(model, "model_variant", None) or resolved_cfg.get("rtmdet_config_name")
    if model_variant:
        metadata["model_variant"] = str(model_variant)

    for key, attr_name, cfg_key in (
        ("model_config_path", "model_config_path", "model_config_path"),
        ("rtmdet_config_name", "rtmdet_config_name", "rtmdet_config_name"),
        ("rtmdet_cache_dir", "rtmdet_cache_dir", "rtmdet_cache_dir"),
    ):
        value = getattr(model, attr_name, None) or resolved_cfg.get(cfg_key)
        if value:
            metadata[key] = str(value)
    return metadata


def load_backend_model_from_weights(
    backend: str,
    candidate_path: Path,
    meta: Mapping[str, object],
    display_name: str,
    yolo_loader,
) -> object:
    backend = normalize_backend_name(backend)

    if backend == "yolo":
        model_instance = yolo_loader(str(candidate_path))
        setattr(model_instance, "model_backend", "yolo")
        setattr(model_instance, "model_name", str(display_name))
        if meta.get("model_variant"):
            setattr(model_instance, "model_variant", str(meta["model_variant"]))
        if meta.get("image_size") is not None:
            setattr(model_instance, "resolution", int(meta["image_size"]))
        class_names = meta.get("class_names")
        if isinstance(class_names, dict) and class_names:
            setattr(model_instance, "class_names", {int(k): str(v) for k, v in class_names.items()})
        return model_instance

    if backend == "rfdetr":
        from object_detector_trainer.backends import rfdetr as core_rfdetr

        model_variant = str(meta.get("model_variant", "base")).strip().lower() or "base"
        resolution = int(meta.get("image_size", 640) or 640)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        rfdetr_model = core_rfdetr._get_rfdetr_model(
            model_variant=model_variant,
            checkpoint_path=str(candidate_path),
            device=device,
            resolution=int(resolution),
        )
        return RFDETRModelAdapter(
            rfdetr_model,
            model_name=str(display_name),
            resolution=int(resolution),
            model_variant=model_variant,
        )

    from object_detector_trainer.backends import rtmdet as core_rtmdet

    return core_rtmdet.load_rtmdet_baseline(
        weights_path=candidate_path,
        metadata=dict(meta),
        display_name=str(display_name),
    )


def bootstrap_model_assets(model_key: str, model_cfg: Mapping[str, Any]) -> Path:
    backend = normalize_backend_name(model_cfg.get("backend"))

    if backend == "yolo":
        asset_id = _require_asset_id(model_key=model_key, model_cfg=model_cfg)
        cache_dir = _resolve_cache_dir(backend=backend, model_cfg=model_cfg)
        checkpoint = cache_dir / asset_id

        allow_download = bool(model_cfg.get("allow_download", True))
        if not is_ready_file(checkpoint):
            if not allow_download:
                raise FileNotFoundError(
                    f"models.{model_key} YOLO checkpoint is missing: {checkpoint}."
                )
            _download_yolo_checkpoint(checkpoint)
        return require_bootstrapped_file(checkpoint, label=f"models.{model_key}.checkpoint")

    if backend == "rtmdet":
        from object_detector_trainer.backends import rtmdet as core_rtmdet

        config_name = _require_asset_id(model_key=model_key, model_cfg=model_cfg)
        cache_dir = _resolve_cache_dir(backend=backend, model_cfg=model_cfg)
        allow_download = bool(model_cfg.get("allow_download", True))

        try:
            resolved_cfg, resolved_ckpt, _ = core_rtmdet._resolve_rtmdet_assets(
                config_path=None,
                checkpoint_path=None,
                config_name=str(config_name),
                cache_dir=cache_dir,
            )
            if is_ready_file(resolved_cfg) and is_ready_file(resolved_ckpt):
                return resolved_cfg
        except FileNotFoundError:
            pass

        if not allow_download:
            raise FileNotFoundError(
                f"models.{model_key} RTMDet assets are missing under {cache_dir} for config '{config_name}'. "
            )

        cache_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [sys.executable, "-m", "mim", "download", "mmdet", "--config", config_name, "--dest", str(cache_dir)],
            cwd=Path.cwd(),
            check=True,
        )
        resolved_cfg, resolved_ckpt, _ = core_rtmdet._resolve_rtmdet_assets(
            config_path=None,
            checkpoint_path=None,
            config_name=str(config_name),
            cache_dir=cache_dir,
        )
        require_bootstrapped_file(resolved_cfg, label=f"models.{model_key}.config")
        require_bootstrapped_file(resolved_ckpt, label=f"models.{model_key}.checkpoint")
        return resolved_cfg

    asset_id = _require_asset_id(model_key=model_key, model_cfg=model_cfg)
    cache_dir = _resolve_cache_dir(backend=backend, model_cfg=model_cfg)
    checkpoint_path = cache_dir / asset_id

    allow_download = bool(model_cfg.get("allow_download", True))
    if not is_ready_file(checkpoint_path):
        if not allow_download:
            raise FileNotFoundError(
                f"models.{model_key} RF-DETR checkpoint is missing: {checkpoint_path}."
            )
        from rfdetr.main import HOSTED_MODELS
        from rfdetr.util.files import download_file

        url = HOSTED_MODELS.get(checkpoint_path.name)
        if not url:
            raise ValueError(
                f"Unsupported RF-DETR asset for bootstrap: {checkpoint_path.name}. "
                "Use a hosted RF-DETR checkpoint name or provision it manually."
            )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        download_file(url, str(checkpoint_path))
    return require_bootstrapped_file(
        checkpoint_path,
        label=f"models.{model_key}.checkpoint",
    )
