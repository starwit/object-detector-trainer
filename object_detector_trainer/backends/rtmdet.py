from __future__ import annotations

import gc
import json
import logging
import shutil
import subprocess
from pathlib import Path

import cv2
import torch

from object_detector_trainer.datasets.yolo_yaml import get_dataset_classes
from object_detector_trainer.utils.path_ops import resolve_unique_run_dir, safe_dataset_dirname

logger = logging.getLogger(__name__)


def _cleanup_mmengine_singletons() -> None:
    """Close MMEngine global singletons so their file handles / log queues are
    released eagerly, before the GC or interpreter shutdown races with them.

    MMEngine stores logger, message-hub, and scope instances in class-level
    OrderedDicts (ManagerMixin._instance_dict).  Clearing those dicts makes the
    objects available for immediate GC instead of living until process exit,
    which would otherwise leave QueueFeederThread / Connection file descriptors
    open past the point where they are still valid.
    """
    try:
        from mmengine.logging.logger import MMLogger
        from mmengine.logging.message_hub import MessageHub
        from mmengine.registry.default_scope import DefaultScope
    except ImportError:
        return

    for inst in list(MMLogger._instance_dict.values()):
        for handler in list(inst.handlers):
            try:
                handler.close()
            except Exception:
                pass
            inst.removeHandler(handler)
    MMLogger._instance_dict.clear()
    MessageHub._instance_dict.clear()
    DefaultScope._instance_dict.clear()


def _iter_image_files(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        return []
    return sorted(
        p
        for p in images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )


def _yolo_split_to_coco(
    *,
    images_dir: Path,
    labels_dir: Path,
    categories: list[dict],
) -> dict:
    cat_ids = {int(cat["id"]) for cat in categories}
    images: list[dict] = []
    annotations: list[dict] = []
    image_id = 1
    ann_id = 1

    for image_path in _iter_image_files(images_dir):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )

        label_path = labels_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    try:
                        cls_id = int(float(parts[0]))
                        cx = float(parts[1])
                        cy = float(parts[2])
                        bw = float(parts[3])
                        bh = float(parts[4])
                    except ValueError:
                        logger.debug(
                            "Skipping invalid YOLO label line in %s: %r",
                            label_path,
                            line.strip(),
                        )
                        continue
                    if cls_id not in cat_ids:
                        continue

                    x = (cx - bw / 2.0) * width
                    y = (cy - bh / 2.0) * height
                    box_w = bw * width
                    box_h = bh * height

                    x = max(0.0, min(x, float(width)))
                    y = max(0.0, min(y, float(height)))
                    box_w = max(0.0, min(box_w, float(width) - x))
                    box_h = max(0.0, min(box_h, float(height) - y))
                    if box_w <= 0.0 or box_h <= 0.0:
                        continue

                    annotations.append(
                        {
                            "id": ann_id,
                            "image_id": image_id,
                            "category_id": cls_id,
                            "bbox": [x, y, box_w, box_h],
                            "area": box_w * box_h,
                            "iscrowd": 0,
                        }
                    )
                    ann_id += 1
        image_id += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def _prepare_rtmdet_coco_layout(training_path: Path, dataset_name: str) -> tuple[Path, dict[int, str]]:
    base_dir = Path(".tmp") / "rtmdet_datasets"
    output_dir = base_dir / safe_dataset_dirname(str(dataset_name))
    if not output_dir.resolve(strict=False).is_relative_to(base_dir.resolve(strict=False)):
        raise ValueError(f"Unsafe dataset_name for MMDetection export dir: {dataset_name!r}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "train").symlink_to((training_path / "train").resolve())
    (output_dir / "val").symlink_to((training_path / "val").resolve())
    ann_dir = output_dir / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    class_names, _ = get_dataset_classes(training_path / "dataset.yaml")
    categories = [{"id": cls_id, "name": name} for cls_id, name in class_names.items()]

    train_coco = _yolo_split_to_coco(
        images_dir=training_path / "train" / "images",
        labels_dir=training_path / "train" / "labels",
        categories=categories,
    )
    val_coco = _yolo_split_to_coco(
        images_dir=training_path / "val" / "images",
        labels_dir=training_path / "val" / "labels",
        categories=categories,
    )

    with open(ann_dir / "instances_train.json", "w", encoding="utf-8") as f:
        json.dump(train_coco, f)
    with open(ann_dir / "instances_val.json", "w", encoding="utf-8") as f:
        json.dump(val_coco, f)

    return output_dir, class_names


def _resolve_path(path_like: str | Path | None) -> Path | None:
    if not path_like:
        return None
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _download_rtmdet_assets(config_name: str, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "mim",
        "download",
        "mmdet",
        "--config",
        str(config_name),
        "--dest",
        str(cache_dir),
    ]
    try:
        subprocess.run(cmd, check=True, timeout=60 * 30)
    except FileNotFoundError as e:
        raise RuntimeError(
            "OpenMIM is required for RTMDet auto-download. "
            "Install it (e.g. `pip install openmim`) or provide models.<key>.config_path/checkpoint."
        ) from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"Timed out downloading MMDetection assets for config '{config_name}'. "
            f"Command: {' '.join(cmd)}"
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to download MMDetection assets for config '{config_name}'. Command: {' '.join(cmd)}"
        ) from e


def _resolve_rtmdet_assets(
    *,
    config_path: str | Path | None,
    checkpoint_path: str | Path | None,
    config_name: str | None,
    cache_dir: str | Path | None,
    allow_download: bool,
) -> tuple[Path, Path | None, str]:
    cfg_path = _resolve_path(config_path)
    ckpt_path = _resolve_path(checkpoint_path)
    cache_root = _resolve_path(cache_dir) or (Path.cwd() / "models" / "pretrained" / "rtmdet")
    variant = str(config_name or "").strip()

    if cfg_path is not None and not cfg_path.exists():
        raise FileNotFoundError(f"MMDetection config_path does not exist: {cfg_path}")
    if ckpt_path is not None and not ckpt_path.exists():
        raise FileNotFoundError(f"MMDetection checkpoint does not exist: {ckpt_path}")

    if cfg_path is None:
        if not variant:
            raise ValueError("MMDetection backend needs either config_path or config_name.")
        if allow_download:
            _download_rtmdet_assets(variant, cache_root)

        candidate = cache_root / f"{variant}.py"
        if candidate.exists():
            cfg_path = candidate
        else:
            matches = sorted(cache_root.glob(f"**/{variant}.py"))
            if not matches:
                raise FileNotFoundError(
                    f"Could not find config '{variant}.py' under {cache_root}. "
                    "Use `mim download mmdet --config <name> --dest <cache_dir>` "
                    "or set models.<key>.config_path."
                )
            cfg_path = matches[-1]

    if ckpt_path is None:
        prefixed = sorted(cache_root.glob(f"**/{variant}*.pth")) if variant else []
        any_pth = sorted(cache_root.glob("**/*.pth"))
        candidates = prefixed or any_pth
        if candidates:
            ckpt_path = max(candidates, key=lambda p: p.stat().st_mtime)

    resolved_variant = variant or cfg_path.stem
    return cfg_path, ckpt_path, resolved_variant


def _patch_pipeline_scales(node, image_size: int) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key in {"scale", "img_scale"} and isinstance(value, (list, tuple)) and len(value) == 2:
                node[key] = (int(image_size), int(image_size))
            else:
                _patch_pipeline_scales(value, image_size)
        return
    if isinstance(node, list):
        for item in node:
            _patch_pipeline_scales(item, image_size)


def _configure_dataset(dataset_cfg: dict, *, data_root: Path, ann_file: str, img_prefix: str, classes: tuple[str, ...]) -> None:
    if "dataset" in dataset_cfg and isinstance(dataset_cfg["dataset"], dict):
        _configure_dataset(
            dataset_cfg["dataset"],
            data_root=data_root,
            ann_file=ann_file,
            img_prefix=img_prefix,
            classes=classes,
        )
    dataset_cfg["data_root"] = str(data_root)
    dataset_cfg["ann_file"] = ann_file
    dataset_cfg["data_prefix"] = {"img": img_prefix}
    dataset_cfg["metainfo"] = {"classes": classes}


def _configure_evaluator_ann_file(evaluator_cfg, *, ann_file: str) -> None:
    if isinstance(evaluator_cfg, list):
        for item in evaluator_cfg:
            _configure_evaluator_ann_file(item, ann_file=ann_file)
        return
    if not isinstance(evaluator_cfg, dict):
        return
    if "ann_file" in evaluator_cfg:
        evaluator_cfg["ann_file"] = ann_file
    for value in evaluator_cfg.values():
        _configure_evaluator_ann_file(value, ann_file=ann_file)


def _set_num_classes(model_cfg: dict, num_classes: int) -> None:
    bbox_head = model_cfg.get("bbox_head")
    if isinstance(bbox_head, dict):
        if "num_classes" in bbox_head:
            bbox_head["num_classes"] = int(num_classes)
        if isinstance(bbox_head.get("head_module"), dict) and "num_classes" in bbox_head["head_module"]:
            bbox_head["head_module"]["num_classes"] = int(num_classes)
    elif isinstance(bbox_head, list):
        for head in bbox_head:
            if isinstance(head, dict) and "num_classes" in head:
                head["num_classes"] = int(num_classes)


def _find_best_checkpoint(run_dir: Path) -> Path | None:
    best_candidates = sorted(run_dir.glob("best*.pth"))
    if best_candidates:
        return max(best_candidates, key=lambda p: p.stat().st_mtime)

    latest = run_dir / "latest.pth"
    if latest.exists() and latest.stat().st_size > 0:
        return latest

    epoch_candidates = sorted(run_dir.glob("epoch_*.pth"))
    if epoch_candidates:
        return max(epoch_candidates, key=lambda p: p.stat().st_mtime)
    return None


def _save_rtmdet_weights(output_dir: Path, source_ckpt: Path) -> Path:
    weights_dir = output_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    dest = weights_dir / "best.pt"
    checkpoint = torch.load(source_ckpt, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        torch.save({"state_dict": checkpoint["state_dict"]}, dest)
    else:
        torch.save(checkpoint, dest)
    return dest


def load_rtmdet_baseline(
    *,
    weights_path: Path,
    metadata: dict,
    display_name: str,
):
    try:
        import mmcv._ext  # type: ignore  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError) as e:
        raise RuntimeError(
            "MMDetection baseline loading requires full mmcv ops. "
            "Install `mmcv` (not `mmcv-lite`) matching your PyTorch/CUDA build."
        ) from e

    config_path = weights_path.parent / "model_config.py"
    if not config_path.exists():
        config_from_meta = _resolve_path(metadata.get("model_config_path"))
        if config_from_meta and config_from_meta.exists():
            config_path = config_from_meta
        else:
            variant = (
                metadata.get("rtmdet_config_name")
                or metadata.get("mmdet_config_name")
                or metadata.get("model_variant")
            )
            if not variant:
                raise RuntimeError(
                    "MMDetection baseline metadata must include model_config_path or model_variant."
                )
            config_path, _unused_ckpt, _ = _resolve_rtmdet_assets(
                config_path=None,
                checkpoint_path=None,
                config_name=str(variant),
                cache_dir=metadata.get("rtmdet_cache_dir") or metadata.get("mmdet_cache_dir"),
                allow_download=bool(
                    metadata.get(
                        "rtmdet_allow_download",
                        metadata.get("mmdet_allow_download", True),
                    )
                ),
            )

    try:
        from mmdet.apis import init_detector
    except (ImportError, ModuleNotFoundError, OSError) as e:
        raise RuntimeError(
            "MMDetection baseline loading failed. Ensure `mmdet` is installed and full `mmcv` "
            "(not `mmcv-lite`) is available."
        ) from e

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    detector = init_detector(str(config_path), str(weights_path), device=device)

    class_names: dict[int, str] = {}
    names_raw = metadata.get("class_names")
    if isinstance(names_raw, list):
        class_names = {i: str(name) for i, name in enumerate(names_raw)}
    elif isinstance(names_raw, dict):
        class_names = {int(k): str(v) for k, v in names_raw.items()}

    from object_detector_trainer.wrappers.rtmdet import RTMDetModelAdapter

    return RTMDetModelAdapter(
        detector,
        model_name=str(display_name),
        resolution=int(metadata.get("image_size", 640) or 640),
        class_names=class_names,
        model_variant=str(metadata.get("model_variant", "")) or None,
        model_config_path=str(config_path),
    )


def train_rtmdet_backend(
    *,
    training_path: Path,
    test_path: Path,
    dataset_name: str,
    resolved_cfg: dict,
    experiment_name: str | None,
) -> tuple[object, Path, str, int, int]:
    try:
        import mmcv._ext  # type: ignore  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError) as e:
        raise RuntimeError(
            "RTMDet backend requires full mmcv ops. Install `mmcv` (not `mmcv-lite`) "
            "matching your PyTorch/CUDA build."
        ) from e

    try:
        from mmengine.config import Config
        from mmengine.runner import Runner
        from mmdet.apis import init_detector
    except (ImportError, ModuleNotFoundError, OSError) as e:
        raise RuntimeError(
            "RTMDet backend requires `mmdet`, `mmengine`, and full `mmcv` "
            "(not `mmcv-lite`)."
        ) from e

    from object_detector_trainer.wrappers.rtmdet import RTMDetModelAdapter

    dataset_dir, class_names = _prepare_rtmdet_coco_layout(training_path, str(dataset_name))
    classes_tuple = tuple(class_names[i] for i in sorted(class_names))

    cfg_path, ckpt_path, variant = _resolve_rtmdet_assets(
        config_path=resolved_cfg.get("rtmdet_config_path"),
        checkpoint_path=resolved_cfg.get("rtmdet_checkpoint"),
        config_name=resolved_cfg.get("rtmdet_config_name"),
        cache_dir=resolved_cfg.get("rtmdet_cache_dir"),
        allow_download=bool(resolved_cfg.get("rtmdet_allow_download", True)),
    )

    run_name = experiment_name or f"{resolved_cfg['model_key']}-rtmdet"
    runs_root = Path("runs") / "rtmdet"
    runs_root.mkdir(parents=True, exist_ok=True)
    output_dir = resolve_unique_run_dir(runs_root, run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(str(cfg_path))
    cfg.work_dir = str(output_dir)
    cfg.load_from = str(ckpt_path) if ckpt_path is not None else None
    cfg.resume = False
    cfg.train_cfg["max_epochs"] = int(resolved_cfg["epochs"])
    cfg.randomness = {"seed": int(resolved_cfg.get("seed", 42)), "deterministic": True}
    cfg.default_hooks.setdefault("checkpoint", {})
    cfg.default_hooks["checkpoint"]["save_best"] = "coco/bbox_mAP"
    cfg.default_hooks["checkpoint"].setdefault("rule", "greater")
    cfg.default_hooks["checkpoint"].setdefault("max_keep_ckpts", 1)

    _configure_dataset(
        cfg.train_dataloader["dataset"],
        data_root=dataset_dir,
        ann_file="annotations/instances_train.json",
        img_prefix="train/images/",
        classes=classes_tuple,
    )
    _configure_dataset(
        cfg.val_dataloader["dataset"],
        data_root=dataset_dir,
        ann_file="annotations/instances_val.json",
        img_prefix="val/images/",
        classes=classes_tuple,
    )
    if "test_dataloader" in cfg and "dataset" in cfg["test_dataloader"]:
        _configure_dataset(
            cfg.test_dataloader["dataset"],
            data_root=dataset_dir,
            ann_file="annotations/instances_val.json",
            img_prefix="val/images/",
            classes=classes_tuple,
        )
    evaluator_ann_file = str((dataset_dir / "annotations" / "instances_val.json").resolve())
    if "val_evaluator" in cfg:
        _configure_evaluator_ann_file(cfg.val_evaluator, ann_file=evaluator_ann_file)
    if "test_evaluator" in cfg:
        _configure_evaluator_ann_file(cfg.test_evaluator, ann_file=evaluator_ann_file)

    cfg.train_dataloader["batch_size"] = int(resolved_cfg["batch_size"])
    # Disable persistent workers so dataloader workers exit cleanly at the end
    # of each epoch (and after training) rather than staying alive until the
    # Runner is garbage-collected.  persistent_workers=True causes a race
    # between worker teardown and the QueueFeederThread, which produces
    # spurious "Bad file descriptor" / semaphore-over-release warnings.
    # Validation runs with num_workers=0 (same pattern as evaluate_stage).
    cfg.train_dataloader["persistent_workers"] = False
    cfg.val_dataloader["num_workers"] = 0
    cfg.val_dataloader["persistent_workers"] = False
    if "test_dataloader" in cfg:
        cfg.test_dataloader["num_workers"] = 0
        cfg.test_dataloader["persistent_workers"] = False
    _set_num_classes(cfg["model"], len(classes_tuple))
    _patch_pipeline_scales(cfg, int(resolved_cfg["image_size"]))

    if resolved_cfg.get("rtmdet_lr") is not None:
        opt_wrapper = cfg.get("optim_wrapper")
        if isinstance(opt_wrapper, dict):
            optimizer = opt_wrapper.get("optimizer")
            if isinstance(optimizer, dict) and "lr" in optimizer:
                optimizer["lr"] = float(resolved_cfg["rtmdet_lr"])

    runner = Runner.from_cfg(cfg)
    # PyTorch ≥2.6 changed weights_only default from False to True.  RTMDet
    # checkpoints bundle mmengine HistoryBuffer objects which are not in the
    # default safe-globals allow-list, causing torch.load to reject them.
    # Register the class explicitly before training so the checkpoint loads.
    try:
        from mmengine.logging.history_buffer import HistoryBuffer as _HistoryBuffer  # type: ignore

        torch.serialization.add_safe_globals([_HistoryBuffer])
    except (ImportError, AttributeError):
        pass
    runner.train()
    # Explicitly release the Runner so fork-based dataloader workers are
    # cleaned up while the interpreter is still healthy.  Without this,
    # MMEngine's worker queues are collected during Python shutdown, which
    # produces spurious "Bad file descriptor" / semaphore-over-release errors.
    del runner
    _cleanup_mmengine_singletons()
    gc.collect()

    best_ckpt = _find_best_checkpoint(output_dir)
    if best_ckpt is None:
        raise FileNotFoundError(f"No MMDetection checkpoint found in {output_dir}")
    best_weights = _save_rtmdet_weights(output_dir, best_ckpt)

    local_config = output_dir / "model_config.py"
    cfg.dump(str(local_config))

    display_name = f"{run_name}-rtmdet"
    class_names_map, _ = get_dataset_classes(test_path / "dataset.yaml")
    device = str(resolved_cfg.get("rtmdet_device") or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    detector = init_detector(str(local_config), str(best_weights), device=device)
    model = RTMDetModelAdapter(
        detector,
        model_name=display_name,
        resolution=int(resolved_cfg["image_size"]),
        class_names=class_names_map,
        model_variant=variant,
        model_config_path=str(local_config),
        config_name=variant,
        cache_dir=str(resolved_cfg.get("rtmdet_cache_dir") or "models/pretrained/rtmdet"),
        allow_download=bool(resolved_cfg.get("rtmdet_allow_download", True)),
    )

    if bool(resolved_cfg.get("rtmdet_cleanup_tmp", False)):
        shutil.rmtree(dataset_dir, ignore_errors=True)

    return model, output_dir, display_name, int(resolved_cfg["image_size"]), int(resolved_cfg["epochs"])


def train_backend(
    *,
    training_path: Path,
    test_path: Path,
    dataset_name: str,
    resolved_cfg: dict,
    experiment_name: str | None,
) -> tuple[object, Path, str, int, int]:
    return train_rtmdet_backend(
        training_path=training_path,
        test_path=test_path,
        dataset_name=dataset_name,
        resolved_cfg=resolved_cfg,
        experiment_name=experiment_name,
    )


def load_mmdet_baseline(*, weights_path: Path, metadata: dict, display_name: str):
    """Backward-compatible alias for legacy call sites."""
    return load_rtmdet_baseline(
        weights_path=weights_path,
        metadata=metadata,
        display_name=display_name,
    )


def train_mmdet_backend(
    *,
    training_path: Path,
    test_path: Path,
    dataset_name: str,
    resolved_cfg: dict,
    experiment_name: str | None,
) -> tuple[object, Path, str, int, int]:
    """Backward-compatible alias for legacy call sites."""
    return train_rtmdet_backend(
        training_path=training_path,
        test_path=test_path,
        dataset_name=dataset_name,
        resolved_cfg=resolved_cfg,
        experiment_name=experiment_name,
    )


__all__ = [
    "train_backend",
    "train_rtmdet_backend",
    "train_mmdet_backend",
    "load_rtmdet_baseline",
    "load_mmdet_baseline",
]
