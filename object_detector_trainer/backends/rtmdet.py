from __future__ import annotations

import gc
import json
import logging
import os
import shutil
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
            except (OSError, RuntimeError, ValueError):
                logger.debug("Ignoring expected RTMDet logger handler close failure.", exc_info=True)
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


def _resolve_baseline_config_path(weights_path: Path, metadata: dict) -> Path | None:
    adjacent_config = weights_path.parent / "model_config.py"
    if adjacent_config.exists():
        return adjacent_config

    raw_config_path = metadata.get("model_config_path")
    if not raw_config_path:
        return None

    raw_path = Path(str(raw_config_path)).expanduser()
    candidates: list[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append(weights_path.parent / raw_path)
        candidates.append(Path.cwd() / raw_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _resolve_rtmdet_assets(
    *,
    config_path: str | Path | None,
    checkpoint_path: str | Path | None,
    config_name: str | None,
    cache_dir: str | Path | None,
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

        candidate = cache_root / f"{variant}.py"
        if candidate.exists():
            cfg_path = candidate
        else:
            matches = sorted(cache_root.glob(f"**/{variant}.py"))
            if not matches:
                raise FileNotFoundError(
                    f"Could not find config '{variant}.py' under {cache_root}. "
                    "Run the bootstrap stage (or `mim download mmdet --config <name> --dest <cache_dir>`) "
                    "to provision pretrained RTMDet assets."
                )
            cfg_path = matches[-1]

    if ckpt_path is None:
        if not variant:
            raise ValueError(
                "MMDetection backend needs either checkpoint_path or config_name "
                "to locate a pretrained checkpoint."
            )

        candidates = sorted(cache_root.glob(f"**/{variant}*.pth"))
        if not candidates:
            raise FileNotFoundError(
                f"Could not find checkpoint '{variant}*.pth' under {cache_root}. "
                "Run the bootstrap stage (or `mim download mmdet --config <name> --dest <cache_dir>`) "
                "to provision pretrained RTMDet assets."
            )
        ckpt_path = max(candidates, key=lambda p: p.stat().st_mtime)

    resolved_variant = variant or cfg_path.stem
    return cfg_path, ckpt_path, resolved_variant


def _patch_pipeline_scales(node, image_size: int) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key in {"scale", "img_scale", "crop_size", "size"} and isinstance(value, (list, tuple)) and len(value) == 2:
                node[key] = (int(image_size), int(image_size))
            else:
                _patch_pipeline_scales(value, image_size)
        return
    if isinstance(node, list):
        for item in node:
            _patch_pipeline_scales(item, image_size)


def _fix_mosaic_pipeline_resize(node, image_size: int) -> None:
    """In pipeline lists that contain CachedMosaic, set RandomResize.scale to 2×image_size.

    CachedMosaic stitches four img_scale-sized tiles into a composite roughly
    2×img_scale.  The subsequent RandomResize must operate on that larger canvas
    (scale = 2×image_size) so the multi-scale augmentation range is correct
    relative to the final RandomCrop (which extracts an image_size patch).
    """
    if isinstance(node, list):
        has_mosaic = any(
            isinstance(item, dict) and "Mosaic" in str(item.get("type", ""))
            for item in node
        )
        if has_mosaic:
            for item in node:
                if isinstance(item, dict) and item.get("type") == "RandomResize":
                    if "scale" in item and isinstance(item["scale"], (list, tuple)) and len(item["scale"]) == 2:
                        item["scale"] = (int(image_size * 2), int(image_size * 2))
        for item in node:
            _fix_mosaic_pipeline_resize(item, image_size)
    elif isinstance(node, dict):
        for value in node.values():
            _fix_mosaic_pipeline_resize(value, image_size)


def _convert_syncbn_to_bn(node) -> None:
    """Replace SyncBN with BN for single-GPU training."""
    if isinstance(node, dict):
        if node.get("type") == "SyncBN":
            node["type"] = "BN"
        for value in node.values():
            _convert_syncbn_to_bn(value)
    elif isinstance(node, list):
        for item in node:
            _convert_syncbn_to_bn(item)


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

    config_path = _resolve_baseline_config_path(weights_path, metadata)
    if config_path is None:
        config_name = metadata.get("rtmdet_config_name")
        if not config_name:
            raise RuntimeError(
                "MMDetection baseline metadata must include model_config_path or rtmdet_config_name."
            )
        config_path, _unused_ckpt, _ = _resolve_rtmdet_assets(
            config_path=None,
            checkpoint_path=None,
            config_name=str(config_name),
            cache_dir=metadata.get("rtmdet_cache_dir"),
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
        config_path=None,
        checkpoint_path=None,
        config_name=resolved_cfg.get("rtmdet_config_name"),
        cache_dir=resolved_cfg.get("rtmdet_cache_dir"),
    )

    run_name = experiment_name or f"{resolved_cfg['model_key']}-rtmdet"
    runs_root = Path("runs")
    runs_root.mkdir(parents=True, exist_ok=True)
    output_dir = resolve_unique_run_dir(runs_root, run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(str(cfg_path))
    cfg.work_dir = str(output_dir)
    cfg.load_from = str(ckpt_path) if ckpt_path is not None else None
    cfg.resume = False
    cfg.train_cfg["max_epochs"] = int(resolved_cfg["epochs"])
    # The config's cosine schedule is hardcoded for 300-epoch COCO training
    # (cosine from epoch 150→300, eta_min=0.0002 at base_lr=0.004 = 5% of peak).
    # Rescale to cover the second half of our actual epoch count and keep the
    # same 5% eta_min ratio relative to our base_lr.
    epochs = int(resolved_cfg["epochs"])
    cosine_start = epochs // 2
    base_lr = float(resolved_cfg["rtmdet_lr"])
    for sched in cfg.get("param_scheduler", []):
        if isinstance(sched, dict) and sched.get("type") == "CosineAnnealingLR":
            sched["begin"] = cosine_start
            sched["end"] = epochs
            sched["T_max"] = epochs - cosine_start
            sched["eta_min"] = round(base_lr * 0.05, 8)
    # The default LinearLR warmup runs for 1000 iterations — calibrated for
    # COCO (118K images / BS 32 ≈ 3700 iters/epoch → ~0.27 epochs).  For
    # smaller datasets that would extend warmup over many epochs, cap it
    # at one epoch's worth of iterations (minimum 100).
    batch_size = int(resolved_cfg["batch_size"])
    train_ann_path = dataset_dir / "annotations" / "instances_train.json"
    with open(train_ann_path, encoding="utf-8") as f:
        num_train_images = len(json.load(f).get("images", []))
    iters_per_epoch = max(1, num_train_images // batch_size)
    warmup_iters = max(100, min(1000, iters_per_epoch))
    for sched in cfg.get("param_scheduler", []):
        if isinstance(sched, dict) and sched.get("type") == "LinearLR":
            sched["end"] = warmup_iters
    # The config's PipelineSwitchHook (switch_epoch=280) drops mosaic/mixup
    # for the final 20 epochs of the default 300-epoch schedule.  Rescale to
    # the final ~7% of our actual epoch count (min 1 to always fire).
    # Similarly, dynamic_intervals switches val from every-10 to every-1 at
    # the same epoch; rescale it to match.
    stage2_start = max(1, int(epochs * 280 / 300))
    for hook in cfg.get("custom_hooks", []):
        if isinstance(hook, dict) and hook.get("type") == "PipelineSwitchHook":
            hook["switch_epoch"] = stage2_start
    if "dynamic_intervals" in cfg.train_cfg:
        cfg.train_cfg["dynamic_intervals"] = [(stage2_start, 1)]
    # Validate more often than the config's default val_interval=10 so that
    # save_best has finer granularity.  ~20 validation runs per training.
    cfg.train_cfg["val_interval"] = max(1, epochs // 20)

    cfg.randomness = {"seed": int(resolved_cfg.get("seed", 42)), "deterministic": True}
    cfg.default_hooks.setdefault("checkpoint", {})
    cfg.default_hooks["checkpoint"]["save_best"] = "coco/bbox_mAP"
    cfg.default_hooks["checkpoint"]["rule"] = "greater"
    cfg.default_hooks["checkpoint"]["max_keep_ckpts"] = 1

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
    cfg.optim_wrapper["accumulative_counts"] = int(resolved_cfg["rtmdet_accum"])
    # Preserve negative/background images (no annotations) in training.
    # The default config drops them (filter_empty_gt=True) which is fine for
    # COCO pre-training but can hurt fine-tuning when the dataset includes
    # intentional hard-negative examples.
    train_dataset = cfg.train_dataloader["dataset"]
    if "dataset" in train_dataset and isinstance(train_dataset["dataset"], dict):
        train_dataset = train_dataset["dataset"]
    if "filter_cfg" in train_dataset:
        train_dataset["filter_cfg"]["filter_empty_gt"] = False
    # Disable persistent workers so dataloader workers exit cleanly at the end
    # of each epoch (and after training) rather than staying alive until the
    # Runner is garbage-collected.  persistent_workers=True causes a race
    # between worker teardown and the QueueFeederThread, which produces
    # spurious "Bad file descriptor" / semaphore-over-release warnings.
    # Validation runs with num_workers=0 (same pattern as evaluate_stage).
    train_num_workers = min(4, os.cpu_count() or 2)
    cfg.train_dataloader["num_workers"] = train_num_workers
    cfg.train_dataloader["persistent_workers"] = False
    cfg.val_dataloader["num_workers"] = 0
    cfg.val_dataloader["persistent_workers"] = False
    if "test_dataloader" in cfg:
        cfg.test_dataloader["num_workers"] = 0
        cfg.test_dataloader["persistent_workers"] = False
    _set_num_classes(cfg["model"], len(classes_tuple))
    _convert_syncbn_to_bn(cfg["model"])
    _patch_pipeline_scales(cfg, int(resolved_cfg["image_size"]))
    _fix_mosaic_pipeline_resize(cfg, int(resolved_cfg["image_size"]))

    opt_wrapper = cfg.get("optim_wrapper")
    if isinstance(opt_wrapper, dict):
        optimizer = opt_wrapper.get("optimizer")
        if isinstance(optimizer, dict) and "lr" in optimizer:
            optimizer["lr"] = float(resolved_cfg["rtmdet_lr"])

    runner = Runner.from_cfg(cfg)
    # mmengine 0.10.x predates the PyTorch 2.6 weights_only=True default and
    # calls torch.load without that argument.  Override for the training call.
    _orig_load, torch.load = torch.load, lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})
    try:
        runner.train()
    finally:
        torch.load = _orig_load
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


__all__ = [
    "train_backend",
    "train_rtmdet_backend",
    "load_rtmdet_baseline",
]
