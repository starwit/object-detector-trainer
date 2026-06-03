from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import cv2
import torch

from object_detector_trainer.backends.assets import (
    is_ready_file,
    require_asset_id,
    require_bootstrapped_file,
    resolve_cache_dir,
)
from object_detector_trainer.datasets.yolo_yaml import get_dataset_classes
from object_detector_trainer.utils.path_ops import (
    resolve_unique_run_dir,
    resolve_workspace_path,
    safe_dataset_dirname,
)


def resolve_config(
    *,
    model_key: str,
    model_cfg: Mapping[str, Any],
    shared_image_size: int,
    shared_epochs: int,
    shared_batch_size: int,
    finetune_enabled: bool,
    finetune_epochs: int | None,
) -> dict[str, Any]:
    config_name = require_asset_id(model_key=model_key, model_cfg=model_cfg)
    cache_dir = resolve_cache_dir(backend="rtmdet", model_cfg=model_cfg)

    batch_size = int(model_cfg.get("batch_size", shared_batch_size))
    explicit_grad_accum = model_cfg.get("grad_accum_steps")
    target_effective_batch = int(model_cfg.get("target_effective_batch", 32))
    accum = (
        int(explicit_grad_accum)
        if explicit_grad_accum is not None
        else max(1, target_effective_batch // batch_size)
    )

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


def bootstrap_assets(model_key: str, model_cfg: Mapping[str, Any]) -> Path:
    config_name = require_asset_id(model_key=model_key, model_cfg=model_cfg)
    cache_dir = resolve_cache_dir(backend="rtmdet", model_cfg=model_cfg)

    resolved_cfg = cache_dir / f"{config_name}.py"
    resolved_ckpt = _find_cached_rtmdet_checkpoint(cache_dir, config_name)
    if is_ready_file(resolved_cfg) and is_ready_file(resolved_ckpt):
        return resolved_cfg

    if not bool(model_cfg.get("allow_download", True)):
        raise FileNotFoundError(
            f"models.{model_key} RTMDet assets are missing under {cache_dir} "
            f"for config '{config_name}'."
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "mim",
            "download",
            "mmdet",
            "--config",
            config_name,
            "--dest",
            str(cache_dir),
        ],
        cwd=Path.cwd(),
        check=True,
    )
    resolved_cfg, resolved_ckpt, _ = _resolve_rtmdet_assets(
        config_name=str(config_name),
        cache_dir=cache_dir,
    )
    require_bootstrapped_file(resolved_cfg, label=f"models.{model_key}.config")
    require_bootstrapped_file(resolved_ckpt, label=f"models.{model_key}.checkpoint")
    return resolved_cfg


def build_reload_metadata(model: object, resolved_cfg: Mapping[str, Any]) -> dict[str, object]:
    config_name = str(
        getattr(model, "rtmdet_config_name", None) or resolved_cfg["rtmdet_config_name"]
    )
    metadata: dict[str, object] = {
        "model_variant": str(getattr(model, "model_variant", None) or config_name),
        "rtmdet_config_name": config_name,
        "rtmdet_cache_dir": str(
            getattr(model, "rtmdet_cache_dir", None) or resolved_cfg["rtmdet_cache_dir"]
        ),
    }
    model_config_path = getattr(model, "model_config_path", None)
    if model_config_path:
        metadata["model_config_path"] = str(Path("..") / Path(str(model_config_path)).name)
    return metadata


def load_model_from_weights(
    candidate_path: Path,
    meta: Mapping[str, object],
    display_name: str,
    yolo_loader=None,
) -> object:
    return load_rtmdet_baseline(
        weights_path=candidate_path,
        metadata=dict(meta),
        display_name=str(display_name),
    )


def _iter_image_files(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        raise FileNotFoundError(f"Dataset images directory not found: {images_dir}")
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
            raise FileNotFoundError(f"Could not read dataset image: {image_path}")
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
                    if not parts:
                        continue
                    if len(parts) < 5:
                        raise ValueError(
                            f"Malformed YOLO label line in {label_path}: {line.strip()!r}"
                        )
                    cls_id = int(float(parts[0]))
                    cx = float(parts[1])
                    cy = float(parts[2])
                    bw = float(parts[3])
                    bh = float(parts[4])
                    if cls_id not in cat_ids:
                        raise ValueError(
                            f"Label {label_path} references class id {cls_id}, "
                            "which is not present in dataset.yaml."
                        )

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


def _prepare_rtmdet_coco_layout(
    training_path: Path,
    dataset_name: str,
) -> tuple[Path, dict[int, str]]:
    base_dir = Path(".tmp") / "rtmdet_datasets"
    output_dir = base_dir / safe_dataset_dirname(str(dataset_name))

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


def _resolve_baseline_config_path(weights_path: Path, metadata: Mapping[str, object]) -> Path:
    configured_path = metadata.get("model_config_path")
    if not configured_path:
        raise ValueError("RTMDet metadata must include model_config_path.")

    raw_path = Path(str(configured_path)).expanduser()
    config_path = raw_path if raw_path.is_absolute() else weights_path.parent / raw_path
    if not is_ready_file(config_path):
        raise FileNotFoundError(
            f"RTMDet metadata model_config_path does not point to a non-empty file: {configured_path}"
        )
    return config_path


def _find_cached_rtmdet_checkpoint(cache_root: Path, config_name: str) -> Path | None:
    matches = sorted(path for path in cache_root.glob(f"{config_name}*.pth") if is_ready_file(path))
    return max(matches, key=lambda p: p.stat().st_mtime) if matches else None


def _resolve_rtmdet_assets(
    *,
    config_name: str,
    cache_dir: str | Path | None,
) -> tuple[Path, Path, str]:
    cache_root = resolve_workspace_path(cache_dir) or (
        Path.cwd() / "models" / "pretrained" / "rtmdet"
    )
    variant = str(config_name).strip()
    if not variant:
        raise ValueError("RTMDet backend requires models.<key>.asset_id.")

    cfg_path = cache_root / f"{variant}.py"
    if not is_ready_file(cfg_path):
        raise FileNotFoundError(
            f"Could not find RTMDet config '{variant}.py' under {cache_root}. "
            "Run the bootstrap stage first."
        )

    ckpt_path = _find_cached_rtmdet_checkpoint(cache_root, variant)
    if ckpt_path is None:
        raise FileNotFoundError(
            f"Could not find RTMDet checkpoint '{variant}*.pth' under {cache_root}. "
            "Run the bootstrap stage first."
        )

    return cfg_path, ckpt_path, variant


def _find_best_checkpoint(run_dir: Path) -> Path:
    best_candidates = sorted(run_dir.glob("best*.pth"))
    if not best_candidates:
        raise FileNotFoundError(f"No best*.pth checkpoint found in {run_dir}")
    return max(best_candidates, key=lambda p: p.stat().st_mtime)


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


def _require_rtmdet_runtime() -> None:
    try:
        import mmcv._ext  # type: ignore  # noqa: F401
    except (ImportError, ModuleNotFoundError, OSError) as e:
        raise RuntimeError(
            "RTMDet backend requires full mmcv ops. Install `mmcv` (not `mmcv-lite`) "
            "matching your PyTorch/CUDA build."
        ) from e


def load_rtmdet_baseline(
    *,
    weights_path: Path,
    metadata: dict,
    display_name: str,
):
    _require_rtmdet_runtime()

    config_path = _resolve_baseline_config_path(weights_path, metadata)

    from mmdet.apis import init_detector

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
        resolution=int(metadata["image_size"]),
        class_names=class_names,
        model_variant=str(metadata.get("model_variant", "")) or None,
        model_config_path=str(config_path),
    )


def train_backend(
    *,
    training_path: Path,
    test_path: Path,
    dataset_name: str,
    resolved_cfg: dict,
    experiment_name: str | None,
) -> tuple[object, Path, str, int, int]:
    _require_rtmdet_runtime()
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmdet.apis import init_detector

    from object_detector_trainer.wrappers.rtmdet import RTMDetModelAdapter

    dataset_dir, class_names = _prepare_rtmdet_coco_layout(training_path, str(dataset_name))
    classes_tuple = tuple(class_names[i] for i in sorted(class_names))

    cfg_path, ckpt_path, variant = _resolve_rtmdet_assets(
        config_name=str(resolved_cfg["rtmdet_config_name"]),
        cache_dir=resolved_cfg.get("rtmdet_cache_dir"),
    )

    run_name = experiment_name or f"{resolved_cfg['model_key']}-rtmdet"
    runs_root = Path(str(resolved_cfg.get("runs_root", "runs")))
    runs_root.mkdir(parents=True, exist_ok=True)
    output_dir = resolve_unique_run_dir(runs_root, run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config.fromfile(str(cfg_path))
    cfg.work_dir = str(output_dir)
    cfg.load_from = str(ckpt_path)
    cfg.resume = False
    cfg.train_cfg["max_epochs"] = int(resolved_cfg["epochs"])
    cfg.train_cfg["val_interval"] = 1
    cfg.param_scheduler = []
    cfg.randomness = {"seed": int(resolved_cfg.get("seed", 42)), "deterministic": True}
    cfg.default_hooks.setdefault("checkpoint", {})
    cfg.default_hooks["checkpoint"]["save_best"] = "coco/bbox_mAP"
    cfg.default_hooks["checkpoint"]["rule"] = "greater"
    cfg.default_hooks["checkpoint"]["max_keep_ckpts"] = 1
    cfg.default_hooks["checkpoint"]["interval"] = 1

    train_dataset = cfg.train_dataloader["dataset"]
    val_dataset = cfg.val_dataloader["dataset"]
    if not isinstance(train_dataset, dict) or not isinstance(val_dataset, dict):
        raise ValueError("RTMDet config must define train/val dataloader datasets as mappings.")
    if "dataset" in train_dataset or "dataset" in val_dataset:
        raise ValueError("RTMDet backend supports the stock direct CocoDataset config shape.")

    train_dataset["data_root"] = str(dataset_dir)
    train_dataset["ann_file"] = "annotations/instances_train.json"
    train_dataset["data_prefix"] = {"img": "train/images/"}
    train_dataset["metainfo"] = {"classes": classes_tuple}
    if "filter_cfg" in train_dataset:
        train_dataset["filter_cfg"]["filter_empty_gt"] = False

    val_dataset["data_root"] = str(dataset_dir)
    val_dataset["ann_file"] = "annotations/instances_val.json"
    val_dataset["data_prefix"] = {"img": "val/images/"}
    val_dataset["metainfo"] = {"classes": classes_tuple}

    if "test_dataloader" in cfg and "dataset" in cfg.test_dataloader:
        test_dataset = cfg.test_dataloader["dataset"]
        if not isinstance(test_dataset, dict) or "dataset" in test_dataset:
            raise ValueError("RTMDet config must define test dataloader dataset as a mapping.")
        test_dataset["data_root"] = str(dataset_dir)
        test_dataset["ann_file"] = "annotations/instances_val.json"
        test_dataset["data_prefix"] = {"img": "val/images/"}
        test_dataset["metainfo"] = {"classes": classes_tuple}

    evaluator_ann_file = str((dataset_dir / "annotations" / "instances_val.json").resolve())
    if not isinstance(cfg.val_evaluator, dict):
        raise ValueError("RTMDet config must define val_evaluator as a mapping.")
    cfg.val_evaluator["ann_file"] = evaluator_ann_file
    if "test_evaluator" in cfg:
        if not isinstance(cfg.test_evaluator, dict):
            raise ValueError("RTMDet config must define test_evaluator as a mapping.")
        cfg.test_evaluator["ann_file"] = evaluator_ann_file

    cfg.train_dataloader["batch_size"] = int(resolved_cfg["batch_size"])
    cfg.optim_wrapper["accumulative_counts"] = int(resolved_cfg["rtmdet_accum"])
    cfg.train_dataloader["num_workers"] = 0
    cfg.train_dataloader["persistent_workers"] = False
    cfg.val_dataloader["num_workers"] = 0
    cfg.val_dataloader["persistent_workers"] = False
    if "test_dataloader" in cfg:
        cfg.test_dataloader["num_workers"] = 0
        cfg.test_dataloader["persistent_workers"] = False

    bbox_head = cfg.model["bbox_head"]
    if not isinstance(bbox_head, dict) or "num_classes" not in bbox_head:
        raise ValueError("RTMDet config must define model.bbox_head.num_classes.")
    bbox_head["num_classes"] = len(classes_tuple)

    for section_name in ("backbone", "neck", "bbox_head"):
        section = cfg.model.get(section_name)
        if isinstance(section, dict) and isinstance(section.get("norm_cfg"), dict):
            if section["norm_cfg"].get("type") == "SyncBN":
                section["norm_cfg"]["type"] = "BN"

    image_size = int(resolved_cfg["image_size"])
    pipelines = [
        ("train_dataloader.dataset.pipeline", train_dataset["pipeline"]),
        ("val_dataloader.dataset.pipeline", val_dataset["pipeline"]),
    ]
    if "test_dataloader" in cfg and "dataset" in cfg.test_dataloader:
        pipelines.append(("test_dataloader.dataset.pipeline", cfg.test_dataloader["dataset"]["pipeline"]))
    for hook in cfg.get("custom_hooks", []):
        if isinstance(hook, dict) and hook.get("type") == "PipelineSwitchHook":
            pipelines.append(("PipelineSwitchHook.switch_pipeline", hook["switch_pipeline"]))

    for pipeline_name, pipeline in pipelines:
        if not isinstance(pipeline, list):
            raise ValueError(f"RTMDet config {pipeline_name} must be a list.")
        uses_mosaic = any(
            isinstance(step, dict) and "Mosaic" in str(step.get("type", ""))
            for step in pipeline
        )
        for step in pipeline:
            if not isinstance(step, dict):
                raise ValueError(f"RTMDet config {pipeline_name} contains a non-mapping step.")
            step_type = str(step.get("type", ""))
            if step_type in {"CachedMosaic", "CachedMixUp"} and "img_scale" in step:
                step["img_scale"] = (image_size, image_size)
            elif step_type in {"RandomResize", "Resize"} and "scale" in step:
                scale = image_size * 2 if step_type == "RandomResize" and uses_mosaic else image_size
                step["scale"] = (scale, scale)
            elif step_type == "RandomCrop" and "crop_size" in step:
                step["crop_size"] = (image_size, image_size)
            elif step_type == "Pad" and "size" in step:
                step["size"] = (image_size, image_size)

    opt_wrapper = cfg.get("optim_wrapper")
    if isinstance(opt_wrapper, dict):
        optimizer = opt_wrapper.get("optimizer")
        if isinstance(optimizer, dict) and "lr" in optimizer:
            optimizer["lr"] = float(resolved_cfg["rtmdet_lr"])

    runner = Runner.from_cfg(cfg)
    # mmengine 0.10.x predates the PyTorch 2.6 weights_only=True default and
    # calls torch.load without that argument.  Override for the training call.
    _orig_load = torch.load
    torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})
    try:
        runner.train()
    finally:
        torch.load = _orig_load
    del runner

    best_weights = _save_rtmdet_weights(output_dir, _find_best_checkpoint(output_dir))

    local_config = output_dir / "model_config.py"
    cfg.dump(str(local_config))

    display_name = f"{run_name}-rtmdet"
    class_names_map, _ = get_dataset_classes(test_path / "dataset.yaml")
    device = str(
        resolved_cfg.get("rtmdet_device")
        or ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    detector = init_detector(str(local_config), str(best_weights), device=device)
    model = RTMDetModelAdapter(
        detector,
        model_name=display_name,
        resolution=int(resolved_cfg["image_size"]),
        class_names=class_names_map,
        model_variant=variant,
        model_config_path="../model_config.py",
        config_name=variant,
        cache_dir=str(resolved_cfg.get("rtmdet_cache_dir") or "models/pretrained/rtmdet"),
    )

    if bool(resolved_cfg.get("rtmdet_cleanup_tmp", False)):
        shutil.rmtree(dataset_dir, ignore_errors=True)

    return (
        model,
        output_dir,
        display_name,
        int(resolved_cfg["image_size"]),
        int(resolved_cfg["epochs"]),
    )


__all__ = [
    "bootstrap_assets",
    "build_reload_metadata",
    "load_model_from_weights",
    "resolve_config",
    "train_backend",
    "load_rtmdet_baseline",
]
