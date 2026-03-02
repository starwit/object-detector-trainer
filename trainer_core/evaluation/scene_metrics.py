from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


class SceneMetricsError(RuntimeError):
    """Raised when scene-metric inputs or execution are invalid."""


def _load_dataset_config_for_scenes(data: str) -> dict:
    try:
        with open(data, "r", encoding="utf-8") as handle:
            parsed = yaml.safe_load(handle) or {}
    except FileNotFoundError as exc:
        raise SceneMetricsError(f"Dataset YAML not found at {data}") from exc
    except (OSError, yaml.YAMLError) as exc:
        raise SceneMetricsError(f"Error reading dataset YAML {data}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SceneMetricsError(
            f"Invalid dataset YAML {data}: expected mapping, got {type(parsed)}"
        )
    return parsed


def _collect_scene_images(val_images_dir: Path, val_labels_dir: Path) -> dict:
    scene_images = {}
    for img_path in val_images_dir.glob("*"):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        if "__scene_" not in img_path.name:
            continue
        parts = img_path.stem.split("__scene_")
        if len(parts) <= 1:
            continue
        scene_name = parts[1]
        scene_images.setdefault(scene_name, []).append(
            (img_path, val_labels_dir / img_path.with_suffix(".txt").name)
        )
    return scene_images


def _copy_scene_to_temp(scene_name: str, images_labels: list) -> tuple[Path | None, int]:
    temp_path = Path(tempfile.mkdtemp())
    temp_images_dir = temp_path / "images"
    temp_labels_dir = temp_path / "labels"
    temp_images_dir.mkdir(parents=True, exist_ok=True)
    temp_labels_dir.mkdir(parents=True, exist_ok=True)

    copied_files = 0
    for img_path, label_path in images_labels:
        if not img_path.exists() or not label_path.exists():
            logger.warning(
                "Source file missing during copy for scene '%s': %s or %s",
                scene_name,
                img_path,
                label_path,
            )
            continue
        new_img_name = img_path.name.replace(f"__scene_{scene_name}", "")
        new_label_name = label_path.name.replace(f"__scene_{scene_name}", "")
        try:
            shutil.copy(img_path, temp_images_dir / new_img_name)
            shutil.copy(label_path, temp_labels_dir / new_label_name)
            copied_files += 1
        except OSError as exc:
            logger.warning(
                "Error copying %s or %s to %s: %s",
                img_path,
                label_path,
                temp_path,
                exc,
            )
    return (temp_path if copied_files > 0 else None), copied_files


def _write_scene_yaml(temp_path: Path, dataset_config: dict, scene_name: str) -> Path:
    temp_yaml_path = temp_path / f"scene_{scene_name}.yaml"
    scene_config = {
        "path": str(temp_path),
        "train": "images",
        "val": "images",
        "nc": len(dataset_config.get("names", [])),
        "names": dataset_config.get("names", {}),
    }
    try:
        with temp_yaml_path.open("w", encoding="utf-8") as handle:
            yaml.dump(scene_config, handle)
    except OSError as exc:
        raise SceneMetricsError(f"Error writing temporary YAML {temp_yaml_path}: {exc}") from exc
    return temp_yaml_path


def _validate_scene(model, temp_yaml_path: Path, class_ids: list | None, kwargs: dict) -> float:
    eval_kwargs = {k: v for k, v in kwargs.items() if k in ["imgsz", "batch", "conf", "iou"]}
    eval_kwargs["workers"] = 0
    eval_kwargs["batch"] = 1
    scene_results = model.val(
        data=str(temp_yaml_path),
        classes=class_ids if class_ids else None,
        verbose=False,
        save=False,
        plots=False,
        **eval_kwargs,
    )
    return float(getattr(scene_results, "fitness", 0.0))


def calculate_scene_metrics(model, data, **kwargs):
    """Calculate per-scene fitness for test datasets with `__scene_` suffixes."""
    dataset_config = _load_dataset_config_for_scenes(data)

    if "path" not in dataset_config or "val" not in dataset_config:
        raise SceneMetricsError(f"Dataset YAML {data} is missing 'path' or 'val' key.")

    dataset_path = Path(dataset_config["path"])
    val_images_dir = dataset_path / dataset_config["val"]
    val_labels_dir = val_images_dir.parent / "labels"

    from trainer_core.evaluation import validate as validate_core

    _, class_ids = validate_core.get_dataset_classes(data)
    scene_images = _collect_scene_images(val_images_dir, val_labels_dir)
    scene_metrics = {}

    for scene_name, images_labels in scene_images.items():
        temp_path = None
        try:
            temp_path, copied_files = _copy_scene_to_temp(scene_name, images_labels)
            if not temp_path or copied_files == 0:
                logger.warning("No files were copied for scene '%s'. Skipping validation.", scene_name)
                continue

            temp_yaml_path = _write_scene_yaml(temp_path, dataset_config, scene_name)

            fitness = _validate_scene(model, temp_yaml_path, class_ids, kwargs)
            scene_metrics[f"scene_{scene_name}_fitness"] = fitness
        except (OSError, RuntimeError, SceneMetricsError) as exc:
            logger.warning("Error evaluating scene '%s' using data '%s': %s", scene_name, data, exc)
            scene_metrics[f"scene_{scene_name}_fitness"] = 0.0
        finally:
            if temp_path and temp_path.exists():
                shutil.rmtree(temp_path, ignore_errors=True)

    return scene_metrics


__all__ = ["SceneMetricsError", "calculate_scene_metrics"]
