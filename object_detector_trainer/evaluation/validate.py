from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from object_detector_trainer.datasets.yolo_yaml import get_dataset_classes as _get_dataset_classes
from object_detector_trainer.evaluation import scene_metrics
from object_detector_trainer.evaluation.reports import (
    append_results_to_csv,
    create_formatted_table,
    write_merged_class_results,
)

logger = logging.getLogger(__name__)


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def evaluate_and_log_model_results(
    model,
    model_name,
    test_path,
    image_size,
    output_dir,
    val_split,
    train_epochs=0,
    is_original=False,
    metrics_json_path: Path | None = None,
):
    """
    Evaluate one model and append structured outputs.

    Returns:
        tuple: (metadata, metrics_dict)
    """
    dataset_yaml_path = test_path / "dataset.yaml"
    class_names, class_ids = get_dataset_classes(dataset_yaml_path)

    results = validate_model(
        model,
        data=str(dataset_yaml_path),
        class_ids=class_ids,
        imgsz=image_size,
        workers=0,
        write_json=metrics_json_path is not None,
        metrics_json_path=metrics_json_path,
    )

    model_backend = getattr(model, "model_backend", None)
    if not model_backend:
        raise ValueError(f"Model {model_name!r} does not declare model_backend.")

    metadata = {
        "experiment_name": model_name,
        "split_parameters": {
            "val_split": val_split,
        },
        "num_epochs": train_epochs,
        "model_size": model.model_name if hasattr(model, "model_name") else "Unknown",
        "model_backend": str(model_backend),
        "image_size": image_size,
    }
    model_variant = getattr(model, "model_variant", None)
    if model_variant:
        metadata["model_variant"] = str(model_variant)
    model_config_path = getattr(model, "model_config_path", None)
    if model_config_path:
        metadata["model_config_path"] = str(model_config_path)
    rtmdet_config_name = getattr(model, "rtmdet_config_name", None)
    if rtmdet_config_name:
        metadata["rtmdet_config_name"] = str(rtmdet_config_name)
    rtmdet_cache_dir = getattr(model, "rtmdet_cache_dir", None)
    if rtmdet_cache_dir:
        metadata["rtmdet_cache_dir"] = str(rtmdet_cache_dir)
    rtmdet_allow_download = getattr(model, "rtmdet_allow_download", None)
    if rtmdet_allow_download is not None:
        metadata["rtmdet_allow_download"] = bool(rtmdet_allow_download)
    class_names_meta = getattr(model, "class_names", None)
    if isinstance(class_names_meta, dict) and class_names_meta:
        metadata["class_names"] = {int(k): str(v) for k, v in class_names_meta.items()}

    append_results_to_csv(output_dir, results, metadata, is_original)

    return metadata, results


def get_dataset_classes(dataset_yaml_path):
    """Return (class_id_to_name, class_id_list) from dataset.yaml."""
    return _get_dataset_classes(dataset_yaml_path)


def _extract_per_class_metrics(metrics, data):
    """Extract per-class metrics from model validation results."""
    names, _ = _get_dataset_classes(data)

    per_class = {}

    box = getattr(metrics, "box", None)
    if box is not None and hasattr(box, "ap_class_index"):
        ap_class_idx = box.ap_class_index
        if hasattr(ap_class_idx, "__len__") and len(ap_class_idx) > 0:
            ap50_vals = box.ap50
            ap_vals = box.ap
            for i, cls_idx in enumerate(ap_class_idx):
                cls_id = int(cls_idx)
                cls_name = names.get(cls_id, f"class_{cls_id}")
                p = float(box.p[i])
                r = float(box.r[i])
                a50 = float(ap50_vals[i])
                ap = float(ap_vals[i])
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                per_class[cls_name] = {
                    "precision": p,
                    "recall": r,
                    "map50": a50,
                    "map": ap,
                    "f1_score": f1,
                }
        return per_class

    native_per_class = getattr(metrics, "per_class", None)
    if native_per_class:
        return dict(native_per_class)

    return {}


def validate_model(model, data, class_ids=None, write_json=False, metrics_json_path=None, **kwargs):
    """Validate model and return normalized metrics payload."""
    validation_kwargs = kwargs.copy()
    if class_ids is not None:
        validation_kwargs["classes"] = class_ids
    validation_kwargs.setdefault("batch", 1)

    metrics = model.val(
        data=data,
        verbose=False,
        save=False,
        plots=False,
        **validation_kwargs,
    )

    spd = metrics.speed
    ms_per_frame = (
        float(spd.get("preprocess", 0.0))
        + float(spd.get("inference", 0.0))
        + float(spd.get("postprocess", 0.0))
    )

    precision = float(metrics.results_dict["metrics/precision(B)"])
    recall = float(metrics.results_dict["metrics/recall(B)"])
    map50 = float(metrics.results_dict["metrics/mAP50(B)"])
    map50_95 = float(metrics.results_dict["metrics/mAP50-95(B)"])
    fitness = float(metrics.fitness)

    f1_from_model = _safe_float(metrics.results_dict.get("metrics/f1(B)"))
    if f1_from_model is not None:
        f1_score = f1_from_model
    else:
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

    effective_imgsz = getattr(model, "resolution", kwargs.get("imgsz"))

    metrics_dict = {
        "img_size": int(effective_imgsz) if effective_imgsz is not None else None,
        "precision": precision,
        "recall": recall,
        "map": map50_95,
        "map50": map50,
        "fitness": fitness,
        "f1_score": f1_score,
        "ms_per_frame": ms_per_frame,
    }

    per_class = _extract_per_class_metrics(metrics, data)
    if per_class:
        metrics_dict["per_class"] = per_class

    try:
        scene_metrics_dict = scene_metrics.calculate_scene_metrics(model, data, **kwargs)
    except scene_metrics.SceneMetricsError as exc:
        logger.warning("Skipping scene metrics for %s: %s", data, exc)
    else:
        metrics_dict.update(scene_metrics_dict)

    if write_json:
        output_path = Path(metrics_json_path) if metrics_json_path is not None else Path("metrics.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, indent=4)

    return metrics_dict


__all__ = [
    "append_results_to_csv",
    "create_formatted_table",
    "evaluate_and_log_model_results",
    "get_dataset_classes",
    "validate_model",
    "write_merged_class_results",
]
