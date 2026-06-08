from __future__ import annotations

import json
from pathlib import Path

from object_detector_trainer.datasets.yolo_yaml import get_dataset_classes, load_yolo_dataset_yaml
from object_detector_trainer.evaluation import scene_metrics
from object_detector_trainer.evaluation.reports import (
    append_results_to_csv,
    create_formatted_table,
    write_merged_class_results,
)


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
    _, class_ids = get_dataset_classes(dataset_yaml_path)

    results = validate_model(
        model,
        data=str(dataset_yaml_path),
        class_ids=class_ids,
        imgsz=image_size,
        workers=0,
        write_json=metrics_json_path is not None,
        metrics_json_path=metrics_json_path,
    )

    metadata = {
        "experiment_name": model_name,
        "split_parameters": {
            "val_split": val_split,
        },
        "num_epochs": train_epochs,
        "model_size": str(model.model_name),
        "model_backend": str(model.model_backend),
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
    class_names_meta = getattr(model, "class_names", None)
    if isinstance(class_names_meta, dict) and class_names_meta:
        metadata["class_names"] = {int(k): str(v) for k, v in class_names_meta.items()}

    append_results_to_csv(output_dir, results, metadata, is_original)

    return metadata, results


_ZERO_CLASS_METRICS = {
    "precision": 0.0,
    "recall": 0.0,
    "map50": 0.0,
    "map": 0.0,
    "f1_score": 0.0,
}


def _present_class_names(data, names: dict[int, str]) -> set[str]:
    dataset_yaml = load_yolo_dataset_yaml(data)
    dataset_root = Path(dataset_yaml.get("path") or Path(data).parent)
    if not dataset_root.is_absolute():
        dataset_root = Path(data).parent / dataset_root

    val_path = Path(dataset_yaml.get("val", "val/images"))
    if not val_path.is_absolute():
        val_path = dataset_root / val_path
    label_dir = val_path.parent / "labels" if val_path.name == "images" else val_path

    present: set[str] = set()
    for label_file in sorted(label_dir.glob("*.txt")):
        with label_file.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls_id = int(parts[0])
                    if cls_id in names:
                        present.add(names[cls_id])
    return present


def _normalize_per_class_metrics(
    per_class: dict[str, dict[str, float]],
    present_class_names: set[str],
) -> dict[str, dict[str, float]]:
    if not present_class_names:
        return {}
    normalized = {
        class_name: dict(per_class[class_name])
        for class_name in sorted(present_class_names)
        if class_name in per_class
    }
    for class_name in sorted(present_class_names - set(normalized)):
        normalized[class_name] = dict(_ZERO_CLASS_METRICS)
    return normalized


def _extract_per_class_metrics(metrics, data):
    """Extract per-class metrics for classes present in the evaluated labels."""
    names, _ = get_dataset_classes(data)
    present_class_names = _present_class_names(data, names)

    per_class = {}

    box = getattr(metrics, "box", None)
    if box is not None and hasattr(box, "ap_class_index"):
        ap_class_idx = box.ap_class_index
        if hasattr(ap_class_idx, "__len__") and len(ap_class_idx) > 0:
            ap50_vals = box.ap50
            ap_vals = box.ap
            for i, cls_idx in enumerate(ap_class_idx):
                cls_id = int(cls_idx)
                cls_name = names[cls_id]
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
        return _normalize_per_class_metrics(per_class, present_class_names)

    native_per_class = getattr(metrics, "per_class", None)
    if native_per_class:
        return _normalize_per_class_metrics(dict(native_per_class), present_class_names)

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
        float(spd["preprocess"])
        + float(spd["inference"])
        + float(spd["postprocess"])
    )

    precision = float(metrics.results_dict["metrics/precision(B)"])
    recall = float(metrics.results_dict["metrics/recall(B)"])
    map50 = float(metrics.results_dict["metrics/mAP50(B)"])
    map50_95 = float(metrics.results_dict["metrics/mAP50-95(B)"])
    fitness = float(metrics.fitness)

    if "metrics/f1(B)" in metrics.results_dict:
        f1_score = float(metrics.results_dict["metrics/f1(B)"])
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

    scene_metrics_dict = scene_metrics.calculate_scene_metrics(model, data, **kwargs)
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
    "validate_model",
    "write_merged_class_results",
]
