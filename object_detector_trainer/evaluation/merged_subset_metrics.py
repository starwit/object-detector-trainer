from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

import yaml

from object_detector_trainer.datasets.yolo_yaml import get_dataset_classes

logger = logging.getLogger(__name__)


class MergedSubsetEvaluationError(RuntimeError):
    """Raised when merged-subset evaluation inputs are invalid."""


def evaluate_merged_class_subsets(
    model,
    model_name,
    test_path,
    raw_test_path,
    class_mapping_config,
    custom_classes,
    **kwargs,
):
    """Evaluate model on subsets containing classes that were merged during training."""
    if not class_mapping_config or not custom_classes:
        return {}
    if not raw_test_path:
        raise MergedSubsetEvaluationError("raw_test_path must be provided for merged-class evaluation.")

    raw_test_path = Path(raw_test_path)
    if not raw_test_path.exists():
        raise MergedSubsetEvaluationError(
            f"Merged-class raw test path does not exist: {raw_test_path}"
        )

    merged_sources = {}
    for target, sources in class_mapping_config.items():
        if isinstance(sources, str):
            sources = [sources]
        for src in sources:
            if src != target:
                merged_sources[src] = target

    if not merged_sources:
        return {}

    original_class_to_id = {cls: i for i, cls in enumerate(custom_classes)}
    test_images_dir = Path(test_path) / "val" / "images"
    test_labels_dir = Path(test_path) / "val" / "labels"
    dataset_yaml = Path(test_path) / "dataset.yaml"

    if not dataset_yaml.exists():
        raise MergedSubsetEvaluationError(f"Prepared dataset YAML not found: {dataset_yaml}")
    if not test_images_dir.exists():
        raise MergedSubsetEvaluationError(
            f"Prepared test images directory not found: {test_images_dir}"
        )

    try:
        with dataset_yaml.open("r", encoding="utf-8") as handle:
            ds_config = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise MergedSubsetEvaluationError(f"Failed to read dataset YAML {dataset_yaml}: {exc}") from exc
    if not isinstance(ds_config, dict):
        raise MergedSubsetEvaluationError(
            f"Invalid dataset YAML {dataset_yaml}: expected mapping, got {type(ds_config)}"
        )

    results: dict[str, dict] = {}

    for src_class, target_class in merged_sources.items():
        src_class_id = original_class_to_id.get(src_class)
        if src_class_id is None:
            raise MergedSubsetEvaluationError(
                f"class_mapping references unknown source class {src_class!r}."
            )

        matching_stems: set[str] = set()
        n_objects = 0
        for scene_dir in sorted(raw_test_path.iterdir()):
            if not scene_dir.is_dir():
                continue
            scene_name = scene_dir.name
            labels_dir = scene_dir / "labels"
            if not labels_dir.exists():
                raise MergedSubsetEvaluationError(
                    f"Raw test scene is missing labels directory: {labels_dir}"
                )
            for label_file in labels_dir.glob("*.txt"):
                if label_file.stat().st_size == 0:
                    continue
                file_hits = 0
                with label_file.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        if len(parts) < 5:
                            raise MergedSubsetEvaluationError(
                                f"Malformed label row in {label_file}: {line.strip()!r}"
                            )
                        if int(parts[0]) == src_class_id:
                            file_hits += 1
                if file_hits:
                    prepared_stem = f"{label_file.stem}__scene_{scene_name}"
                    matching_stems.add(prepared_stem)
                    n_objects += file_hits

        if not matching_stems:
            logger.info("No test images found containing '%s' annotations.", src_class)
            continue

        temp_dir = Path(tempfile.mkdtemp())
        try:
            temp_images = temp_dir / "images"
            temp_labels = temp_dir / "labels"
            temp_images.mkdir()
            temp_labels.mkdir()

            copied = 0
            for stem in matching_stems:
                for ext in (".jpg", ".jpeg", ".png", ".bmp"):
                    img_src = test_images_dir / f"{stem}{ext}"
                    if not img_src.exists():
                        continue
                    shutil.copy2(img_src, temp_images / img_src.name)
                    label_src = test_labels_dir / f"{stem}.txt"
                    shutil.copy2(label_src, temp_labels / f"{stem}.txt")
                    copied += 1
                    break

            if copied == 0:
                raise MergedSubsetEvaluationError(
                    f"Could not locate prepared test images for '{src_class}' subset."
                )

            temp_yaml = temp_dir / "dataset.yaml"
            temp_config = {
                "path": str(temp_dir),
                "train": "images",
                "val": "images",
                "nc": ds_config["nc"],
                "names": ds_config["names"],
            }
            with temp_yaml.open("w", encoding="utf-8") as handle:
                yaml.dump(temp_config, handle)

            eval_kwargs = {k: v for k, v in kwargs.items() if k in ["imgsz", "batch", "conf", "iou"]}
            eval_kwargs["workers"] = 0
            eval_kwargs["batch"] = 1

            _, class_ids = get_dataset_classes(str(temp_yaml))
            subset_metrics = model.val(
                data=str(temp_yaml),
                classes=class_ids,
                verbose=False,
                save=False,
                plots=False,
                **eval_kwargs,
            )

            p = float(subset_metrics.results_dict["metrics/precision(B)"])
            r = float(subset_metrics.results_dict["metrics/recall(B)"])
            ap50 = float(subset_metrics.results_dict["metrics/mAP50(B)"])
            ap = float(subset_metrics.results_dict["metrics/mAP50-95(B)"])
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

            results[src_class] = {
                "target_class": target_class,
                "precision": p,
                "recall": r,
                "ap50": ap50,
                "ap": ap,
                "f1_score": f1,
                "n_objects": n_objects,
            }
            logger.info(
                "Merged-class subset '%s->%s': %s images, %s objects, mAP50=%.4f, mAP50-95=%.4f",
                src_class,
                target_class,
                copied,
                n_objects,
                ap50,
                ap,
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    return results


__all__ = ["MergedSubsetEvaluationError", "evaluate_merged_class_subsets"]
