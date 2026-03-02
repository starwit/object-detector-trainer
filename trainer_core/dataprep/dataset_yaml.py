from __future__ import annotations

from pathlib import Path

from trainer_core.dataprep.class_mapping import apply_class_mapping_config, get_class_mapping
from trainer_core.datasets.yolo_yaml import write_yolo_dataset_yaml


def create_dataset_yaml(
    dataset_path: Path,
    custom_classes=None,
    use_coco_classes: bool = True,
    class_mapping_config=None,
) -> None:
    """Create YOLO dataset.yaml with optional merged class mapping."""
    final_classes = custom_classes
    if class_mapping_config and custom_classes:
        final_classes, _ = apply_class_mapping_config(custom_classes, class_mapping_config)

    class_mapping = get_class_mapping(final_classes, use_coco_classes)

    write_yolo_dataset_yaml(
        output_path=dataset_path / "dataset.yaml",
        dataset_root=dataset_path,
        class_names=class_mapping,
        train_rel="train/images",
        val_rel="val/images",
    )

    print(f"Created dataset.yaml with {len(class_mapping)} classes: {list(class_mapping.values())}")


__all__ = ["create_dataset_yaml"]
