"""Class names: the configured classes, merging with class_mapping, and label IDs.

Terms used here:

- configured classes: ``data.custom_classes`` (or the COCO classes), in label-ID order.
- final classes: the classes the model is trained on, after ``data.class_mapping``
  merged or renamed configured classes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from object_detector_trainer.config import DataConfig

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


def configured_classes(data: DataConfig) -> list[str]:
    if data.use_coco_classes:
        return list(COCO_CLASSES)
    return list(data.custom_classes)


def final_name_by_configured_name(data: DataConfig) -> dict[str, str]:
    """Map every configured class to the final class it is trained as."""
    configured = configured_classes(data)
    final_names = {name: name for name in configured}
    for target, sources in data.class_mapping.items():
        if isinstance(sources, str):
            sources = [sources]
        for source in sources:
            if source not in final_names:
                raise ValueError(
                    f"data.class_mapping uses {source!r}, which is not a configured class. "
                    f"Configured classes: {configured}"
                )
            final_names[source] = target
    return final_names


def final_classes(data: DataConfig) -> list[str]:
    """The training classes, in the order they first appear in the configured classes."""
    names: list[str] = []
    for final_name in final_name_by_configured_name(data).values():
        if final_name not in names:
            names.append(final_name)
    return names


def merged_classes(data: DataConfig) -> dict[str, str]:
    """Configured classes that are trained under a different name, mapped to that name."""
    merged = {}
    for configured_name, final_name in final_name_by_configured_name(data).items():
        if configured_name != final_name:
            merged[configured_name] = final_name
    return merged


def source_label_names(source_names: list[str] | dict | None, data: DataConfig) -> dict[int, str | None]:
    """Map the label IDs of one source to configured class names.

    ``source_names`` are the class names from the source's own data.yaml.
    Without them, the source's label IDs are assumed to follow the configured
    class order. Names are matched case-insensitively. Classes listed in
    ``data.ignore_classes`` map to None: their labels are dropped.
    """
    configured = configured_classes(data)
    if source_names is None:
        return dict(enumerate(configured))

    if isinstance(source_names, list):
        source_names = dict(enumerate(source_names))

    configured_by_lower_name = {name.lower(): name for name in configured}
    ignored = {name.lower() for name in data.ignore_classes}
    label_names = {}
    for label_id, name in source_names.items():
        if str(name).lower() in ignored:
            label_names[int(label_id)] = None
            continue
        configured_name = configured_by_lower_name.get(str(name).lower())
        if configured_name is None:
            raise ValueError(
                f"Source class {name!r} (label ID {label_id}) is not a configured class. "
                f"Configured classes: {configured}. Add it to data.custom_classes, "
                "or to data.ignore_classes to drop its labels."
            )
        label_names[int(label_id)] = configured_name
    return label_names
