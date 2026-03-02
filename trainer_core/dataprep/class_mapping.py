"""Class-mapping utilities for dataprep label remapping.

This file owns class-name/id mapping logic so transforms orchestration does not
mix mapping policy with filesystem/splitting work.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)

COCO_CLASSES = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "airplane": 4,
    "bus": 5,
    "train": 6,
    "truck": 7,
    "boat": 8,
    "traffic light": 9,
    "fire hydrant": 10,
    "stop sign": 11,
    "parking meter": 12,
    "bench": 13,
    "bird": 14,
    "cat": 15,
    "dog": 16,
    "horse": 17,
    "sheep": 18,
    "cow": 19,
    "elephant": 20,
    "bear": 21,
    "zebra": 22,
    "giraffe": 23,
    "backpack": 24,
    "umbrella": 25,
    "handbag": 26,
    "tie": 27,
    "suitcase": 28,
    "frisbee": 29,
    "skis": 30,
    "snowboard": 31,
    "sports ball": 32,
    "kite": 33,
    "baseball bat": 34,
    "baseball glove": 35,
    "skateboard": 36,
    "surfboard": 37,
    "tennis racket": 38,
    "bottle": 39,
    "wine glass": 40,
    "cup": 41,
    "fork": 42,
    "knife": 43,
    "spoon": 44,
    "bowl": 45,
    "banana": 46,
    "apple": 47,
    "sandwich": 48,
    "orange": 49,
    "broccoli": 50,
    "carrot": 51,
    "hot dog": 52,
    "pizza": 53,
    "donut": 54,
    "cake": 55,
    "chair": 56,
    "couch": 57,
    "potted plant": 58,
    "bed": 59,
    "dining table": 60,
    "toilet": 61,
    "tv": 62,
    "laptop": 63,
    "mouse": 64,
    "remote": 65,
    "keyboard": 66,
    "cell phone": 67,
    "microwave": 68,
    "oven": 69,
    "toaster": 70,
    "sink": 71,
    "refrigerator": 72,
    "book": 73,
    "clock": 74,
    "vase": 75,
    "scissors": 76,
    "teddy bear": 77,
    "hair drier": 78,
    "toothbrush": 79,
}

ALL_COCO_BY_ID = [name for name, _id in sorted(COCO_CLASSES.items(), key=lambda kv: kv[1])]
SELECTED_COCO_CLASSES = {i: cls for i, cls in enumerate(ALL_COCO_BY_ID)}


def get_class_mapping(custom_classes=None, use_coco_classes=True):
    if custom_classes:
        return {i: class_name for i, class_name in enumerate(custom_classes)}
    if use_coco_classes:
        return SELECTED_COCO_CLASSES
    return {}


def apply_class_mapping_config(custom_classes, class_mapping_config):
    if not class_mapping_config or not custom_classes:
        return custom_classes, {}

    source_to_target = {}
    for target_class, source_classes in class_mapping_config.items():
        if isinstance(source_classes, list):
            for source_class in source_classes:
                source_to_target[source_class] = target_class
        else:
            source_to_target[source_classes] = target_class

    mapped_classes = []
    seen_targets = set()
    for original_class in custom_classes:
        target_class = source_to_target.get(original_class, original_class)
        if target_class not in seen_targets:
            mapped_classes.append(target_class)
            seen_targets.add(target_class)

    logger.info("Class mapping applied: original=%s mapped=%s rules=%s", custom_classes, mapped_classes, source_to_target)
    return mapped_classes, source_to_target


def map_class_names_to_ids(class_names, target_mapping):
    class_mapping = {}

    if isinstance(class_names, dict):
        source_id_to_name = {int(k): v for k, v in class_names.items()}
    elif isinstance(class_names, list):
        source_id_to_name = {i: name for i, name in enumerate(class_names)}
    else:
        logger.warning(
            "Unexpected format for class_names. Expected dict or list, got %s",
            type(class_names),
        )
        return {}

    target_name_to_id = {name.lower(): id_num for id_num, name in target_mapping.items()}

    for source_id, source_name in source_id_to_name.items():
        source_name_lower = source_name.lower()
        if source_name_lower in target_name_to_id:
            class_mapping[source_id] = target_name_to_id[source_name_lower]
            logger.info(
                "Mapped %s (%s) -> %s",
                source_name,
                source_id,
                target_name_to_id[source_name_lower],
            )
        else:
            logger.warning("No mapping found for class '%s' (ID: %s)", source_name, source_id)

    return class_mapping


def remap_labels_with_class_mapping(
    label_path: Path,
    source_to_target_map: dict,
    original_classes: list,
    final_classes: list,
) -> None:
    if not source_to_target_map or not label_path.exists() or label_path.stat().st_size == 0:
        return

    # Map every original class-id to its post-merge class-id so we:
    # - remap merged classes, and
    # - shift IDs for classes after removed/merged classes.
    final_name_to_id = {name: idx for idx, name in enumerate(final_classes)}
    source_to_target_id: dict[int, int] = {}
    for source_id, source_name in enumerate(original_classes):
        target_name = source_to_target_map.get(source_name, source_name)
        target_id = final_name_to_id.get(target_name)
        if target_id is None:
            logger.warning(
                "No target id for class %r (mapped from %r) while remapping %s; keeping id %s unchanged",
                target_name,
                source_name,
                label_path,
                source_id,
            )
            continue
        source_to_target_id[source_id] = target_id

    if not source_to_target_id:
        return

    remapped_lines = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    class_id = int(parts[0])
                    target_id = source_to_target_id.get(class_id)
                    if target_id is not None:
                        parts[0] = str(target_id)
                    else:
                        logger.warning(
                            "Label row in %s references unknown class id %s; leaving unchanged",
                            label_path,
                            class_id,
                        )
                    remapped_lines.append(" ".join(parts))
                except ValueError as exc:
                    logger.warning(
                        "Skipping malformed label row in %s: %r (%s)",
                        label_path,
                        line.strip(),
                        exc,
                    )
                    continue

    with open(label_path, "w", encoding="utf-8") as f:
        for line in remapped_lines:
            f.write(line + "\n")


__all__ = [
    "apply_class_mapping_config",
    "get_class_mapping",
    "map_class_names_to_ids",
    "remap_labels_with_class_mapping",
]
