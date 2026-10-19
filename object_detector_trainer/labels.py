"""Read YOLO label files and rewrite them with final class IDs.

A label row is either a box (``class x_center y_center width height``) or a
polygon (``class x1 y1 x2 y2 ...``). Polygons are converted to their bounding
box. All coordinates are normalized to 0..1.
"""

from __future__ import annotations

from pathlib import Path


def normalize_yolo_label(label_path: Path | None, class_id_map: dict[int, int | None]) -> str:
    """Return the label as box rows with final class IDs. The source file is not changed.

    ``class_id_map`` maps the label's class IDs to final class IDs; rows of
    class IDs mapped to None (ignored classes) are dropped. A missing or empty
    label file means a background image and returns an empty string.
    """
    if label_path is None or not label_path.exists():
        return ""

    rows = []
    with label_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.split()
            if not parts:
                continue
            location = f"{label_path}:{line_number}"
            if len(parts) < 5:
                raise ValueError(f"Malformed YOLO label row in {location}: {line.strip()!r}")
            if len(parts) > 5 and len(parts) % 2 == 0:
                raise ValueError(f"Malformed YOLO polygon row in {location}: {line.strip()!r}")

            label_id = int(float(parts[0]))
            if label_id not in class_id_map:
                raise ValueError(f"Unknown class id {label_id} in {location}.")
            if class_id_map[label_id] is None:
                continue

            coordinates = [float(value) for value in parts[1:]]
            if len(coordinates) == 4:
                box = coordinates
            else:
                box = _polygon_to_box(coordinates)
            _check_box(box, location)

            box_text = " ".join(f"{value:.6f}" for value in box)
            rows.append(f"{class_id_map[label_id]} {box_text}\n")
    return "".join(rows)


def label_class_ids(label_path: Path | None) -> list[int]:
    """The class ID of every object in a YOLO label file, one entry per object."""
    if label_path is None or not label_path.exists():
        return []
    class_ids = []
    with label_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if parts:
                class_ids.append(int(float(parts[0])))
    return class_ids


def _polygon_to_box(points: list[float]) -> list[float]:
    xs = points[0::2]
    ys = points[1::2]
    left = max(0.0, min(xs))
    top = max(0.0, min(ys))
    right = min(1.0, max(xs))
    bottom = min(1.0, max(ys))
    return [(left + right) / 2, (top + bottom) / 2, right - left, bottom - top]


def _check_box(box: list[float], location: str) -> None:
    x_center, y_center, width, height = box
    inside = 0 <= x_center <= 1 and 0 <= y_center <= 1
    sized = 0 < width <= 1 and 0 < height <= 1
    if not (inside and sized):
        raise ValueError(
            f"Invalid normalized YOLO box in {location}: {x_center} {y_center} {width} {height}"
        )
