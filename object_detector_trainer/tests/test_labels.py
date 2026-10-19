from __future__ import annotations

from pathlib import Path

import pytest

from object_detector_trainer.labels import label_class_ids, normalize_yolo_label


def test_normalize_remaps_classes_and_leaves_the_source_unchanged(tmp_path: Path) -> None:
    label = tmp_path / "label.txt"
    original = "0 0.5 0.5 0.3 0.3\n1 0.7 0.7 0.2 0.2\n2 0.1 0.1 0.1 0.1\n"
    label.write_text(original, encoding="utf-8")

    normalized = normalize_yolo_label(label, {0: 0, 1: 0, 2: 1})

    assert normalized == (
        "0 0.500000 0.500000 0.300000 0.300000\n"
        "0 0.700000 0.700000 0.200000 0.200000\n"
        "1 0.100000 0.100000 0.100000 0.100000\n"
    )
    assert label.read_text(encoding="utf-8") == original


def test_polygon_becomes_its_bounding_box(tmp_path: Path) -> None:
    label = tmp_path / "label.txt"
    label.write_text("0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n", encoding="utf-8")

    assert normalize_yolo_label(label, {0: 0}) == "0 0.500000 0.500000 0.800000 0.800000\n"


def test_missing_and_empty_labels_mean_background(tmp_path: Path) -> None:
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")

    assert normalize_yolo_label(tmp_path / "missing.txt", {0: 0}) == ""
    assert normalize_yolo_label(empty, {0: 0}) == ""
    assert normalize_yolo_label(None, {0: 0}) == ""


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ("0 1.111 0.5 0.2 0.2", "Invalid normalized YOLO box"),
        ("0 0.5 0.5 -0.2 0.2", "Invalid normalized YOLO box"),
        ("0 0.5 0.5", "Malformed YOLO label row"),
        ("0 0.1 0.1 0.9 0.1 0.9", "Malformed YOLO polygon row"),
        ("5 0.5 0.5 0.2 0.2", "Unknown class id 5"),
    ],
)
def test_invalid_rows_are_rejected_with_their_location(tmp_path: Path, row: str, message: str) -> None:
    label = tmp_path / "label.txt"
    label.write_text(row + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        normalize_yolo_label(label, {0: 0})


def test_label_class_ids_lists_one_id_per_object(tmp_path: Path) -> None:
    label = tmp_path / "label.txt"
    label.write_text("1 0.5 0.5 0.1 0.1\n\n0 0.2 0.2 0.1 0.1\n1 0.3 0.3 0.1 0.1\n", encoding="utf-8")

    assert label_class_ids(label) == [1, 0, 1]
    assert label_class_ids(tmp_path / "missing.txt") == []
