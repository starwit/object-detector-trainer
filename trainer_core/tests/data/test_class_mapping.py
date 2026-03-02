"""Tests for class mapping functionality."""

from __future__ import annotations

import tempfile
from pathlib import Path

from trainer_core.dataprep.class_mapping import (
    apply_class_mapping_config,
    remap_labels_with_class_mapping,
)


def test_apply_class_mapping_config_basic() -> None:
    """Test basic class mapping configuration."""
    custom_classes = ["waste", "cigarette"]
    class_mapping_config = {"waste": ["waste", "cigarette"]}

    mapped_classes, source_to_target = apply_class_mapping_config(
        custom_classes, class_mapping_config
    )

    # Should have only one class after mapping
    assert len(mapped_classes) == 1
    assert mapped_classes == ["waste"]

    # Both source classes should map to waste
    assert source_to_target["waste"] == "waste"
    assert source_to_target["cigarette"] == "waste"


def test_apply_class_mapping_config_multiple_targets() -> None:
    """Test mapping with multiple target classes."""
    custom_classes = ["waste", "cigarette", "bottle", "can"]
    class_mapping_config = {
        "waste": ["waste", "cigarette"],
        "recyclable": ["bottle", "can"],
    }

    mapped_classes, source_to_target = apply_class_mapping_config(
        custom_classes, class_mapping_config
    )

    # Should have two classes after mapping
    assert len(mapped_classes) == 2
    assert "waste" in mapped_classes
    assert "recyclable" in mapped_classes

    # Check mappings
    assert source_to_target["waste"] == "waste"
    assert source_to_target["cigarette"] == "waste"
    assert source_to_target["bottle"] == "recyclable"
    assert source_to_target["can"] == "recyclable"


def test_apply_class_mapping_config_partial() -> None:
    """Test mapping where only some classes are mapped."""
    custom_classes = ["waste", "cigarette", "person"]
    class_mapping_config = {"waste": ["waste", "cigarette"]}

    mapped_classes, source_to_target = apply_class_mapping_config(
        custom_classes, class_mapping_config
    )

    # Should have two classes: waste (merged) and person (unmapped)
    assert len(mapped_classes) == 2
    assert "waste" in mapped_classes
    assert "person" in mapped_classes

    # Check mappings
    assert source_to_target["waste"] == "waste"
    assert source_to_target["cigarette"] == "waste"
    assert "person" not in source_to_target  # Unmapped class


def test_apply_class_mapping_config_empty() -> None:
    """Test with no class mapping configuration."""
    custom_classes = ["waste", "cigarette"]
    class_mapping_config = {}

    mapped_classes, source_to_target = apply_class_mapping_config(
        custom_classes, class_mapping_config
    )

    # Should return original classes
    assert mapped_classes == custom_classes
    assert source_to_target == {}


def test_remap_labels_with_class_mapping() -> None:
    """Test remapping label files."""
    # Create a temporary label file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        temp_label_path = Path(f.name)
        # Write YOLO format labels
        # class_id center_x center_y width height
        f.write("0 0.5 0.5 0.3 0.3\n")  # waste
        f.write("1 0.7 0.7 0.2 0.2\n")  # cigarette
        f.write("0 0.3 0.3 0.4 0.4\n")  # waste

    try:
        original_class_list = ["waste", "cigarette"]
        final_class_list = ["waste"]
        source_to_target_map = {
            "waste": "waste",
            "cigarette": "waste",
        }

        # Apply remapping
        remap_labels_with_class_mapping(
            temp_label_path,
            source_to_target_map,
            original_class_list,
            final_class_list,
        )

        # Read the remapped file
        with open(temp_label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # All labels should now have class ID 0 (waste)
        assert len(lines) == 3
        for line in lines:
            parts = line.strip().split()
            assert parts[0] == "0"  # All should be mapped to class 0

    finally:
        # Clean up
        temp_label_path.unlink()


def test_remap_labels_preserves_coordinates() -> None:
    """Test that remapping preserves bounding box coordinates."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        temp_label_path = Path(f.name)
        # Write labels with specific coordinates
        f.write("1 0.123456 0.654321 0.5 0.5\n")

    try:
        original_class_list = ["waste", "cigarette"]
        final_class_list = ["waste"]
        source_to_target_map = {"cigarette": "waste"}

        # Apply remapping
        remap_labels_with_class_mapping(
            temp_label_path,
            source_to_target_map,
            original_class_list,
            final_class_list,
        )

        # Read the remapped file
        with open(temp_label_path, "r", encoding="utf-8") as f:
            line = f.readline().strip()

        parts = line.split()
        # Class should be remapped to 0, but coordinates should be preserved
        assert parts[0] == "0"
        assert parts[1] == "0.123456"
        assert parts[2] == "0.654321"
        assert parts[3] == "0.5"
        assert parts[4] == "0.5"

    finally:
        temp_label_path.unlink()


def test_remap_labels_empty_file() -> None:
    """Test remapping an empty label file."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        temp_label_path = Path(f.name)

    try:
        original_class_list = ["waste", "cigarette"]
        final_class_list = ["waste"]
        source_to_target_map = {"cigarette": "waste"}

        # Should not raise an error
        remap_labels_with_class_mapping(
            temp_label_path,
            source_to_target_map,
            original_class_list,
            final_class_list,
        )

        # File should still exist and be empty
        assert temp_label_path.exists()
        assert temp_label_path.stat().st_size == 0

    finally:
        temp_label_path.unlink()


def test_remap_labels_preserves_unmapped_rows_and_shifts_ids() -> None:
    """Unmapped classes must be preserved, and their IDs should shift after merges."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        temp_label_path = Path(f.name)
        f.write("0 0.5 0.5 0.3 0.3\n")  # waste (unmapped)
        f.write("1 0.7 0.7 0.2 0.2\n")  # cigarette (merged into waste)
        f.write("2 0.1 0.1 0.1 0.1\n")  # person (unmapped but must shift from 2 -> 1)

    try:
        original_class_list = ["waste", "cigarette", "person"]
        final_class_list, source_to_target_map = apply_class_mapping_config(
            original_class_list,
            {"waste": ["cigarette"]},
        )

        remap_labels_with_class_mapping(
            temp_label_path,
            source_to_target_map,
            original_class_list,
            final_class_list,
        )

        with open(temp_label_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        assert len(lines) == 3
        assert lines[0].split()[0] == "0"
        assert lines[1].split()[0] == "0"
        assert lines[2].split()[0] == "1"
    finally:
        temp_label_path.unlink()
