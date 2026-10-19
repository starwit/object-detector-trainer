from __future__ import annotations

import pytest

from object_detector_trainer.classes import final_classes, merged_classes, source_label_names
from object_detector_trainer.config import DataConfig


def data(custom_classes: list[str], class_mapping: dict | None = None) -> DataConfig:
    return DataConfig(dataset_name="sample", custom_classes=custom_classes, class_mapping=class_mapping or {})


def test_without_mapping_the_configured_classes_are_trained() -> None:
    assert final_classes(data(["waste", "cigarette"])) == ["waste", "cigarette"]
    assert merged_classes(data(["waste", "cigarette"])) == {}


def test_mapping_merges_and_renames() -> None:
    config = data(
        ["waste", "cigarette", "bottle", "leaf_pile"],
        {"waste": ["waste", "bottle"], "leaves": ["leaf_pile"]},
    )

    assert final_classes(config) == ["waste", "cigarette", "leaves"]
    assert merged_classes(config) == {"bottle": "waste", "leaf_pile": "leaves"}


def test_mapping_a_single_name_as_text() -> None:
    assert final_classes(data(["waste", "cigarette"], {"waste": "cigarette"})) == ["waste"]


def test_mapping_an_unknown_class_is_an_error() -> None:
    with pytest.raises(ValueError, match="'can', which is not a configured class"):
        final_classes(data(["waste"], {"waste": ["can"]}))


def test_source_names_are_matched_by_name_case_insensitively() -> None:
    config = data(["waste", "cigarette"])

    assert source_label_names(["Cigarette", "waste"], config) == {0: "cigarette", 1: "waste"}
    assert source_label_names({"3": "waste"}, config) == {3: "waste"}
    assert source_label_names(None, config) == {0: "waste", 1: "cigarette"}


def test_unknown_source_class_is_an_error() -> None:
    with pytest.raises(ValueError, match="'person'"):
        source_label_names(["person"], data(["waste"]))
