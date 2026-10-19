from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from object_detector_trainer.config import load_config
from object_detector_trainer.dataset import build_dataset, read_manifest
from object_detector_trainer.tests.helpers import minimal_params, write_image, write_label, write_params


def prepare(workspace: Path, **sections) -> dict:
    """Run preparation and return the prepared image names per split."""
    config = load_config(write_params(workspace / "params.yaml", minimal_params(**sections)))
    build_dataset(config, workspace)
    dataset_dir = workspace / "datasets" / config.data.dataset_name
    names = {}
    for split in ("train", "val", "test"):
        names[split] = sorted(path.name for path in (dataset_dir / split / "images").iterdir())
    return names


def test_random_split_is_reproducible_with_requested_counts(tmp_path: Path) -> None:
    for index in range(10):
        write_image(tmp_path / f"raw_data/train/source/images/{index}.jpg", index * 20)

    names = prepare(tmp_path, prepare={"val_split": 0.2, "test_split": 0.2})

    assert (len(names["train"]), len(names["val"]), len(names["test"])) == (6, 2, 2)
    assert prepare(tmp_path, prepare={"val_split": 0.2, "test_split": 0.2}) == names


def test_val_and_test_folders_replace_the_random_splits(tmp_path: Path) -> None:
    for index in range(4):
        write_image(tmp_path / f"raw_data/train/source/images/{index}.jpg", index * 20)
    write_image(tmp_path / "raw_data/test/held_out/images/t.jpg", 200)

    names = prepare(tmp_path, prepare={"val_split": 0.25, "test_split": 0.25})
    assert names["test"] == ["t__scene_held_out.jpg"]
    assert (len(names["train"]), len(names["val"])) == (3, 1)

    write_image(tmp_path / "raw_data/val/checked/images/v.jpg", 210)
    names = prepare(tmp_path, prepare={"val_split": 0.25, "test_split": 0.25})
    assert names == {
        "train": ["0__scene_source.jpg", "1__scene_source.jpg", "2__scene_source.jpg", "3__scene_source.jpg"],
        "val": ["v__scene_checked.jpg"],
        "test": ["t__scene_held_out.jpg"],
    }


def test_duplicates_are_removed_within_and_across_sources(tmp_path: Path) -> None:
    for index in range(4):
        write_image(tmp_path / f"raw_data/train/a/images/{index}.jpg", index * 20)
    write_image(tmp_path / "raw_data/train/a/images/copy_of_0.jpg", 0)
    write_image(tmp_path / "raw_data/train/b/images/copy_of_1.jpg", 20)
    write_image(tmp_path / "raw_data/test/t/images/t.jpg", 200)

    names = prepare(tmp_path, prepare={"val_split": 0.25})

    assert sorted(names["train"] + names["val"]) == [
        "0__scene_a.jpg", "1__scene_a.jpg", "2__scene_a.jpg", "3__scene_a.jpg",
    ]


def test_missing_and_empty_labels_become_empty_label_files(tmp_path: Path) -> None:
    source = tmp_path / "raw_data/train/source"
    for index in range(5):
        write_image(source / f"images/image{index}.jpg", index * 40)
    write_label(source / "labels/image0.txt", "0 0.5 0.5 0.2 0.2\n")
    write_label(source / "labels/image1.txt", "")

    for split, seed in (("train", 180), ("val", 200), ("test", 220)):
        write_image(tmp_path / f"raw_data/{split}/negatives/images/negative.jpg", seed)
    metadata = tmp_path / "raw_data/val/negatives/data.yaml"
    metadata.write_text("names: [waste]\n")
    original = {path: path.read_bytes() for path in (tmp_path / "raw_data").rglob("*") if path.is_file()}

    names = prepare(tmp_path)

    assert [len(names[split]) for split in ("train", "val", "test")] == [6, 1, 1]
    labels = sorted((tmp_path / "datasets/sample").glob("*/labels/*.txt"))
    assert len(labels) == 8
    assert [label.name for label in labels if label.read_text()] == ["image0__scene_source.txt"]
    assert {path: path.read_bytes() for path in (tmp_path / "raw_data").rglob("*") if path.is_file()} == original


def test_cvat_train_txt_lists_the_images(tmp_path: Path) -> None:
    source = tmp_path / "raw_data/train/cvat"
    write_image(source / "images/img1.jpg", 128)
    write_image(source / "images/img2.jpg", 200)
    write_image(source / "images/not_listed.jpg", 50)
    (source / "train.txt").write_text("data/images/img1.jpg\nimages/img2.jpg\n", encoding="utf-8")

    names = prepare(tmp_path, prepare={"val_split": 0, "test_split": 0.5})

    assert len(names["train"]) + len(names["test"]) == 2


@pytest.mark.parametrize(
    ("split_settings", "image_count", "message"),
    [
        ({"val_split": 0.25, "test_split": 0.25}, 1, "0 validation frames"),
        ({"val_split": 0.25, "test_split": 0}, 4, "0 test frames"),
        ({"val_split": 0.8, "test_split": 0.2}, 4, r"val_split \+ prepare.test_split must be < 1"),
    ],
)
def test_impossible_splits_are_reported(tmp_path: Path, split_settings: dict, image_count: int, message: str) -> None:
    for index in range(image_count):
        write_image(tmp_path / f"raw_data/train/source/images/{index}.jpg", index * 20)

    with pytest.raises(ValueError, match=message):
        prepare(tmp_path, prepare=split_settings)


def test_source_data_yaml_maps_classes_by_name_and_merges(tmp_path: Path) -> None:
    source = tmp_path / "raw_data/train/source"
    (source).mkdir(parents=True)
    (source / "data.yaml").write_text("names: [cigarette, waste]\n", encoding="utf-8")
    for index in range(4):
        write_image(source / f"images/{index}.jpg", index * 20)
        write_label(source / f"labels/{index}.txt", "0 0.5 0.5 0.2 0.2\n1 0.2 0.2 0.1 0.1\n")
    data = {
        "dataset_name": "sample",
        "custom_classes": ["waste", "cigarette"],
        "class_mapping": {"waste": ["cigarette"]},
    }

    prepare(tmp_path, data=data, prepare={"val_split": 0.25, "test_split": 0.25})

    dataset_dir = tmp_path / "datasets/sample"
    assert yaml.safe_load((dataset_dir / "dataset.yaml").read_text())["names"] == {0: "waste"}
    for label in dataset_dir.glob("*/labels/*.txt"):
        assert [line.split()[0] for line in label.read_text().splitlines()] == ["0", "0"]
    manifest = read_manifest(dataset_dir)
    assert {row["source_classes"] for row in manifest} == {"cigarette;waste"}
    assert {row["source"] for row in manifest} == {"source"}


def test_ignored_classes_are_dropped_from_the_labels(tmp_path: Path) -> None:
    source = tmp_path / "raw_data/train/cvat"
    source.mkdir(parents=True)
    (source / "data.yaml").write_text("names: {0: waste, 1: leaf_pile}\n", encoding="utf-8")
    for index in range(4):
        write_image(source / f"images/{index}.jpg", index * 20)
        write_label(source / f"labels/{index}.txt", "0 0.5 0.5 0.2 0.2\n1 0.2 0.2 0.1 0.1\n")
    data = {"dataset_name": "sample", "custom_classes": ["waste"], "ignore_classes": ["Leaf_Pile"]}

    prepare(tmp_path, data=data, prepare={"val_split": 0.25, "test_split": 0.25})

    dataset_dir = tmp_path / "datasets/sample"
    for label in dataset_dir.glob("*/labels/*.txt"):
        assert label.read_text() == "0 0.500000 0.500000 0.200000 0.200000\n"
    assert {row["source_classes"] for row in read_manifest(dataset_dir)} == {"waste"}

    with pytest.raises(ValueError, match="or to data.ignore_classes"):
        prepare(tmp_path, data={"dataset_name": "sample", "custom_classes": ["waste"]})


def prepare_cvat(workspace: Path, **sections) -> dict:
    return prepare(workspace, data={"dataset_name": "sample", "custom_classes": ["waste", "cigarette"]}, **sections)


def write_cvat_export(source: Path, subsets: dict[str, list[int]], *, inverted: bool = False) -> None:
    """A CVAT-style export with explicit lists, labels and source class names."""
    metadata = {"path": ".", "names": ["cigarette", "waste"]}
    for subset, seeds in subsets.items():
        paths = []
        for index, seed in enumerate(seeds):
            relative = f"{subset}/images/{index}.jpg" if inverted else f"images/{subset}/{index}.jpg"
            image = source / relative
            write_image(image, seed)
            label = source / relative.replace("images/", "labels/")
            write_label(label.with_suffix(".txt"), "0 0.5 0.5 0.2 0.2\n")
            paths.append("data/" + relative)
        source.mkdir(parents=True, exist_ok=True)
        (source / f"{subset}.txt").write_text("\n".join(paths))
        metadata[subset] = f"{subset}.txt"
    (source / "data.yaml").write_text(yaml.safe_dump(metadata))


@pytest.mark.parametrize("inverted", [False, True])
@pytest.mark.parametrize("direct_root", [False, True])
def test_cvat_project_keeps_membership_and_maps_classes(tmp_path: Path, inverted: bool, direct_root: bool) -> None:
    source = tmp_path / "raw_data" if direct_root else tmp_path / "raw_data/cvat"
    write_cvat_export(source, {"TRAIN": [10, 20], "Validation": [100], "test": [200]}, inverted=inverted)
    names = prepare_cvat(tmp_path)
    assert [len(names[split]) for split in ("train", "val", "test")] == [2, 1, 1]
    rows = read_manifest(tmp_path / "datasets/sample")
    for row in rows:
        assert row["split"] == {"TRAIN": "train", "Validation": "val", "test": "test"}[
            next(part for part in Path(row["source_image"]).parts if part in ("TRAIN", "Validation", "test"))
        ]
    for label in (tmp_path / "datasets/sample").glob("*/labels/*.txt"):
        assert label.read_text() == "1 0.500000 0.500000 0.200000 0.200000\n"


@pytest.mark.parametrize("fixed", [None, "Validation", "Test"])
def test_cvat_only_generates_missing_splits(tmp_path: Path, fixed: str | None) -> None:
    subsets = {"Train": list(range(10))}
    if fixed:
        subsets[fixed] = [100, 200]
    write_cvat_export(tmp_path / "raw_data/cvat", subsets)
    names = prepare_cvat(tmp_path, prepare={"val_split": 0.2, "test_split": 0.2})
    assert [len(names[split]) for split in ("train", "val", "test")] == ([8, 2, 2] if fixed else [6, 2, 2])
    if fixed:
        rows = read_manifest(tmp_path / "datasets/sample")
        expected_split = "val" if fixed == "Validation" else "test"
        assert all(row["split"] == expected_split for row in rows if f"images/{fixed}/" in row["source_image"])


def test_outer_folder_assigns_a_single_task_export(tmp_path: Path) -> None:
    write_cvat_export(tmp_path / "raw_data/Train/training", {"Train": list(range(10))})
    write_cvat_export(tmp_path / "raw_data/Validation/validation", {"Train": [100]})
    write_cvat_export(tmp_path / "raw_data/TEST/test", {"Train": [200]})
    names = prepare_cvat(tmp_path)
    assert [len(names[split]) for split in ("train", "val", "test")] == [10, 1, 1]


def test_replacing_export_rebuilds_without_stale_images(tmp_path: Path) -> None:
    source = tmp_path / "raw_data/cvat"
    write_cvat_export(source, {"Train": [10, 20], "Val": [100], "Test": [200]})
    prepare_cvat(tmp_path)
    stale = tmp_path / "datasets/sample/train/images/stale.jpg"
    write_image(stale)
    shutil.rmtree(source)
    write_cvat_export(source, {"Train": [30], "Val": [100], "Test": [200, 210]})
    names = prepare_cvat(tmp_path)
    assert [len(names[split]) for split in ("train", "val", "test")] == [1, 1, 2]
    assert not (tmp_path / "datasets/sample/train/images/1__scene_cvat.jpg").exists()
    assert not stale.exists()


@pytest.mark.parametrize("problem, message", [
    ("empty", "contains no images"),
    ("missing", "references missing"),
    ("repeated", "listed more than once"),
    ("nested", "contains several subsets"),
    ("alias", "Multiple entries"),
    ("unknown", "Unsupported subset"),
])
def test_invalid_cvat_exports_fail_before_writing(tmp_path: Path, problem: str, message: str) -> None:
    source = tmp_path / ("raw_data/train/cvat" if problem == "nested" else "raw_data/cvat")
    subsets = {"Train": [10], "Val": [100], "Test": [200]}
    if problem == "empty":
        subsets["Val"] = []
    write_cvat_export(source, subsets)
    if problem == "missing":
        (source / "Test.txt").write_text("data/images/Test/missing.jpg\n")
    if problem == "repeated":
        path = source / "Train.txt"
        path.write_text(path.read_text() + "\n" + path.read_text())
    if problem in ("alias", "unknown"):
        path = source / "data.yaml"
        metadata = yaml.safe_load(path.read_text())
        metadata["validation" if problem == "alias" else "holdout"] = "Val.txt"
        path.write_text(yaml.safe_dump(metadata))
    with pytest.raises((ValueError, FileNotFoundError), match=message):
        prepare_cvat(tmp_path)
    assert not (tmp_path / "datasets/sample").exists()


def test_export_named_train_can_reference_subset_directories(tmp_path: Path) -> None:
    source = tmp_path / "raw_data/train"
    write_cvat_export(source, {"Train": [10, 20], "Valid": [100], "Test": [200]})
    path = source / "data.yaml"
    metadata = yaml.safe_load(path.read_text())
    for subset in ("Train", "Valid", "Test"):
        metadata[subset] = f"images/{subset}"
    path.write_text(yaml.safe_dump(metadata))
    names = prepare_cvat(tmp_path)
    assert [len(names[split]) for split in ("train", "val", "test")] == [2, 1, 1]


@pytest.mark.parametrize("relative", ["project.zip", "train/task.ZIP"])
def test_zip_files_must_be_unpacked_first(tmp_path: Path, relative: str) -> None:
    archive = tmp_path / "raw_data" / relative
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"zip")
    with pytest.raises(ValueError, match="is a ZIP file") as error:
        prepare_cvat(tmp_path)
    assert f"unzip {archive} -d {archive.with_suffix('')}" in str(error.value)
    assert not (tmp_path / "datasets/sample").exists()


def test_duplicates_are_removed_when_all_sets_are_supplied(tmp_path: Path) -> None:
    for index in range(3):
        write_image(tmp_path / f"raw_data/train/a/images/{index}.jpg", index * 20)
    write_image(tmp_path / "raw_data/train/b/images/copy_of_0.jpg", 0)
    write_image(tmp_path / "raw_data/val/v/images/v.jpg", 100)
    write_image(tmp_path / "raw_data/val/v/images/copy_of_v.jpg", 100)
    write_image(tmp_path / "raw_data/test/t/images/t.jpg", 200)

    names = prepare(tmp_path)

    assert names == {
        "train": ["0__scene_a.jpg", "1__scene_a.jpg", "2__scene_a.jpg"],
        "val": ["copy_of_v__scene_v.jpg"],
        "test": ["t__scene_t.jpg"],
    }


@pytest.mark.parametrize("pair, kept", [
    (("train", "val"), "val"), (("train", "test"), "test"), (("val", "test"), "test"),
])
def test_near_duplicates_across_sets_are_kept_in_the_evaluation_set(
    tmp_path: Path, pair: tuple[str, str], kept: str,
) -> None:
    import cv2

    for index, split in enumerate(("train", "val", "test")):
        write_image(tmp_path / f"raw_data/{split}/{split}/images/unique.jpg", index + 100)

    first = tmp_path / f"raw_data/{pair[0]}/{pair[0]}/images/duplicate.PNG"
    second = tmp_path / f"raw_data/{pair[1]}/{pair[1]}/images/reencoded.bmp"
    write_image(first, 10)
    assert cv2.imwrite(str(second), cv2.imread(str(first)))
    assert first.read_bytes() != second.read_bytes()
    original = {path: path.read_bytes() for path in (tmp_path / "raw_data").rglob("*") if path.is_file()}

    names = prepare(tmp_path)

    assert [len(names[split]) for split in ("train", "val", "test")] == [
        2 if split == kept else 1 for split in ("train", "val", "test")
    ]
    assert {path: path.read_bytes() for path in original} == original


def test_same_stem_with_different_extensions_keeps_separate_labels(tmp_path: Path) -> None:
    source = tmp_path / "raw_data/train/source"
    write_image(source / "images/a/frame.jpg", 10)
    write_image(source / "images/b/frame.png", 20)
    write_label(source / "labels/a/frame.txt", "0 0.3 0.3 0.1 0.1\n")
    write_label(source / "labels/b/frame.txt", "0 0.7 0.7 0.2 0.2\n")
    write_image(tmp_path / "raw_data/val/val/images/val.jpg", 100)
    write_image(tmp_path / "raw_data/test/test/images/test.jpg", 200)

    names = prepare(tmp_path)

    assert names["train"] == ["frame__scene_source.jpg", "frame__scene_source__dup1.png"]
    labels = tmp_path / "datasets/sample/train/labels"
    assert (labels / "frame__scene_source.txt").read_text() == "0 0.300000 0.300000 0.100000 0.100000\n"
    assert (labels / "frame__scene_source__dup1.txt").read_text() == "0 0.700000 0.700000 0.200000 0.200000\n"


def add_export(workspace: Path, relative: str, subsets: dict[str, list[int]]) -> None:
    write_cvat_export(workspace / "raw_data" / relative, subsets)


def test_project_export_and_manual_sources_are_combined(tmp_path: Path) -> None:
    add_export(tmp_path, "project", {"Train": [10, 20], "Validation": [100], "Test": [200]})
    for split, seeds in (("train", [30, 40]), ("val", [110]), ("test", [210])):
        add_export(tmp_path, f"{split}/extra-{split}", {"Train": seeds})
    names = prepare_cvat(tmp_path)
    assert [len(names[split]) for split in ("train", "val", "test")] == [4, 2, 2]
    rows = read_manifest(tmp_path / "datasets/sample")
    assert {row["source"] for row in rows} == {"project", "extra-train", "extra-val", "extra-test"}
    assert all(row["split"] == row["source"].removeprefix("extra-")
               for row in rows if row["source"].startswith("extra-"))


@pytest.mark.parametrize("partial", [False, True])
def test_multiple_root_exports_combine_assignments_before_splitting(tmp_path: Path, partial: bool) -> None:
    first = {"Train": [10, 20], "Validation": [100]}
    second = {"Train": [30, 40], "Test": [200]}
    if not partial:
        first["Test"] = [210]
        second["Validation"] = [110]
    add_export(tmp_path, "first", first)
    add_export(tmp_path, "second", second)
    names = prepare_cvat(tmp_path)
    assert [len(names[split]) for split in ("train", "val", "test")] == [4, 1 if partial else 2, 1 if partial else 2]
    assert {row["source"] for row in read_manifest(tmp_path / "datasets/sample")} == {"first", "second"}


def test_multiple_train_only_exports_split_the_combined_pool(tmp_path: Path) -> None:
    add_export(tmp_path, "first", {"Train": list(range(10))})
    add_export(tmp_path, "second", {"Train": list(range(100, 110))})
    names = prepare_cvat(tmp_path, prepare={"val_split": 0.2, "test_split": 0.2})
    assert [len(names[split]) for split in ("train", "val", "test")] == [12, 4, 4]


def test_multiple_sources_work_only_inside_outer_split_folders(tmp_path: Path) -> None:
    for split, seeds in (("train", [10, 20]), ("val", [100, 110]), ("test", [200, 210])):
        for index, seed in enumerate(seeds):
            add_export(tmp_path, f"{split}/source-{index}", {"Train": [seed]})
    names = prepare_cvat(tmp_path)
    assert [len(names[split]) for split in ("train", "val", "test")] == [2, 2, 2]


@pytest.mark.parametrize("exported", [False, True])
def test_images_directly_in_split_folders_use_the_outer_assignment(tmp_path: Path, exported: bool) -> None:
    for split, seed in (("Train", 10), ("Validation", 100), ("TEST", 200)):
        source = tmp_path / "raw_data" / split
        if exported:
            write_cvat_export(source, {"Train": [seed]})
        else:
            write_image(source / "images/image.jpg", seed)
            write_label(source / "labels/image.txt", "0 0.5 0.5 0.2 0.2\n")
    names = prepare_cvat(tmp_path)
    assert [len(names[split]) for split in ("train", "val", "test")] == [1, 1, 1]


def test_identical_image_in_train_and_test_is_kept_only_in_test(tmp_path: Path) -> None:
    add_export(tmp_path, "project", {"Train": [10], "Validation": [100], "Test": [200]})
    write_cvat_export(tmp_path / "raw_data/train/task", {"Train": [200]})
    names = prepare_cvat(tmp_path)
    assert [len(names[split]) for split in ("train", "val", "test")] == [1, 1, 1]
    assert names["train"] == ["0__scene_project.jpg"]


def test_prepared_dataset_is_identical_in_different_project_folders(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    for split, value in (("train", 20), ("val", 100), ("test", 200)):
        write_image(first / f"raw_data/{split}/camera/images/frame.jpg", value)
    shutil.copytree(first / "raw_data", second / "raw_data")

    prepare(first)
    prepare(second)

    def contents(workspace: Path) -> dict[str, bytes]:
        dataset = workspace / "datasets/sample"
        return {str(path.relative_to(dataset)): path.read_bytes() for path in dataset.rglob("*") if path.is_file()}

    assert contents(first) == contents(second)
    assert all(row["source_image"].startswith("raw_data/") for row in read_manifest(first / "datasets/sample"))
