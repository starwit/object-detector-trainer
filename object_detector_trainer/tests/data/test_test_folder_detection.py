from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from object_detector_trainer.dataprep.source_ingest import check_for_test_images


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((32, 32, 3), 180, dtype=np.uint8)
    cv2.imwrite(str(path), image)


def test_check_for_test_images_requires_at_least_one_real_image(tmp_path: Path) -> None:
    test_root = tmp_path / "raw_data" / "test"

    empty_source = test_root / "empty_source"
    (empty_source / "images").mkdir(parents=True, exist_ok=True)
    assert not check_for_test_images(test_root)

    labels_only_source = test_root / "labels_only"
    (labels_only_source / "labels").mkdir(parents=True, exist_ok=True)
    (labels_only_source / "labels" / "frame.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    assert not check_for_test_images(test_root)

    _write_image(test_root / "real_source" / "images" / "frame.jpg")
    assert check_for_test_images(test_root)
