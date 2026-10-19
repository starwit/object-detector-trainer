"""Small real-image workflow fixture, with all objects retained.

Images/boxes: Ultralytics COCO8, https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip
Source archive SHA256: 54c67fe9ef88313e021ec0e92b73c200167bb0a86633e8df8658d832cca828c9.
COCO person becomes source ID 1; all other objects become source ID 0.
Explicit splits exercise ingestion only; this is not an accuracy benchmark.
"""

import shutil
from pathlib import Path

import cv2
import numpy as np


def create_sources(workspace: Path) -> None:
    source = workspace / "raw_data/camera"
    shutil.copytree(Path(__file__).parent / "data/coco8", source)
    # Intentionally separate negative source with no labels directory.
    negative = workspace / "raw_data/train/negatives/images/background.jpg"
    negative.parent.mkdir(parents=True)
    assert cv2.imwrite(str(negative), np.random.default_rng(199).integers(0, 256, (128, 256, 3), dtype=np.uint8))
    # The export also contains a different verified empty scene with an empty label.
    negative = source / "images/Test/background.jpg"
    assert cv2.imwrite(str(negative), np.random.default_rng(200).integers(0, 256, (192, 128, 3), dtype=np.uint8))
    (source / "labels/Test/background.txt").write_text("")
    with (source / "Test.txt").open("a") as handle:
        handle.write("data/images/Test/background.jpg\n")
