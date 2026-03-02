"""Duplicate detection smoke test (pytest).

Creates two identical images and one distinct image, runs DuplicateDetector
with reasonably strict thresholds, and verifies that duplicates are clustered
and the number of unique images is computed as expected.
"""

from pathlib import Path

import numpy as np
import cv2

from trainer_core.dataprep.find_duplicates import DuplicateDetector


def test_detect_duplicates(tmp_path: Path):
    """Detect duplicates and validate cluster size and unique count."""
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Create two identical images and one different image
    img1 = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img1, (10, 10), (90, 90), (255, 0, 0), -1)
    img2 = img1.copy()  # duplicate
    img3 = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(img3, (50, 50), 30, (0, 255, 0), -1)

    cv2.imwrite(str(images_dir / "a.jpg"), img1)
    cv2.imwrite(str(images_dir / "b.jpg"), img2)
    cv2.imwrite(str(images_dir / "c.jpg"), img3)

    detector = DuplicateDetector(phash_threshold=2, ssim_threshold=0.95)
    image_paths = list(images_dir.glob("*.jpg"))
    clusters = detector.find_duplicates(image_paths)

    # Expect a cluster containing both duplicates, and c.jpg must not join it
    membersets = [{p.name for p in clusterset} for clusterset in clusters.values()]
    dup_pair = {"a.jpg", "b.jpg"}
    assert any(dup_pair.issubset(s) for s in membersets), "a.jpg and b.jpg should be clustered together"
    assert not any("c.jpg" in s and dup_pair.issubset(s) for s in membersets), "c.jpg must not join the duplicate cluster"

    # Unique images should be 2 (one from duplicate pair + the distinct one)
    uniques = detector.get_unique_images(image_paths)
    assert len(uniques) == 2
