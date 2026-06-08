from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def generate_side_by_side_comparisons(
    original_model,
    retrained_model,
    test_img_dir: Path,
    output_dir: Path,
    conf_threshold: float = 0.25,
) -> None:
    side_by_side_dir = output_dir / "side_by_side_comparisons"
    side_by_side_dir.mkdir(exist_ok=True)

    for img_path in test_img_dir.glob("*"):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue

        original_results = original_model.predict(
            str(img_path),
            conf=conf_threshold,
            save=False,
            verbose=False,
        )
        retrained_results = retrained_model.predict(
            str(img_path),
            conf=conf_threshold,
            save=False,
            verbose=False,
        )

        original_img = original_results[0].plot()
        retrained_img = retrained_results[0].plot()
        comparison_img = np.hstack((original_img, retrained_img))
        save_path = side_by_side_dir / f"comparison_{img_path.name}"
        cv2.imwrite(str(save_path), comparison_img)


__all__ = ["generate_side_by_side_comparisons"]
