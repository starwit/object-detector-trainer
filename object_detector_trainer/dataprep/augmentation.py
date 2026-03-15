from __future__ import annotations

import os

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A


def _coerce_yolo_labels(labels: list[list[float]]) -> list[list[float]]:
    normalized: list[list[float]] = []
    for raw_label in labels:
        if len(raw_label) < 5:
            continue
        class_id = int(raw_label[0])
        bbox = [float(value) for value in raw_label[1:5]]
        normalized.append([float(class_id), *bbox])
    return normalized


def _build_augmenter() -> A.Compose:
    return A.Compose(
        [
            A.GaussNoise(std_range=(0.04, 0.16), mean_range=(0.0, 0.0), p=0.7),
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.5,
            ),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.RandomFog(fog_coef_range=(0.1, 0.25), alpha_coef=0.08, p=0.3),
            A.RandomRain(
                slant_range=(-10, 10),
                drop_length=12,
                drop_width=1,
                blur_value=3,
                brightness_coefficient=0.9,
                p=0.2,
            ),
            A.ImageCompression(quality_range=(70, 99), p=0.7),
            A.Affine(
                scale=(0.8, 1.2),
                translate_percent=(-0.2, 0.2),
                rotate=(-5, 5),
                shear=(-5, 5),
                fit_output=False,
                p=0.3,
            ),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.1,
            clip=True,
        ),
    )


class YOLOAugmenter:
    def __init__(self, multiplier: int = 1):
        self.multiplier = max(1, int(multiplier))
        self.augmenter = _build_augmenter()

    def augment_image_and_labels(self, image, labels):
        normalized_labels = _coerce_yolo_labels(labels)
        results = [(image, normalized_labels)]

        bboxes = [label[1:5] for label in normalized_labels]
        class_labels = [int(label[0]) for label in normalized_labels]

        for _ in range(self.multiplier - 1):
            transformed = self.augmenter(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels,
            )
            aug_labels = [
                [float(class_id), *[float(value) for value in bbox]]
                for class_id, bbox in zip(
                    transformed["class_labels"],
                    transformed["bboxes"],
                    strict=False,
                )
            ]
            results.append((transformed["image"], aug_labels))

        return results
