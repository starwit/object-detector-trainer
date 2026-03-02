import numpy as np


def _load_imgaug_components():
    # Compatibility shims for old imgaug on modern NumPy versions.
    # This is applied only when augmentation is used.
    if not hasattr(np, "bool"):
        np.bool = np.bool_
    if not hasattr(np, "complex"):
        np.complex = np.complex128
    if not hasattr(np, "sctypes"):
        np.sctypes = {
            "float": [np.float16, np.float32, np.float64, np.longdouble],
            "int": [np.int8, np.int16, np.int32, np.int64],
            "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
            "complex": [np.complex64, np.complex128, np.clongdouble],
        }
    import imgaug.augmenters as iaa
    from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
    return iaa, BoundingBox, BoundingBoxesOnImage


class YOLOAugmenter:
    def __init__(self, multiplier=1):
        iaa, bounding_box_cls, bboxes_on_image_cls = _load_imgaug_components()
        self.multiplier = multiplier
        self._bounding_box_cls = bounding_box_cls
        self._bboxes_on_image_cls = bboxes_on_image_cls
        self.augmenter = iaa.Sequential(
            [
                # Noise and camera simulation.
                iaa.Sometimes(0.7, iaa.AdditiveGaussianNoise(scale=(10, 50))),
                iaa.Sometimes(
                    0.5,
                    iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30)),
                ),
                iaa.Sometimes(0.3, iaa.GaussianBlur(sigma=(0.0, 3.0))),
                # Lighting and weather conditions.
                iaa.Sometimes(0.5, iaa.LinearContrast((0.5, 1.5))),
                iaa.Sometimes(0.3, iaa.Fog()),
                iaa.Sometimes(0.2, iaa.Rain(speed=(0.1, 0.3))),
                iaa.Sometimes(0.7, iaa.JpegCompression(compression=(70, 99))),
                # Geometric transforms can shift bbox placement.
                iaa.Sometimes(
                    0.3,
                    iaa.Affine(
                        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                        rotate=(-5, 5),
                        shear=(-5, 5),
                    ),
                ),
            ]
        )

    def augment_image_and_labels(self, image, labels):
        """
        Augment an image and its YOLO format labels.

        Args:
            image: numpy array of shape (H, W, C)
            labels: list of YOLO format labels [class_id, x_center, y_center, width, height]

        Returns:
            List of tuples (augmented_image, augmented_labels)
        """
        height, width = image.shape[:2]

        # Convert YOLO format to BoundingBoxesOnImage
        bbs = []
        for label in labels:
            class_id = label[0]
            x_center, y_center = label[1] * width, label[2] * height
            box_width, box_height = label[3] * width, label[4] * height
            x1 = x_center - box_width / 2
            y1 = y_center - box_height / 2
            x2 = x_center + box_width / 2
            y2 = y_center + box_height / 2
            bbs.append(self._bounding_box_cls(x1=x1, y1=y1, x2=x2, y2=y2, label=class_id))

        bbs = self._bboxes_on_image_cls(bbs, shape=image.shape)

        # Store original image and labels
        results = [(image, labels)]

        # Generate augmented versions
        for _ in range(self.multiplier - 1):
            # Augment image and bounding boxes
            aug_image, aug_bbs = self.augmenter(image=image, bounding_boxes=bbs)

            # Convert back to YOLO format
            aug_labels = []
            for bb in aug_bbs.bounding_boxes:
                x_center = (bb.x1 + bb.x2) / (2 * width)
                y_center = (bb.y1 + bb.y2) / (2 * height)
                box_width = (bb.x2 - bb.x1) / width
                box_height = (bb.y2 - bb.y1) / height

                # Clip values to [0, 1]
                x_center = np.clip(x_center, 0, 1)
                y_center = np.clip(y_center, 0, 1)
                box_width = np.clip(box_width, 0, 1)
                box_height = np.clip(box_height, 0, 1)

                aug_labels.append([bb.label, x_center, y_center, box_width, box_height])

            results.append((aug_image, aug_labels))

        return results
