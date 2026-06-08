from pathlib import Path
from typing import NamedTuple


class ImageLabelPair(NamedTuple):
    """Pair of image and label paths with scene identifier."""
    image: Path
    label: Path
    scene: str


class ProcessedFolder(NamedTuple):
    """Result of processing a folder into image/label pairs."""
    pairs: list[ImageLabelPair]
    temp_folders: list[Path]
    empty_label_count: int
