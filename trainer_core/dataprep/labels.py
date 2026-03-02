from __future__ import annotations

from pathlib import Path


def _poly_to_bbox_row(row: list[float]) -> list[float]:
    """Convert normalized polygon row to YOLO bbox row."""
    cls, pts = int(row[0]), row[1:]
    xs, ys = pts[0::2], pts[1::2]
    x1, y1, x2, y2 = max(0, min(xs)), max(0, min(ys)), min(1, max(xs)), min(1, max(ys))
    xc, yc, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    return [cls, xc, yc, w, h]


def convert_polygons_to_bboxes_inplace(label_path: Path) -> None:
    """Rewrite YOLO polygon labels in-place as YOLO bbox labels."""
    with label_path.open("r", encoding="utf-8") as handle:
        rows = [line.strip().split() for line in handle if line.strip()]

    changed = False
    out: list[str] = []
    for row in rows:
        if len(row) > 5:
            values = _poly_to_bbox_row(list(map(float, row)))
            changed = True
        else:
            values = list(map(float, row))
        out.append(" ".join(f"{v:.6f}" if i else str(int(v)) for i, v in enumerate(values)))

    if changed:
        with label_path.open("w", encoding="utf-8") as handle:
            handle.write("\n".join(out) + "\n")


__all__ = ["convert_polygons_to_bboxes_inplace"]
