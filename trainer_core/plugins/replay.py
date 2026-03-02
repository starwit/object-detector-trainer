from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

from trainer_core.utils.path_ops import link_or_copy


@dataclass
class Box:
    cls: int
    conf: float
    xc: float
    yc: float
    w: float
    h: float


def _read_gt_labels(label_path: Path) -> List[Box]:
    if not label_path.exists() or label_path.stat().st_size == 0:
        return []
    out: List[Box] = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            c = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:5])
            out.append(Box(c, 1.0, x, y, w, h))
    return out


def _pred_boxes_from_result(result) -> List[Box]:
    # ultralytics result.boxes exposes .cls, .conf, .xywhn
    boxes = []
    for b in getattr(result, "boxes", []) or []:
        cls_raw = b.cls.item() if hasattr(b.cls, "item") else b.cls
        conf_raw = b.conf.item() if hasattr(b.conf, "item") else b.conf
        cls_id = int(cls_raw)
        conf = float(conf_raw)
        xywhn = getattr(b, "xywhn", None)
        if xywhn is None:
            # Fallback: try xywh in pixels and approximate normalize later (not ideal)
            continue
        x, y, w, h = [float(v) for v in xywhn[0].tolist()]
        boxes.append(Box(cls_id, conf, x, y, w, h))
    return boxes


def _iou(a: Box, b: Box) -> float:
    # boxes are normalized [0..1]
    ax1, ay1 = a.xc - a.w / 2, a.yc - a.h / 2
    ax2, ay2 = a.xc + a.w / 2, a.yc + a.h / 2
    bx1, by1 = b.xc - b.w / 2, b.yc - b.h / 2
    bx2, by2 = b.xc + b.w / 2, b.yc + b.h / 2

    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = max(1e-9, area_a + area_b - inter)
    return inter / union


def _match_and_score(preds: List[Box], gts: List[Box], iou_thr: float, conf_thr: float, border_conf: float) -> Tuple[int, int, int]:
    # greedy match by IoU
    used_pred = set()
    used_gt = set()
    borderline = 0

    # sort preds high->low conf to prefer confident matches
    order = sorted(range(len(preds)), key=lambda i: preds[i].conf, reverse=True)
    for i in order:
        p = preds[i]
        best_j = -1
        best_iou = 0.0
        for j, g in enumerate(gts):
            if j in used_gt or p.cls != g.cls:
                continue
            iou = _iou(p, g)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0 and best_iou >= iou_thr:
            used_pred.add(i)
            used_gt.add(best_j)
            if p.conf < border_conf:
                borderline += 1

    fn = len(gts) - len(used_gt)
    fp = len(preds) - len(used_pred)
    return fn, fp, borderline


def _append_index(index_csv: Path, rows: List[dict]) -> None:
    index_csv.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "run_id",
        "src_image",
        "src_label",
        "score",
        "fn",
        "fp",
        "borderline",
    ]
    new_file = not index_csv.exists()
    with open(index_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if new_file:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


def mine_hard_examples(
    model,
    images_dir: Path,
    labels_dir: Path,
    max_new: int,
    iou_thr: float = 0.5,
    conf_thr: float = 0.25,
    border_conf: float = 0.35,
    seed: int = 42,
    include_empty: bool = True,
):
    """Return list of (img_path, lbl_path, score, fn, fp, borderline)."""
    rng = random.Random(seed)
    candidates: List[Tuple[Path, Path, float, int, int, int]] = []

    img_files = [p for p in images_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    # Predict one by one to keep memory bounded; users can increase speed later if needed
    for img_path in img_files:
        label_path = labels_dir / (img_path.stem + ".txt")
        gts = _read_gt_labels(label_path)
        results = model.predict(str(img_path), conf=conf_thr, save=False, verbose=False)
        if not results:
            preds = []
        else:
            preds = _pred_boxes_from_result(results[0])
        fn, fp, borderline = _match_and_score(preds, gts, iou_thr, conf_thr, border_conf)
        score = 3 * fn + 2 * fp + 1 * borderline
        # keep hard negatives (empty gt but with fp)
        if len(gts) == 0 and (fp > 0 or include_empty):
            # Add a small baseline score for truly empty frames so some get sampled
            score = max(score, 0.5)
        if score > 0:
            candidates.append((img_path, label_path, float(score), fn, fp, borderline))

    # Sort by score (desc) and sample top max_new
    candidates.sort(key=lambda t: t[2], reverse=True)
    return candidates[:max_new]


def update_replay_folder(
    candidates: List[Tuple[Path, Path, float, int, int, int]],
    run_id: str,
    dest_root: Path,
    max_total: int,
):
    images_out = dest_root / "images"
    labels_out = dest_root / "labels"
    index_csv = dest_root / "index.csv"

    # Add new items
    added_rows: List[dict] = []
    for img_path, lbl_path, score, fn, fp, borderline in candidates:
        img_name = img_path.name
        lbl_name = lbl_path.name
        out_img = images_out / img_name
        out_lbl = labels_out / lbl_name

        if not out_img.exists():
            link_or_copy(img_path, out_img)
        if not out_lbl.exists() and lbl_path.exists():
            link_or_copy(lbl_path, out_lbl)
        added_rows.append(
            {
                "run_id": run_id,
                "src_image": str(img_path),
                "src_label": str(lbl_path),
                "score": f"{score:.3f}",
                "fn": fn,
                "fp": fp,
                "borderline": borderline,
            }
        )

    if added_rows:
        _append_index(index_csv, added_rows)

    # Prune if exceeding max_total (simple oldest-first by filesystem time)
    def _sorted_by_mtime(paths: List[Path]) -> List[Path]:
        return sorted(paths, key=lambda p: p.stat().st_mtime if p.exists() else 0.0)

    imgs = list(images_out.glob("*")) if images_out.exists() else []
    if len(imgs) > max_total:
        to_remove = len(imgs) - max_total
        for p in _sorted_by_mtime(imgs)[:to_remove]:
            lbl = labels_out / (p.stem + ".txt")
            p.unlink(missing_ok=True)
            if lbl.exists():
                lbl.unlink(missing_ok=True)


def build_or_update_replay_set(
    model,
    training_path: Path,
    train_output_dir: Path,
    config: dict,
):
    """
    Mine hard examples from the last run's validation split and persist them
    as `raw_data/train/replay/` for future fine-tunes.

    This intentionally avoids using the holdout `raw_data/test` set to prevent
    contamination of the benchmark.
    """
    # Resolve config with sane defaults
    max_new = int(config.get("max_new", 200))
    max_total = int(config.get("max_total", 400))
    iou_thr = float(config.get("iou_thr", 0.5))
    conf_thr = float(config.get("conf_thr", 0.25))
    border_conf = float(config.get("border_conf", 0.35))
    include_empty = bool(config.get("include_empty", True))
    seed = int(config.get("seed", 42))
    dest = Path(config.get("dest", "raw_data/train/replay"))

    val_images = training_path / "val" / "images"
    val_labels = training_path / "val" / "labels"
    if not val_images.exists():
        print("Replay: no validation images folder found; skipping replay mining.")
        return

    print("\n[Replay] Mining hard examples from validation split …")
    cands = mine_hard_examples(
        model,
        val_images,
        val_labels,
        max_new=max_new,
        iou_thr=iou_thr,
        conf_thr=conf_thr,
        border_conf=border_conf,
        seed=seed,
        include_empty=include_empty,
    )
    if not cands:
        print("[Replay] No hard examples found (or model extremely confident).")
        return

    run_id = Path(train_output_dir).name
    update_replay_folder(cands, run_id, dest, max_total=max_total)
    print(f"[Replay] Added {len(cands)} examples to {dest} (max_total={max_total}).")
