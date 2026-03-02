from __future__ import annotations

import csv
from pathlib import Path

from tabulate import tabulate


def append_results_to_csv(train_output_dir, results, metadata, is_original=False):
    """
    Append test results to CSV.
    If `is_original` is True, mark the model as the original (not retrained) in the CSV.
    """
    model_type = (
        f"{metadata['experiment_name']} (original)"
        if is_original
        else metadata["experiment_name"]
    )

    reported_img_size = results.get("img_size")
    if reported_img_size is None:
        reported_img_size = metadata["image_size"]
    csv_data = {
        "MODEL": model_type,
        "img_size": reported_img_size,
        "precision": results["precision"],
        "recall": results["recall"],
        "map": results["map"],
        "map50": results["map50"],
        "fitness": results["fitness"],
        "f1_score": results["f1_score"],
        "ms_per_frame": results["ms_per_frame"],
        "val_split": metadata["split_parameters"]["val_split"],
    }
    for key, value in results.items():
        if key.startswith("scene_") and key.endswith("_fitness"):
            csv_data[key] = value

    csv_path = train_output_dir / "test_results.csv"
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_data.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(csv_data)

    per_class = results.get("per_class", {})
    if per_class:
        _append_per_class_to_csv(
            train_output_dir / "test_per_class_results.csv",
            per_class,
            model_type,
        )


def _append_per_class_to_csv(csv_path, per_class, model_name):
    """Append per-class metrics rows to a CSV file."""
    per_class_columns = ["MODEL", "CLASS", "precision", "recall", "ap50", "ap", "f1_score"]
    path = Path(csv_path)
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=per_class_columns)
        if write_header:
            writer.writeheader()
        for cls_name, cls_metrics in per_class.items():
            row = {
                "MODEL": model_name,
                "CLASS": cls_name,
            }
            key_map = {"ap50": "map50", "ap": "map"}
            for col in per_class_columns[2:]:
                src_key = key_map.get(col, col)
                val = cls_metrics.get(src_key, cls_metrics.get(col, 0.0))
                row[col] = f"{val:.4f}" if isinstance(val, (int, float)) else val
            writer.writerow(row)


def _format_per_class_table(per_class_csv_path):
    """Format per-class CSV into a readable table grouped by model."""
    with open(per_class_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return "No per-class metrics available.\n"

    from collections import OrderedDict

    models: dict[str, list[dict]] = OrderedDict()
    for row in rows:
        model_name = row["MODEL"]
        models.setdefault(model_name, []).append(row)

    metric_cols = ["precision", "recall", "ap50", "ap", "f1_score"]
    output_parts: list[str] = []

    for model_name, model_rows in models.items():
        output_parts.append(f"  Model: {model_name}")
        table_data = []
        for row in model_rows:
            table_row = [row["CLASS"]]
            for col in metric_cols:
                val = row.get(col, "-")
                try:
                    val = float(val)
                    table_row.append(f"{val:.4f}")
                except (ValueError, TypeError):
                    table_row.append(str(val))
            table_data.append(table_row)

        headers = ["CLASS"] + metric_cols
        table_str = tabulate(table_data, headers=headers, tablefmt="simple")
        output_parts.append(table_str)
        output_parts.append("")

    return "\n".join(output_parts)


def _format_merged_class_table(csv_path):
    """Format merged-class CSV into a readable table grouped by model."""
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return "No merged-class metrics available.\n"

    from collections import OrderedDict

    models: dict[str, list[dict]] = OrderedDict()
    for row in rows:
        models.setdefault(row["MODEL"], []).append(row)

    metric_cols = ["n_objects", "precision", "recall", "ap50", "ap", "f1_score"]
    output_parts: list[str] = []

    for model_name, model_rows in models.items():
        output_parts.append(f"  Model: {model_name}")
        table_data = []
        for row in model_rows:
            table_row = [f"{row['source_class']}→{row['target_class']}"]
            for col in metric_cols:
                val = row.get(col, "-")
                try:
                    fval = float(val)
                    table_row.append(f"{fval:.4f}" if col != "n_objects" else str(int(fval)))
                except (ValueError, TypeError):
                    table_row.append(str(val))
            table_data.append(table_row)

        headers = ["merged_class"] + metric_cols
        table_str = tabulate(table_data, headers=headers, tablefmt="simple")
        output_parts.append(table_str)
        output_parts.append("")

    return "\n".join(output_parts)


def create_formatted_table(csv_path, *, output_dir: Path | None = None):
    csv_path = Path(csv_path)
    target_dir = Path(output_dir) if output_dir is not None else csv_path.parent

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
        data = list(reader)

    best_values = {}
    for col_index, header in enumerate(headers[1:], start=1):
        column_values = []
        for row in data:
            if col_index >= len(row):
                continue
            raw = row[col_index].strip()
            if not raw or raw == "-":
                continue
            try:
                column_values.append(float(raw))
            except ValueError:
                continue

        if not column_values:
            best_values[header] = None
        elif header == "ms_per_frame":
            best_values[header] = min(column_values)
        else:
            best_values[header] = max(column_values)

    formatted_data = []
    for row in data:
        if not row:
            continue
        formatted_row = [row[0]]
        for col_index, header in enumerate(headers[1:], start=1):
            raw = row[col_index].strip() if col_index < len(row) else ""
            if not raw or raw == "-":
                formatted_row.append("-")
                continue
            try:
                value = float(raw)
            except ValueError:
                formatted_row.append("-")
                continue

            is_best = best_values.get(header) is not None and value == best_values[header]
            formatted_row.append(f"*{value:.4f}*" if is_best else f"{value:.4f}")
        formatted_data.append(formatted_row)

    formatted_table = tabulate(formatted_data, headers=headers, tablefmt="simple")

    descriptions = [
        "precision (higher is better): Fraction of correct positive predictions out of all positive predictions.",
        "recall (higher is better): Fraction of actual positive cases correctly identified.",
        "map (higher is better): Mean Average Precision across IoU thresholds 0.5-0.95.",
        "map50 (higher is better): Mean Average Precision at IoU threshold 0.5.",
        "fitness (higher is better): Combined metric (0.1*mAP50 + 0.9*mAP50-95).",
        "f1_score (higher is better): Harmonic mean of precision and recall.",
        "ms_per_frame (lower is better): Average time per frame in milliseconds (preprocess + inference + postprocess).",
    ]

    for header in headers:
        if header.startswith("scene_") and header.endswith("_fitness"):
            scene_name = header[6:-8]
            descriptions.append(
                f"{header} (higher is better): Fitness score for scene '{scene_name}'."
            )

    formatted_descriptions = "\n\n".join(descriptions)

    txt_path = target_dir / "results.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Overall Metrics:\n")
        f.write("=" * 80)
        f.write("\n\n")
        f.write(formatted_table)
        f.write("\n\n")
        f.write("Metric Descriptions:\n")
        f.write(formatted_descriptions)

        per_class_csv_path = target_dir / "per_class_results.csv"
        if per_class_csv_path.exists():
            f.write("\n\n")
            f.write("=" * 80)
            f.write("\nPer-Class Metrics:\n")
            f.write("=" * 80)
            f.write("\n\n")
            per_class_table = _format_per_class_table(per_class_csv_path)
            f.write(per_class_table)
            f.write("\n")
            f.write("Per-class metric descriptions:\n")
            f.write("  precision: Fraction of correct positive predictions for this class (at best-F1 confidence).\n")
            f.write("  recall:    Fraction of actual positives correctly detected for this class (at best-F1 confidence).\n")
            f.write("  ap50:      Average Precision at IoU threshold 0.5 for this class.\n")
            f.write("  ap:        Average Precision across IoU 0.5-0.95 for this class.\n")
            f.write("  f1_score:  Harmonic mean of precision and recall for this class (at best-F1 confidence).\n")

        merged_csv_path = target_dir / "merged_class_results.csv"
        if merged_csv_path.exists():
            f.write("\n\n")
            f.write("=" * 80)
            f.write("\nMerged-Class Subset Metrics:\n")
            f.write("(Evaluation on test images that originally contained the source class\n")
            f.write(" before it was merged into the target class during training)\n")
            f.write("=" * 80)
            f.write("\n\n")
            merged_table = _format_merged_class_table(merged_csv_path)
            f.write(merged_table)
            f.write("\n")
            f.write("Merged-class metric descriptions:\n")
            f.write("  merged_class: Source class -> target class it was merged into.\n")
            f.write("  n_objects:    Number of source-class annotations (objects) in the subset.\n")
            f.write("  precision:    Precision on this image subset (all classes).\n")
            f.write("  recall:       Recall on this image subset (all classes).\n")
            f.write("  ap50:         mAP@50 on this image subset (all classes).\n")
            f.write("  ap:           mAP@50-95 on this image subset (all classes).\n")
            f.write("  f1_score:     F1 score on this image subset (all classes).\n")


def write_merged_class_results(output_dir, all_model_results):
    """Write merged-class subset results to a CSV file."""
    csv_path = Path(output_dir) / "merged_class_results.csv"
    columns = ["MODEL", "source_class", "target_class", "n_objects", "precision", "recall", "ap50", "ap", "f1_score"]

    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if write_header:
            writer.writeheader()
        for model_name, res in all_model_results:
            for src_class, metrics in res.items():
                row = {
                    "MODEL": model_name,
                    "source_class": src_class,
                    "target_class": metrics["target_class"],
                    "n_objects": metrics["n_objects"],
                }
                for col in ("precision", "recall", "ap50", "ap", "f1_score"):
                    row[col] = f"{metrics[col]:.4f}"
                writer.writerow(row)


def _collect_scene_columns(path_results2: dict | None) -> list:
    if path_results2 is None:
        return []
    return [key for key in path_results2 if key.startswith("scene_") and key.endswith("_fitness")]


def _format_values(rows: list) -> list:
    formatted_data = []
    for row in rows:
        formatted_row = [row[0]]
        formatted_values = []
        for value in row[1:]:
            if isinstance(value, (int, float)):
                formatted_values.append(f"{value:.4f}")
            else:
                formatted_values.append(value)
        formatted_row.extend(formatted_values)
        formatted_data.append(formatted_row)
    return formatted_data


def _build_mean_table_rows(basic_columns, scene_columns, path_results1, path_results2, experiment_name, base_run, base_model_name):
    if base_run and path_results1 is not None:
        base_model_display_name = base_model_name if base_model_name else "YOLOv8m (base run)"
        base_row = [base_model_display_name]
        for col in basic_columns[1:]:
            base_row.append(path_results1.get(col.lower(), "-"))
        for col in scene_columns:
            base_row.append(path_results1.get(col, 0.0))

        retrained_row = [experiment_name]
        for col in basic_columns[1:]:
            retrained_row.append(path_results2.get(col.lower(), "-"))
        for col in scene_columns:
            retrained_row.append(path_results2.get(col, 0.0))
        return [base_row, retrained_row]

    retrained_row = [experiment_name]
    for col in basic_columns[1:]:
        retrained_row.append(path_results2.get(col.lower(), "-"))
    for col in scene_columns:
        retrained_row.append(path_results2.get(col, 0.0))
    return [retrained_row]


def mean_table(
    path_results1,
    path_results2,
    experiment_name,
    base_run,
    base_model_name=None,
    *,
    output_dir: Path | None = None,
):
    basic_columns = [
        "MODEL",
        "img_size",
        "precision",
        "recall",
        "map",
        "map50",
        "fitness",
        "f1_score",
        "ms_per_frame",
    ]

    scene_columns = _collect_scene_columns(path_results2)
    columns = basic_columns + scene_columns
    data = _build_mean_table_rows(
        basic_columns, scene_columns, path_results1, path_results2, experiment_name, base_run, base_model_name
    )
    formatted_data = _format_values(data)

    target_dir = Path(output_dir) if output_dir is not None else Path("results_comparison")
    target_dir.mkdir(parents=True, exist_ok=True)
    csv_path = target_dir / "results.csv"

    existing_columns: list[str] = []
    existing_rows: list[dict[str, str]] = []
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            try:
                existing_columns = next(reader)
            except StopIteration:
                existing_columns = []
            else:
                for row in reader:
                    row_dict: dict[str, str] = {}
                    for idx, col in enumerate(existing_columns):
                        if idx < len(row):
                            row_dict[col] = row[idx]
                    existing_rows.append(row_dict)

    merged_columns = list(existing_columns) if existing_columns else []
    for col in columns:
        if col not in merged_columns:
            merged_columns.append(col)
    if not merged_columns:
        merged_columns = list(columns)

    for row in formatted_data:
        row_dict: dict[str, str] = {}
        for idx, col in enumerate(columns):
            if idx < len(row):
                row_dict[col] = row[idx]
        existing_rows.append(row_dict)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(merged_columns)
        for row_dict in existing_rows:
            writer.writerow([row_dict.get(col, "") for col in merged_columns])

    per_class_csv_path = target_dir / "per_class_results.csv"
    if base_run and path_results1 is not None:
        base_display = base_model_name if base_model_name else "YOLOv8m (base run)"
        per_class_base = path_results1.get("per_class", {})
        if per_class_base:
            _append_per_class_to_csv(per_class_csv_path, per_class_base, base_display)

    per_class_retrained = path_results2.get("per_class", {}) if path_results2 else {}
    if per_class_retrained:
        _append_per_class_to_csv(per_class_csv_path, per_class_retrained, experiment_name)

    create_formatted_table(csv_path, output_dir=target_dir)


__all__ = [
    "append_results_to_csv",
    "create_formatted_table",
    "mean_table",
    "write_merged_class_results",
]
