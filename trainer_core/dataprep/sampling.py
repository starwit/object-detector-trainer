from __future__ import annotations

import random
from typing import Dict

from trainer_core.dataprep.types import ImageLabelPair


def apply_subset_sampling(
    folder_name: str,
    pairs: list[ImageLabelPair],
    subset_ratio: int | float | str,
) -> list[ImageLabelPair]:
    """Apply per-folder subsampling or oversampling hints."""
    if isinstance(subset_ratio, str) and subset_ratio.isdigit():
        count = int(subset_ratio)
        if count == 1:
            subset_ratio = 1.0
        elif count >= 2:
            pairs_copy = pairs.copy()
            random.shuffle(pairs_copy)
            take = min(count, len(pairs_copy))
            print(f"Folder '{folder_name}': Using exactly {take} images (absolute count)")
            return pairs_copy[:take]

    if isinstance(subset_ratio, int):
        count = int(subset_ratio)
        if count == 1:
            subset_ratio = 1.0
        elif count >= 2:
            pairs_copy = pairs.copy()
            random.shuffle(pairs_copy)
            take = min(count, len(pairs_copy))
            print(f"Folder '{folder_name}': Using exactly {take} images (absolute count)")
            return pairs_copy[:take]

    if isinstance(subset_ratio, float) and subset_ratio > 1:
        print(
            f"Folder '{folder_name}': Oversampling requested "
            f"({subset_ratio*100:.1f}%), applied to training split only"
        )
        return pairs

    if isinstance(subset_ratio, float) and 0 < subset_ratio < 1:
        original_count = len(pairs)
        pairs_copy = pairs.copy()
        random.shuffle(pairs_copy)
        subset_count = int(len(pairs) * subset_ratio)
        print(
            f"Folder '{folder_name}': Using {subset_count}/"
            f"{original_count} images ({subset_ratio*100:.1f}%)"
        )
        return pairs_copy[:subset_count]

    if subset_ratio == 1 or subset_ratio == 1.0:
        print(f"Folder '{folder_name}': Using all {len(pairs)} images (100%)")
        return pairs

    print(
        f"Warning: Invalid subset ratio {subset_ratio} "
        f"for folder '{folder_name}'. Must be > 0."
    )
    return pairs


def oversample_train_pairs(
    train_pairs: list[ImageLabelPair],
    folder_subsets: Dict[str, int | float],
) -> list[ImageLabelPair]:
    if not folder_subsets:
        return train_pairs

    extended_pairs = list(train_pairs)
    for folder_name, ratio in folder_subsets.items():
        if not (isinstance(ratio, float) and ratio > 1.0):
            continue

        folder_pairs = [pair for pair in train_pairs if pair.scene == folder_name]
        if not folder_pairs:
            continue

        oversample_factor = int(ratio)
        remainder = ratio - oversample_factor

        duplicates: list[ImageLabelPair] = []
        if oversample_factor > 1:
            duplicates.extend(folder_pairs * (oversample_factor - 1))

        if remainder > 0:
            folder_copy = folder_pairs.copy()
            random.shuffle(folder_copy)
            partial_count = int(len(folder_pairs) * remainder)
            duplicates.extend(folder_copy[:partial_count])

        if duplicates:
            print(
                f"Folder '{folder_name}': Added {len(duplicates)} extra training copies "
                f"({ratio*100:.1f}%)"
            )
            extended_pairs.extend(duplicates)

    return extended_pairs


def resolve_folder_subsets(
    folder_subsets: Dict[str, int | float] | None,
    cli_overrides,
) -> Dict[str, int | float]:
    resolved = dict(folder_subsets or {})
    if not cli_overrides:
        return resolved

    print("Overriding folder subset configuration with command-line arguments:")
    for folder_name, ratio_str in cli_overrides:
        try:
            ratio = float(ratio_str)
            if ratio <= 0:
                print(f"  Warning: Invalid ratio {ratio} for {folder_name}. Must be > 0.")
                continue
            resolved[folder_name] = ratio
            suffix = " (oversampling)" if ratio > 1 else ""
            print(f"  {folder_name}: {ratio*100:.1f}%{suffix}")
        except ValueError:
            print(f"  Warning: Invalid ratio '{ratio_str}' for {folder_name}. Must be a number.")
    return resolved


__all__ = ["apply_subset_sampling", "oversample_train_pairs", "resolve_folder_subsets"]
