from __future__ import annotations

from typing import Any

import yaml


def _deep_set(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = [part for part in dotted_key.split(".") if part]
    if not parts:
        raise ValueError(f"Invalid override key: {dotted_key!r}")

    cursor = target
    for key in parts[:-1]:
        child = cursor.get(key)
        if not isinstance(child, dict):
            child = {}
            cursor[key] = child
        cursor = child
    cursor[parts[-1]] = value


def apply_set_overrides(raw_config: dict[str, Any], set_overrides: list[str] | None) -> dict[str, Any]:
    merged = dict(raw_config)
    for item in set_overrides or []:
        if "=" not in item:
            raise ValueError(
                f"Invalid --set override {item!r}. Expected format: section.key=value"
            )
        dotted_key, raw_value = item.split("=", 1)
        value = yaml.safe_load(raw_value)
        _deep_set(merged, dotted_key.strip(), value)
    return merged

