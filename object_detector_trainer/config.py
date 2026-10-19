"""The params.yaml schema and how it is loaded.

Loading order:

1. Read params.yaml.
2. Merge ``models_defaults.<backend>`` into every model of that backend.
3. Apply command-line overrides (``--set key=value``, ``--model``, ``--seed``).
4. Validate everything, including each model entry against its backend's options.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, ValidationError, model_validator

from object_detector_trainer.backends import get_backend
from object_detector_trainer.backends.common import ModelOptions
from object_detector_trainer.classes import configured_classes, final_classes


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_name: str = Field(min_length=1)
    custom_classes: list[str] = Field(default_factory=list)
    use_coco_classes: bool = False
    class_mapping: dict[str, str | list[str]] = Field(default_factory=dict)
    # Classes in the sources' data.yaml that are not trained; their labels are dropped.
    ignore_classes: list[str] = Field(default_factory=list)


class PrepareConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Used only when no validation/test set is supplied by exports or folders.
    val_split: float = Field(default=0.1, ge=0, lt=1)
    test_split: float = Field(default=0.1, ge=0, lt=1)


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    image_size: PositiveInt = 640
    epochs: PositiveInt = 100
    device: str = "auto"


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    baseline_weights_path: str | None = None


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = 42
    data: DataConfig
    prepare: PrepareConfig = Field(default_factory=PrepareConfig)
    train: TrainConfig
    models: dict[str, dict[str, Any]]
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @model_validator(mode="after")
    def check_consistency(self) -> Config:
        if self.data.custom_classes and self.data.use_coco_classes:
            raise ValueError("Set either data.custom_classes or data.use_coco_classes, not both.")
        if not self.data.custom_classes and not self.data.use_coco_classes:
            raise ValueError("Set data.custom_classes (or data.use_coco_classes: true).")
        names = [name.lower() for name in configured_classes(self.data)]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise ValueError(f"data.custom_classes lists a class more than once (case is ignored): {repeated}")
        final_classes(self.data)  # Raises if class_mapping uses an unknown class.
        ignored_and_configured = {name.lower() for name in self.data.ignore_classes} & {
            name.lower() for name in configured_classes(self.data)
        }
        if ignored_and_configured:
            raise ValueError(f"data.ignore_classes lists configured classes: {sorted(ignored_and_configured)}")

        if self.train.model not in self.models:
            available = ", ".join(sorted(self.models)) or "<none>"
            raise ValueError(f"train.model {self.train.model!r} is not in models. Available: {available}")
        for model_key in self.models:
            self.model_options(model_key)
        return self

    def model_options(self, model_key: str) -> ModelOptions:
        """Validate one model entry with the options class of its backend."""
        settings = self.models[model_key]
        try:
            backend = get_backend(settings.get("backend", ""))
        except ValueError as error:
            raise ValueError(f"models.{model_key}: {error}") from None
        try:
            options = backend.Options.model_validate(settings)
        except ValidationError as error:
            raise ValueError(f"Invalid settings in models.{model_key}:\n{error}") from None
        owned = sorted(set(options.extra_train_kwargs) & getattr(backend, "TRAINER_OWNED_ARGUMENTS", set()))
        if owned:
            raise ValueError(
                f"models.{model_key}.extra_train_kwargs cannot set {owned}; "
                "the trainer sets these from its own settings."
            )
        return options


def load_config(path: Path, overrides: list[str] | None = None) -> Config:
    """Load params.yaml, apply ``key=value`` overrides and validate."""
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"The top level of {path} must be a mapping.")

    _merge_model_defaults(raw)
    for override in overrides or []:
        _apply_override(raw, override)
    _reject_removed_settings(raw)
    return Config.model_validate(raw)


def _reject_removed_settings(raw: dict[str, Any]) -> None:
    train = raw.get("train") or {}
    if "batch_size" in train:
        raise ValueError(
            "train.batch_size was removed. Set models.<key>.batch_size instead."
        )
    if "finetune" in train:
        raise ValueError(
            "train.finetune was removed in object-detector-trainer 2.0. Add a model entry "
            "with init_weights (and freeze_backbone / extra_train_kwargs) instead. "
            "See 'Fine-tuning' in the README."
        )


def _merge_model_defaults(raw: dict[str, Any]) -> None:
    """Merge ``models_defaults.<backend>`` into each model; model values win."""
    defaults_by_backend = raw.pop("models_defaults", None) or {}
    models = raw.get("models") or {}
    for model_key, settings in models.items():
        defaults = defaults_by_backend.get(settings.get("backend"), {})
        merged = dict(defaults)
        merged.update(settings)
        models[model_key] = merged


def _apply_override(raw: dict[str, Any], override: str) -> None:
    """Apply one ``section.key=value`` override. The value is parsed as YAML."""
    if "=" not in override:
        raise ValueError(f"Invalid override {override!r}. Expected section.key=value.")
    dotted_key, raw_value = override.split("=", 1)
    keys = dotted_key.strip().split(".")

    target = raw
    for key in keys[:-1]:
        if not isinstance(target.get(key), dict):
            target[key] = {}
        target = target[key]
    target[keys[-1]] = yaml.safe_load(raw_value)
