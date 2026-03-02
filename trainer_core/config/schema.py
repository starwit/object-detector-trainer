from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FinetuneConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = False
    weights: str | None = None
    lr: float | None = None
    epochs: int | None = None
    freeze_backbone: bool = False


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    dataset_name: str
    experiment_name: str | None = None
    custom_classes: list[str] = Field(default_factory=list)
    use_coco_classes: bool = True
    class_mapping: dict[str, Any] = Field(default_factory=dict)


class PrepareConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    val_split: float = 0.1
    test_split: float = 0.1
    augment_multiplier: int = 1
    folder_subsets: dict[str, int | float] = Field(default_factory=dict)
    auto_replay: dict[str, Any] = Field(default_factory=dict)


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    image_size: int = 640
    epochs: int = 100
    batch_size: int = 8
    finetune: FinetuneConfig = Field(default_factory=FinetuneConfig)


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    baseline_weights_path: str | None = None


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    data: DataConfig
    prepare: PrepareConfig = Field(default_factory=PrepareConfig)
    train: TrainConfig
    models: dict[str, dict[str, Any]] = Field(default_factory=dict)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @model_validator(mode="after")
    def validate_config(self) -> "AppConfig":
        if self.data.custom_classes and self.data.use_coco_classes:
            raise ValueError(
                "Both data.custom_classes and data.use_coco_classes are set. Choose one strategy."
            )
        if self.train.model not in self.models:
            available = ", ".join(sorted(self.models.keys())) or "<none>"
            raise ValueError(
                f"Unknown train.model '{self.train.model}'. Available models: {available}"
            )
        return self
