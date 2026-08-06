"""Project configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Config:
    """Configuration values for training and generation."""

    training_corpus_path: Path = Path("data/train.txt")
    validation_corpus_path: Path = Path("data/validation.txt")
    training_token_ids_path: Path = Path("data/train_token_ids.pt")
    validation_token_ids_path: Path = Path("data/validation_token_ids.pt")
    tokenizer_resources_directory: Path = Path("resources/tokenizer")
    checkpoint_directory: Path = Path("checkpoints")
    dataset_stride: int = 128
    context_length: int = 128
    batch_size: int = 128
    model_dimension: int = 256
    number_of_heads: int = 8
    number_of_decoder_blocks: int = 6
    feed_forward_dimension: int = 1024
    dropout: float = 0.1
    peak_learning_rate: float = 1e-3
    minimum_learning_rate = 3e-5
    warmup_steps = 2000
    weight_decay: float = 1e-4
    epoch_count: int = 70
    random_seed: int = 42
    gradient_clipping_norm: float = 1.0
    generation_length: int = 200
    generation_temperature: float = 0.8
    maximum_corpus_characters: int | None = None
    resume_from_checkpoint: bool = True

    # Capacity wise breaking configuration
    pre_norm: bool = True  # Use False with models before capacity-05

    # Use False with models before capacity-06
    rope: bool = True
    rms_norm: bool = True

    def with_updates(self, updates: dict[str, Any]) -> "Config":
        """Return a new config with a batch of updated values applied."""

        valid_field_names = {field.name for field in fields(self)}
        unknown_keys = sorted(set(updates) - valid_field_names)
        if unknown_keys:
            msg = f"Unknown config fields: {', '.join(unknown_keys)}"
            raise ValueError(msg)

        merged_values = asdict(self)
        merged_values.update(updates)
        return Config(**merged_values)

    def __post_init__(self) -> None:
        if self.model_dimension % self.number_of_heads != 0:
            msg = "model_dimension must be divisible by number_of_heads"
            raise ValueError(msg)
        if self.context_length < 2:
            msg = "context_length must be at least 2"
            raise ValueError(msg)
        if self.batch_size < 1:
            msg = "batch_size must be at least 1"
            raise ValueError(msg)
        if self.number_of_decoder_blocks < 1:
            msg = "number_of_decoder_blocks must be at least 1"
            raise ValueError(msg)
        if self.feed_forward_dimension < self.model_dimension:
            msg = "feed_forward_dimension must be at least model_dimension"
            raise ValueError(msg)
        if self.generation_temperature <= 0:
            msg = "generation_temperature must be greater than 0"
            raise ValueError(msg)
