"""Training entry point."""

from __future__ import annotations

import importlib
import random
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import Config
from .dataset import LanguageModelDataset
from .generate import generate_text
from .model import DecoderOnlyTransformer
from .train import TrainState, create_optimizer, load_checkpoint, train_model
from .tokenizer import save_token_ids, load_token_ids, encode_parallel


def set_random_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_tokenizer(resources_directory: Path) -> Any:
    """Instantiate the user-implemented tokenizer."""

    tokenizer_module = importlib.import_module("transformer_decoder_slm.tokenizer")
    tokenizer_class = getattr(tokenizer_module, "Tokenizer", None)
    if tokenizer_class is None:
        msg = "Tokenizer is not implemented yet in transformer_decoder_slm/tokenizer.py"
        raise RuntimeError(msg)
    return tokenizer_class(resources_directory)


def read_text(path: Path, maximum_characters: int | None = None) -> str:
    """Read a corpus split from disk."""

    text = path.read_text(encoding="utf-8")
    if maximum_characters is not None:
        text = text[:maximum_characters]
    return text


def load_or_create_token_ids(
    text_path: Path,
    token_ids_path: Path,
    tokenizer: Any,
    maximum_characters: int | None = None,
) -> list[int]:
    """Load cached token IDs for one split or regenerate them."""

    if token_ids_path.exists():
        print(f"Loading cached token IDs from {token_ids_path}...")
        try:
            return load_token_ids(
                token_ids_path,
                tokenizer.fingerprint,
            )
        except ValueError as error:
            print(f"{error} Regenerating token IDs for {text_path}...")

    print(f"Generating token IDs from {text_path}...")
    text = read_text(text_path, maximum_characters=maximum_characters)
    token_ids = encode_parallel(text, tokenizer)
    save_token_ids(
        token_ids,
        token_ids_path,
        tokenizer.fingerprint,
    )
    return token_ids


def main() -> None:
    """Run training and sample generation."""

    config = Config()
    set_random_seed(config.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = create_tokenizer(config.tokenizer_resources_directory)
    train_token_ids = load_or_create_token_ids(
        text_path=config.training_corpus_path,
        token_ids_path=config.training_token_ids_path,
        tokenizer=tokenizer,
        maximum_characters=config.maximum_corpus_characters,
    )
    validation_token_ids = load_or_create_token_ids(
        text_path=config.validation_corpus_path,
        token_ids_path=config.validation_token_ids_path,
        tokenizer=tokenizer,
        maximum_characters=config.maximum_corpus_characters,
    )

    train_dataset = LanguageModelDataset(
        token_ids=train_token_ids,
        context_length=config.context_length,
        stride=config.dataset_stride,
    )
    validation_dataset = LanguageModelDataset(
        token_ids=validation_token_ids,
        context_length=config.context_length,
        stride=config.dataset_stride,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    model = DecoderOnlyTransformer(
        vocabulary_size=tokenizer.vocabulary_size,
        maximum_context_length=config.context_length,
        model_dimension=config.model_dimension,
        number_of_heads=config.number_of_heads,
        number_of_decoder_blocks=config.number_of_decoder_blocks,
        feed_forward_dimension=config.feed_forward_dimension,
        dropout=config.dropout,
    ).to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, config)

    train_state = TrainState()
    if config.resume_from_checkpoint:
        latest_checkpoint_path = config.checkpoint_directory / "best.pt"
        train_state = load_checkpoint(
            checkpoint_path=latest_checkpoint_path,
            model=model,
            optimizer=optimizer,
            tokenizer_vocabulary_size=tokenizer.vocabulary_size,
            tokenizer_fingerprint=tokenizer.fingerprint,
            device=device,
        )

    train_model(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        optimizer=optimizer,
        loss_function=loss_function,
        device=device,
        config=config,
        train_state=train_state,
    )

    best_checkpoint_path = config.checkpoint_directory / "best.pt"
    if best_checkpoint_path.exists():
        load_checkpoint(
            checkpoint_path=best_checkpoint_path,
            model=model,
            optimizer=optimizer,
            tokenizer_vocabulary_size=tokenizer.vocabulary_size,
            tokenizer_fingerprint=tokenizer.fingerprint,
            device=device,
        )

    # sample_text = generate_text(
    #     model=model,
    #     tokenizer=tokenizer,
    #     prompt="Once upon a time",
    #     generation_length=config.generation_length,
    #     temperature=config.generation_temperature,
    #     context_length=config.context_length,
    #     device=device,
    # )
    # print(sample_text)


if __name__ == "__main__":
    main()
