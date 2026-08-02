"""Training and checkpoint helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader

from transformer_decoder_slm.gpu import check_gpu_thermal_and_rest

from .config import Config
import os
import psutil

if TYPE_CHECKING:
    from .model import DecoderOnlyTransformer
    from .tokenizer import Tokenizer


@dataclass(slots=True)
class TrainState:
    """Training state that supports checkpoint resume."""

    starting_epoch: int = 0
    best_validation_loss: float = float("inf")


def create_optimizer(model: nn.Module, config: Config) -> Optimizer:
    """Create the AdamW optimizer."""

    return AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def compute_language_model_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_function: nn.Module,
) -> torch.Tensor:
    """Compute cross-entropy over all token positions."""

    vocabulary_size = logits.size(-1)
    flattened_logits = logits.reshape(-1, vocabulary_size)
    flattened_targets = targets.reshape(-1)
    return loss_function(flattened_logits, flattened_targets)


def run_training_epoch(
    model: nn.Module,
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    optimizer: Optimizer,
    loss_function: nn.Module,
    device: torch.device,
    gradient_clipping_norm: float,
    scaler,
) -> float:
    """Run one training epoch."""

    model.train()
    total_loss = 0.0
    total_tokens = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(inputs)
            loss = compute_language_model_loss(logits, targets, loss_function)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clipping_norm)
        scaler.step(optimizer)
        scaler.update()

        token_count = targets.numel()
        total_loss += loss.detach() * token_count
        total_tokens += token_count

        if batch_idx % 50 == 0:
            check_gpu_thermal_and_rest(max_temp_threshold=82, cooldown_seconds=10)

    process = psutil.Process(os.getpid())

    print(
        f"RAM: {process.memory_info().rss / 1024**3:.2f} GB | "
        f"VRAM allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB | "
        f"VRAM reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB"
    )

    return (total_loss / total_tokens).item()


@torch.inference_mode()
def run_validation_epoch(
    model: nn.Module,
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    loss_function: nn.Module,
    device: torch.device,
) -> float:
    """Run one validation epoch."""

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)
        loss = compute_language_model_loss(logits, targets, loss_function)

        token_count = targets.numel()
        total_loss += loss.item() * token_count
        total_tokens += token_count

    return total_loss / total_tokens


def build_checkpoint_payload(
    epoch: int,
    model: "DecoderOnlyTransformer",
    optimizer: Optimizer,
    train_loss: float,
    validation_loss: float,
    best_validation_loss: float,
    tokenizer: "Tokenizer",
    config: Config,
) -> dict[str, object]:
    """Create a checkpoint payload."""

    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "best_validation_loss": best_validation_loss,
        "model_config": {
            "vocabulary_size": tokenizer.vocabulary_size,
            "context_length": config.context_length,
            "model_dimension": config.model_dimension,
            "number_of_heads": config.number_of_heads,
            "number_of_decoder_blocks": config.number_of_decoder_blocks,
            "feed_forward_dimension": config.feed_forward_dimension,
            "dropout": config.dropout,
        },
        "tokenizer_metadata": {
            "resources_directory": str(config.tokenizer_resources_directory),
            "vocabulary_size": tokenizer.vocabulary_size,
            "fingerprint": tokenizer.fingerprint,
        },
    }


def save_checkpoint(checkpoint_path: Path, payload: dict[str, object]) -> None:
    """Save a checkpoint to disk."""

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    tokenizer_vocabulary_size: int,
    tokenizer_fingerprint: str,
    device: torch.device,
) -> TrainState:
    """Load a checkpoint if it exists and is compatible."""

    if not checkpoint_path.exists():
        return TrainState()

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    tokenizer_metadata = checkpoint.get("tokenizer_metadata", {})
    saved_vocabulary_size = tokenizer_metadata.get("vocabulary_size")
    if saved_vocabulary_size != tokenizer_vocabulary_size:
        msg = (
            "checkpoint tokenizer vocabulary size does not match the current tokenizer"
        )
        raise ValueError(msg)
    saved_fingerprint = tokenizer_metadata.get("fingerprint")
    if saved_fingerprint != tokenizer_fingerprint:
        msg = "checkpoint tokenizer fingerprint does not match the current tokenizer"
        raise ValueError(msg)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    starting_epoch = int(checkpoint["epoch"]) + 1
    best_validation_loss = float(checkpoint["best_validation_loss"])
    return TrainState(
        starting_epoch=starting_epoch,
        best_validation_loss=best_validation_loss,
    )


def train_model(
    model: "DecoderOnlyTransformer",
    tokenizer: "Tokenizer",
    train_dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    validation_dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    optimizer: Optimizer,
    loss_function: nn.Module,
    device: torch.device,
    config: Config,
    train_state: TrainState | None = None,
) -> TrainState:
    """Train the model and save checkpoints."""

    state = train_state or TrainState()
    latest_checkpoint_path = config.checkpoint_directory / "latest.pt"
    best_checkpoint_path = config.checkpoint_directory / "best.pt"

    print("Start training...")
    print(
        f"From epoch {state.starting_epoch if state.starting_epoch else 0} to {config.epoch_count}..."
    )

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    for epoch in range(state.starting_epoch, config.epoch_count):
        epoch_start_time = time.perf_counter()
        train_loss = run_training_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            loss_function=loss_function,
            device=device,
            gradient_clipping_norm=config.gradient_clipping_norm,
            scaler=scaler,
        )
        validation_loss = run_validation_epoch(
            model=model,
            dataloader=validation_dataloader,
            loss_function=loss_function,
            device=device,
        )
        epoch_duration_seconds = time.perf_counter() - epoch_start_time

        print(
            f"Epoch {epoch + 1}/{config.epoch_count} "
            f"- duration: {epoch_duration_seconds:.2f}s "
            f"- train_loss: {train_loss:.4f} "
            f"- validation_loss: {validation_loss:.4f}"
        )

        state.best_validation_loss = min(state.best_validation_loss, validation_loss)
        payload = build_checkpoint_payload(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            train_loss=train_loss,
            validation_loss=validation_loss,
            best_validation_loss=state.best_validation_loss,
            tokenizer=tokenizer,
            config=config,
        )
        save_checkpoint(latest_checkpoint_path, payload)

        if validation_loss <= state.best_validation_loss:
            save_checkpoint(best_checkpoint_path, payload)

    return state
