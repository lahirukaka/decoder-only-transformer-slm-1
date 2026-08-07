"""Training and checkpoint helpers."""

from __future__ import annotations

from collections import Counter
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader

from .gpu import check_gpu_thermal_and_rest
from torch.optim.lr_scheduler import SequentialLR

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
    global_steps: int = 0


def create_optimizer(model: nn.Module, config: Config) -> Optimizer:
    """Create the AdamW optimizer."""

    return AdamW(
        model.parameters(),
        lr=config.peak_learning_rate,
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
    scheduler: SequentialLR,
    global_step_counter: Counter,
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
        step_optimizer_and_scheduler(
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )

        token_count = targets.numel()
        total_loss += loss.detach() * token_count
        total_tokens += token_count
        global_step_counter["batch"] += 1

        if batch_idx % 50 == 0:
            check_gpu_thermal_and_rest(max_temp_threshold=82, cooldown_seconds=10)

    return (total_loss / total_tokens).item()


def step_optimizer_and_scheduler(
    optimizer: Optimizer,
    scheduler: SequentialLR,
    scaler,
) -> None:
    """Advance the scheduler only when AMP performs a real optimizer step."""

    scale_before_step = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()

    if scaler.get_scale() >= scale_before_step:
        scheduler.step()


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


def save_loss_log(
    loss_log_path: Path,
    epoch: int,
    train_loss: float,
    validation_loss: float,
    learning_rate: float,
    global_steps: int,
) -> None:
    """Persist one epoch's loss data as a JSON array entry."""

    existing_entries: list[dict[str, float | int]] = []
    if loss_log_path.exists():
        payload = json.loads(loss_log_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("loss log must contain a JSON array")
        existing_entries = [entry for entry in payload if isinstance(entry, dict)]

    new_entry: dict[str, float | int] = {
        "epoch": epoch,
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "learning_rate": learning_rate,
        "global_steps": global_steps,
    }

    updated = False
    for index, entry in enumerate(existing_entries):
        if entry.get("epoch") == epoch:
            existing_entries[index] = new_entry
            existing_entries = existing_entries[: index + 1]
            updated = True
            break

    if not updated:
        existing_entries.append(new_entry)

    existing_entries.sort(key=lambda entry: int(entry["epoch"]))
    loss_log_path.parent.mkdir(parents=True, exist_ok=True)
    loss_log_path.write_text(
        json.dumps(existing_entries, indent=2),
        encoding="utf-8",
    )


def load_last_global_steps(loss_log_path: Path) -> int:
    """Return the latest persisted global step count from the loss log."""

    if not loss_log_path.exists():
        return 0

    payload = json.loads(loss_log_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("loss log must contain a JSON array")

    existing_entries = [entry for entry in payload if isinstance(entry, dict)]
    if not existing_entries:
        return 0

    latest_entry = max(
        existing_entries,
        key=lambda entry: int(entry.get("epoch", 0)),
    )
    global_steps = latest_entry.get("global_steps", 0)
    if not isinstance(global_steps, int):
        global_steps = int(global_steps)
    return global_steps


def build_checkpoint_payload(
    epoch: int,
    model: "DecoderOnlyTransformer",
    optimizer: Optimizer,
    train_loss: float,
    validation_loss: float,
    best_validation_loss: float,
    tokenizer: "Tokenizer",
    config: Config,
    scheduler: SequentialLR,
) -> dict[str, object]:
    """Create a checkpoint payload."""

    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
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
            "normalization_type": config.normalization_type,
        },
        "tokenizer_metadata": {
            "resources_directory": str(config.tokenizer_resources_directory),
            "vocabulary_size": tokenizer.vocabulary_size,
            "fingerprint": tokenizer.fingerprint,
        },
    }

    if config.pre_norm:
        payload["model_config"]["normalization_placement"] = "pre_norm"
    if config.rope:
        payload["model_config"]["position_encoding"] = "rope"
        payload["model_config"]["rope_base"] = 10_000.0

    return payload


def save_checkpoint(checkpoint_path: Path, payload: dict[str, object]) -> None:
    """Save a checkpoint to disk."""

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: SequentialLR,
    tokenizer_vocabulary_size: int,
    tokenizer_fingerprint: str,
    device: torch.device,
    loss_log_path: Path | None = None,
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
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    starting_epoch = int(checkpoint["epoch"]) + 1
    best_validation_loss = float(checkpoint["best_validation_loss"])
    global_steps = (
        load_last_global_steps(loss_log_path) if loss_log_path is not None else 0
    )
    return TrainState(
        starting_epoch=starting_epoch,
        best_validation_loss=best_validation_loss,
        global_steps=global_steps,
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
    scheduler: SequentialLR,
    train_state: TrainState | None = None,
) -> TrainState:
    """Train the model and save checkpoints."""

    state = train_state or TrainState()
    latest_checkpoint_path = config.checkpoint_directory / "latest.pt"
    best_checkpoint_path = config.checkpoint_directory / "best.pt"
    loss_log_path = config.checkpoint_directory / "loss.json"

    print("Start training...")
    print(
        f"From epoch {state.starting_epoch if state.starting_epoch else 0} to {config.epoch_count}..."
    )

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    global_step_counter = Counter(batch=state.global_steps)

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
            scheduler=scheduler,
            global_step_counter=global_step_counter,
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
            scheduler=scheduler,
        )
        save_checkpoint(latest_checkpoint_path, payload)
        save_loss_log(
            loss_log_path=loss_log_path,
            epoch=epoch + 1,
            train_loss=train_loss,
            validation_loss=validation_loss,
            global_steps=global_step_counter["batch"],
            learning_rate=float(scheduler.get_last_lr()[0]),
        )

        if validation_loss <= state.best_validation_loss:
            save_checkpoint(best_checkpoint_path, payload)

    return state
