"""Autoregressive text generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from .tokenizer import Tokenizer


@dataclass(slots=True)
class TopKPrediction:
    """One candidate token from the next-token distribution."""

    # token_id: int
    token_text: str
    probability: float


@dataclass(slots=True)
class GenerationStepTrace:
    """Trace data for one generated token."""

    # input_token_ids: list[int]
    input_text: str
    # generated_token_id: int
    generated_token_text: str
    top_k_predictions: list[TopKPrediction]


@torch.inference_mode()
def generate_text(
    model: nn.Module,
    tokenizer: "Tokenizer",
    prompt: str,
    generation_length: int,
    temperature: float,
    device: torch.device,
    top_k: int = 40,
    trace_steps: list[GenerationStepTrace] | None = None,
) -> str:
    """Generate text without KV caching."""

    if generation_length < 0:
        msg = "generation_length must be non-negative"
        raise ValueError(msg)
    if temperature <= 0:
        msg = "temperature must be greater than 0"
        raise ValueError(msg)
    if top_k < 1:
        msg = "top_k must be at least 1"
        raise ValueError(msg)

    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        msg = "prompt must produce at least one token"
        raise ValueError(msg)

    context_length = _resolve_context_length(model=model)
    model.eval()
    generated_token_ids = list(token_ids)

    for _ in range(generation_length):
        context_token_ids = generated_token_ids[-context_length:]
        input_tensor = torch.tensor(
            [context_token_ids],
            dtype=torch.long,
            device=device,
        )

        logits = model(input_tensor)
        next_token, top_k_predictions = sample_next_token(
            logits=logits,
            temperature=temperature,
            top_k=top_k,
            tokenizer=tokenizer,
        )
        generated_token_ids.append(next_token)
        if trace_steps is not None:
            trace_steps.append(
                GenerationStepTrace(
                    # input_token_ids=list(context_token_ids),
                    input_text=tokenizer.decode(context_token_ids),
                    # generated_token_id=next_token,
                    generated_token_text=tokenizer.decode([next_token]),
                    top_k_predictions=top_k_predictions,
                )
            )

    return tokenizer.decode(generated_token_ids)


def sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    tokenizer: "Tokenizer",
) -> tuple[int, list[TopKPrediction]]:
    """Sample one token and return its top-k probability trace."""

    next_token_logits = logits[:, -1, :] / temperature
    vocabulary_size = next_token_logits.shape[-1]
    effective_top_k = min(top_k, vocabulary_size)
    top_values, _ = torch.topk(next_token_logits, k=effective_top_k, dim=-1)
    cutoff = top_values[:, -1].unsqueeze(-1)
    filtered_logits = next_token_logits.masked_fill(
        next_token_logits < cutoff,
        float("-inf"),
    )
    probabilities = torch.softmax(filtered_logits, dim=-1)
    next_token_id = torch.multinomial(probabilities, num_samples=1).item()
    top_probabilities, top_token_ids = torch.topk(
        probabilities,
        k=effective_top_k,
        dim=-1,
    )
    trace_predictions = [
        TopKPrediction(
            # token_id=int(token_id),
            token_text=tokenizer.decode([int(token_id)]),
            probability=float(probability),
        )
        for probability, token_id in zip(
            top_probabilities[0].tolist(),
            top_token_ids[0].tolist(),
            strict=True,
        )
    ]
    return int(next_token_id), trace_predictions


def _resolve_context_length(model: nn.Module) -> int:
    """Read context length from the loaded model when available."""

    context_length = getattr(model, "maximum_context_length", None)
    if not isinstance(context_length, int) or context_length < 1:
        msg = "model must expose a positive integer maximum_context_length"
        raise ValueError(msg)
    return context_length
