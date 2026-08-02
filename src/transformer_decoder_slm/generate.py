"""Autoregressive text generation."""

from __future__ import annotations

from .config import Config
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from .tokenizer import Tokenizer


@torch.inference_mode()
def generate_text(
    model: nn.Module,
    tokenizer: "Tokenizer",
    prompt: str,
    generation_length: int,
    temperature: float,
    device: torch.device,
    top_k: int = 40,
) -> str:
    """Generate text without KV caching."""

    if generation_length < 0:
        msg = "generation_length must be non-negative"
        raise ValueError(msg)
    if temperature <= 0:
        msg = "temperature must be greater than 0"
        raise ValueError(msg)

    context_length = Config().context_length
    
    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        msg = "prompt must produce at least one token"
        raise ValueError(msg)

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
        next_token = sample_next_token(
            logits=logits, temperature=temperature, top_k=top_k
        )
        generated_token_ids.append(next_token)

    return tokenizer.decode(generated_token_ids)


def sample_next_token(logits: torch.Tensor, temperature: float, top_k: int):
    next_token_logits = logits[:, -1, :] / temperature
    top_values, _ = torch.topk(next_token_logits, k=top_k, dim=-1)
    cutoff = top_values[:, -1].unsqueeze(-1)
    filtered_logits = next_token_logits.masked_fill(
        next_token_logits < cutoff,
        float("-inf"),
    )
    probabilities = torch.softmax(filtered_logits, dim=-1)
    next_token_id = torch.multinomial(probabilities, num_samples=1).item()
    return int(next_token_id)
