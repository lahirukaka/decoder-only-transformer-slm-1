"""Dataset helpers for next-token prediction."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class LanguageModelDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Fixed-context overlapping windows for language modeling."""

    def __init__(
        self, token_ids: list[int], context_length: int, stride: int = 1
    ) -> None:
        if context_length < 1:
            raise ValueError("context_length must be positive")
        if stride < 1:
            raise ValueError("stride must be positive")
        if len(token_ids) < context_length + 1:
            raise ValueError("not enough tokens to create at least one sample")

        self.token_ids = token_ids
        self.context_length = context_length
        self.stride = stride

    def __len__(self) -> int:
        return ((len(self.token_ids) - self.context_length - 1) // self.stride) + 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)

        start = index * self.stride
        stop = start + self.context_length

        inputs = torch.tensor(self.token_ids[start:stop], dtype=torch.long)
        targets = torch.tensor(self.token_ids[start + 1 : stop + 1], dtype=torch.long)
        return inputs, targets
