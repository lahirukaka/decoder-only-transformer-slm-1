import torch
from torch import nn


class RotaryPositionEmbedding(nn.Module):
    def __init__(
        self, head_dimention: int, maximum_context_length: int, base: float = 10_000.0
    ):
        super().__init__()

        if head_dimention % 2 != 0:
            raise ValueError("Head dimention must be even for RoPE")

        pair_indicies = torch.arange(0, head_dimention, 2, dtype=torch.float32)
        inverse_frequencies = 1.0 / (base ** (pair_indicies / head_dimention))
        positions = torch.arange(maximum_context_length, dtype=torch.float32)
        angles = positions[:, None] * inverse_frequencies[None, :]

        self.register_buffer("cosines", angles.cos(), persistent=False)
        self.register_buffer("sines", angles.sin(), persistent=False)

    def _rotate(
        self,
        tensor: torch.Tensor,
        sequence_length: int,
    ):
        even = tensor[..., 0::2]
        odd = tensor[..., 1::2]

        cosines = self.cosines[:sequence_length][None, None, :, :]
        sines = self.sines[:sequence_length][None, None, :, :]

        rotated_even = even * cosines - odd * sines
        rotated_odd = even * sines + odd * cosines

        rotated = torch.stack(
            (rotated_even, rotated_odd),
            dim=-1,
        )

        return rotated.flatten(-2)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence_length = queries.size(-2)

        if keys.size(-2) != sequence_length:
            raise ValueError("Queries and keys must have the same sequence length.")

        if sequence_length > self.cosines.size(0):
            raise ValueError("Sequence length exceeds RoPE maximum context length.")

        return (
            self._rotate(queries, sequence_length),
            self._rotate(keys, sequence_length),
        )
