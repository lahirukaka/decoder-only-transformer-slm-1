from torch import nn
import torch

from .config import Config


class _RMSNorm(nn.Module):
    def __init__(self, dimention: int, epsilon: float = 1e-6) -> None:
        super().__init__()

        self.epsilon = epsilon  # avoid division by zero
        self.weight = nn.Parameter(torch.ones(dimention))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mean_square = inputs.pow(2).mean(
            dim=-1,
            keepdim=True,
        )
        normalized = inputs * torch.rsqrt(mean_square + self.epsilon)
        return self.weight * normalized


class Normalization(nn.Module):
    def __init__(self, dimention: int, config: Config) -> None:
        super().__init__()

        if config.normalization_type == "rmsnorm":
            self.norm = _RMSNorm(dimention)
        else:
            self.norm = nn.LayerNorm(dimention)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.norm(inputs)
