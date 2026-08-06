from torch import nn
import torch


class RMSNorm(nn.Module):
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
