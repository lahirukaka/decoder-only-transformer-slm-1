from typing import Any

import torch
from torch import nn

from .config import Config


class FeedForward(nn.Module):
    def __init__(
        self,
        model_dimentions: int,
        feed_forward_dimentions: int,
        dropout: float,
        config: Config,
    ) -> None:
        super().__init__()

        if config.feed_forward_type == "swiglu":
            self.feed_forward = SwiGLU(
                model_dimentions=model_dimentions,
                feed_forward_dimentions=feed_forward_dimentions,
                dropout=dropout,
                config=config,
            )
        else:
            self.feed_forward = nn.Sequential(
                nn.Linear(model_dimentions, feed_forward_dimentions),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(feed_forward_dimentions, model_dimentions),
                nn.Dropout(dropout),
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.feed_forward(inputs)


class SwiGLU(nn.Module):
    def __init__(
        self,
        model_dimentions: int,
        feed_forward_dimentions: int,
        dropout: float,
        config: Config,
    ) -> None:
        super().__init__()
        self.config = config

        self.value_projection = nn.Linear(
            model_dimentions,
            feed_forward_dimentions,
        )

        self.gate_projection = nn.Linear(
            model_dimentions,
            feed_forward_dimentions,
        )

        self.output_projection = nn.Linear(
            feed_forward_dimentions,
            model_dimentions,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = self.value_projection(inputs)
        gate = nn.functional.silu(
            self.gate_projection(inputs),
        )

        hidden = values * gate
        output = self.output_projection(hidden)

        return self.dropout(output)
