import torch
from torch import nn

from .config import Config
from .attention import CausalMultiHeadSelfAttention
from .normalization import Normalization
from .feed_forward import FeedForward


class DecoderBlock(nn.Module):
    """Post-norm decoder block scaffold."""

    def __init__(
        self,
        model_dimension: int,
        number_of_attention_heads: int,
        feed_forward_dimension: int,
        dropout: float,
        maximum_context_length: int,
        config: Config,
    ) -> None:
        super().__init__()
        self.config = config

        self.attention = CausalMultiHeadSelfAttention(
            model_dimension=model_dimension,
            number_of_heads=number_of_attention_heads,
            dropout=dropout,
            maximum_context_length=maximum_context_length,
            config=config,
        )
        self.norm_att = Normalization(model_dimension, config)

        self.feed_forward = FeedForward(
            model_dimentions=model_dimension,
            feed_forward_dimentions=feed_forward_dimension,
            dropout=dropout,
            config=config,
        )
        self.norm_ff = Normalization(model_dimension, config)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.config.pre_norm:
            return self.pre_norm_forward(inputs)
        return self.post_norm_forward(inputs)

    # Capacity (version) wise functions

    def pre_norm_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # attention layer
        inputs_norm = self.norm_att(inputs)  # pre-norm
        attention_output = self.attention(inputs_norm)
        x = attention_output + inputs  # residual

        # feed forward layer
        x_norm = self.norm_ff(x)  # pre-norm
        feed_forward_output = self.feed_forward(x_norm)
        x = feed_forward_output + x  # residual

        return x

    def post_norm_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # attention layer
        attention_output = self.attention(inputs)
        x = attention_output + inputs  # residual
        x = self.norm_att(x)

        # feed forward layer
        feed_forward_output = self.feed_forward(x)
        x = feed_forward_output + x  # residual
        x = self.norm_ff(x)

        return x
