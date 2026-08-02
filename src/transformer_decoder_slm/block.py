import torch
from torch import nn
from .attention import CausalMultiHeadSelfAttention


class DecoderBlock(nn.Module):
    """Post-norm decoder block scaffold."""

    def __init__(
        self,
        model_dimension: int,
        number_of_attention_heads: int,
        feed_forward_dimension: int,
        dropout: float,
        maximum_context_length: int,
    ) -> None:
        super().__init__()

        self.attention = CausalMultiHeadSelfAttention(
            model_dimension=model_dimension,
            number_of_heads=number_of_attention_heads,
            dropout=dropout,
            maximum_context_length=maximum_context_length,
        )
        self.norm_att = nn.LayerNorm(model_dimension)

        self.feed_forward = nn.Sequential(
            nn.Linear(model_dimension, feed_forward_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feed_forward_dimension, model_dimension),
            nn.Dropout(dropout),
        )
        self.norm_ff = nn.LayerNorm(model_dimension)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # attention layer
        attention_output = self.attention(inputs)
        x = attention_output + inputs  # residual
        x = self.norm_att(x)

        # feed forward layer
        feed_forward_output = self.feed_forward(x)
        x = feed_forward_output + x  # residual
        x = self.norm_ff(x)

        return x
