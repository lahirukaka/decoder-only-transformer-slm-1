import torch
from torch import nn
import math


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        model_dimension: int,
        number_of_heads: int,
        dropout: float,
        maximum_context_length: int,
    ) -> None:
        super().__init__()

        if model_dimension % number_of_heads != 0:
            raise ValueError("model_dimension must be divisible by number_of_heads")

        self.model_dimension = model_dimension
        self.number_of_heads = number_of_heads
        self.dropout = dropout
        self.maximum_context_length = maximum_context_length
        self.head_dimension = model_dimension // number_of_heads

        # for efficency, project for D and then split into heads
        self.query_projection = nn.Linear(model_dimension, model_dimension)
        self.key_projection = nn.Linear(model_dimension, model_dimension)
        self.value_projection = nn.Linear(model_dimension, model_dimension)
        self.output_projection = nn.Linear(model_dimension, model_dimension)

        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

        causal_mask = torch.tril(
            torch.ones(
                maximum_context_length,
                maximum_context_length,
                dtype=torch.bool,
            )
        )
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, model_dimension = inputs.shape

        # projection
        queries = self.query_projection(inputs)
        keys = self.key_projection(inputs)
        values = self.value_projection(inputs)

        queries = self._split_into_heads(queries, batch_size, sequence_length)
        keys = self._split_into_heads(keys, batch_size, sequence_length)
        values = self._split_into_heads(values, batch_size, sequence_length)

        # calculate raw attention scores
        # output -> [B, H, T, T]
        attention_scores = queries @ keys.transpose(-2, -1)
        # scale
        attention_scores = attention_scores / math.sqrt(self.head_dimension)
        # apply causal mask
        mask = self.causal_mask[
            :sequence_length,
            :sequence_length,
        ]
        attention_scores = attention_scores.masked_fill(~mask, float("-inf"))

        # calculate attention weights
        attention_weights = torch.softmax(attention_scores, dim=-1)  # across last T
        attention_weights = self.attention_dropout(attention_weights)

        # calculate context matrix
        # output -> [B, H, T, d_head]
        context: torch.Tensor = attention_weights @ values
        # merging heads back
        context = context.transpose(1, 2).contiguous()
        # output -> [B, T, D]
        context = self._merge_heads(context, batch_size, sequence_length)
        # output projection
        output = self.output_projection(context)
        output = self.output_dropout(output)

        return output

    # output -> [B, H, T, d_head]
    def _split_into_heads(
        self, projection: torch.Tensor, b_size, t_length
    ) -> torch.Tensor:
        return projection.view(
            b_size,
            t_length,
            self.number_of_heads,
            self.head_dimension,
        ).transpose(1, 2)

    # output -> [B, T, D]
    def _merge_heads(self, matrix: torch.Tensor, b_size, t_length) -> torch.Tensor:
        return matrix.view(b_size, t_length, self.model_dimension)
