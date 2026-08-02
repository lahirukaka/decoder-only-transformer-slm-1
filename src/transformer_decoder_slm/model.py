import torch
from torch import nn

from .block import DecoderBlock


class DecoderOnlyTransformer(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        maximum_context_length: int,
        model_dimension: int,
        number_of_heads: int,
        number_of_decoder_blocks: int,
        feed_forward_dimension: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.maximum_context_length = maximum_context_length
        self.model_dimension = model_dimension

        self.token_embedding = nn.Embedding(vocabulary_size, model_dimension)
        self.position_embedding = nn.Embedding(maximum_context_length, model_dimension)

        self.embedding_dropout = nn.Dropout(dropout)

        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    model_dimension=model_dimension,
                    number_of_attention_heads=number_of_heads,
                    feed_forward_dimension=feed_forward_dimension,
                    dropout=dropout,
                    maximum_context_length=maximum_context_length,
                )
                for _ in range(number_of_decoder_blocks)
            ]
        )

        self.final_norm = nn.LayerNorm(model_dimension)
        self.output_projection = nn.Linear(model_dimension, vocabulary_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length = token_ids.shape

        if sequence_length > self.maximum_context_length:
            raise ValueError("input sequence length exceeds maximum_context_length")

        position_ids = torch.arange(sequence_length, device=token_ids.device)

        token_embeddings = self.token_embedding(token_ids)
        position_embeddings = self.position_embedding(position_ids)
        combined = token_embeddings + position_embeddings
        x = self.embedding_dropout(combined)

        for decoder_block in self.decoder_blocks:
            x = decoder_block(x)

        x = self.final_norm(x)
        logits = self.output_projection(x)

        return logits
