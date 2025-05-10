import torch
import torch.nn as nn
from metamon.data.team_prediction.vocabulary import Vocabulary


class TeamTransformer(nn.Module):
    """
    A simple Transformer encoder model for team prediction.

    Embeddings:
        - token embedding (vocab_size x d_model)
        - type embedding (type_vocab_size x d_model)
        - position embedding (max_seq_len x d_model)

    Args:
        max_seq_len (int): Maximum sequence length for positional embeddings
        d_model (int): Embedding dimension (default: 512)
        nhead (int): Number of attention heads (default: 8)
        num_layers (int): Number of Transformer encoder layers (default: 6)
        dim_feedforward (int): Inner dimension of feedforward networks (default: 2048)
        dropout (float): Dropout probability (default: 0.1)
    """

    def __init__(
        self,
        max_seq_len: int = 20,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Load vocabulary to determine sizes
        self.vocab = Vocabulary()
        vocab_size = len(self.vocab.tokenizer)
        type_vocab_size = max(self.vocab.type_ids.values()) + 1
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.type_embedding = nn.Embedding(type_vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # Output projection to vocabulary size
        self.output_layer = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_tokens: torch.LongTensor,
        type_ids: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Forward pass of the Transformer encoder.

        Args:
            x_tokens (LongTensor): Tensor of shape (batch_size, seq_len) with token IDs
            type_ids (LongTensor): Tensor of shape (batch_size, seq_len) with type IDs

        Returns:
            logits (Tensor): Unnormalized scores of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = x_tokens.size()
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}."
            )

        # Create position IDs (batch_size, seq_len)
        position_ids = torch.arange(seq_len, device=x_tokens.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)

        # Embeddings
        token_emb = self.token_embedding(x_tokens)
        type_emb = self.type_embedding(type_ids)
        pos_emb = self.position_embedding(position_ids)

        # Combine embeddings and apply dropout
        x = token_emb + type_emb + pos_emb
        x = self.dropout(x)

        # Transformer encoder (batch_first=True)
        x = self.transformer_encoder(x)

        # Project back to vocabulary
        logits = self.output_layer(x)
        return logits


# Example usage:
# model = TeamTransformer(
#     max_seq_len=64,
#     d_model=256,
#     nhead=4,
#     num_layers=3,
# )
# x_tokens = torch.randint(0, 10000, (32, 64))
# type_ids = torch.randint(0, 7, (32, 64))
# logits = model(x_tokens, type_ids)
# print(logits.shape)  # (32, 64, 10000)
