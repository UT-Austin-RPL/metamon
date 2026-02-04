"""
TeamPredictionModel: High-level wrapper for team prediction models.

Provides a clean interface for training and inference:
- forward(): Pass-through to underlying model (training)
- predict(): TeamSet -> TeamSet inference with iterative decoding
- save_checkpoint() / load_checkpoint(): Checkpointing
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from metamon.backend.team_prediction.team import TeamSet, Team2Seq
from metamon.backend.team_prediction.vocabulary import Vocabulary, get_vocab
from metamon.backend.team_prediction.iterative_decoder import (
    Decoder,
    IterativeTeamDecoder,
    IterativeDecodingStats,
    OneShotDecoder,
)


###############################################################################
# Neural Network Architectures
###############################################################################


class TeamTransformer(nn.Module):
    """
    A simple Transformer encoder model for team prediction.

    Embeddings:
        - token embedding (vocab_size x d_model)
        - type embedding (type_vocab_size x d_model)
        - position embedding (seq_len x d_model)

    Args:
        include_stats (bool): Whether sequences include nature/EVs/IVs (determines seq length)
        d_model (int): Embedding dimension (default: 512)
        nhead (int): Number of attention heads (default: 8)
        num_layers (int): Number of Transformer encoder layers (default: 6)
        dim_feedforward (int): Inner dimension of feedforward networks (default: 2048)
        dropout (float): Dropout probability (default: 0.1)
        norm_first (bool): Whether to apply normalization before the attention layer (default: True)
    """

    def __init__(
        self,
        include_stats: bool = False,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        norm_first: bool = True,
    ):
        super().__init__()
        self.include_stats = include_stats
        self.seq_len = Team2Seq.seq_len(include_stats)
        self.vocab = Vocabulary()
        vocab_size = len(self.vocab.tokenizer)
        type_vocab_size = max(self.vocab.type_ids.values()) + 1
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.type_embedding = nn.Embedding(type_vocab_size, d_model)
        self.position_embedding = nn.Embedding(self.seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

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
        if seq_len > self.seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds expected {self.seq_len}."
            )

        position_ids = torch.arange(seq_len, device=x_tokens.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)
        token_emb = self.token_embedding(x_tokens)
        type_emb = self.type_embedding(type_ids)
        pos_emb = self.position_embedding(position_ids)
        x = token_emb + type_emb + pos_emb
        x = self.dropout(x)
        x = self.transformer_encoder(x)
        logits = self.output_layer(x)
        return logits


###############################################################################
# High-Level Model Wrapper
###############################################################################


class TeamPredictionModel:
    """
    High-level wrapper for team prediction models.

    Handles the conversion between TeamSet objects and model tensors,
    providing a clean interface for both training and inference.

    Args:
        model_class: The nn.Module class to use (e.g., TeamTransformer)
        model_kwargs: Keyword arguments passed to model_class constructor
        iterative_decoder_kwargs: Keyword arguments for IterativeTeamDecoder
            (num_iterations, temperature, top_p, mask_schedule, deterministic)
        oneshot_decoder_kwargs: Keyword arguments for OneShotDecoder
            (temperature, top_p, deterministic)
        include_stats: Whether sequences include nature/EVs/IVs
        device: Device to run model on (default: auto-detect)
    """

    def __init__(
        self,
        model_class: Type[nn.Module] = TeamTransformer,
        model_kwargs: Optional[Dict[str, Any]] = None,
        iterative_decoder_kwargs: Optional[Dict[str, Any]] = None,
        oneshot_decoder_kwargs: Optional[Dict[str, Any]] = None,
        include_stats: bool = False,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.model_class = model_class
        self.model_kwargs = model_kwargs or {}
        self.iterative_decoder_kwargs = iterative_decoder_kwargs or {}
        self.oneshot_decoder_kwargs = oneshot_decoder_kwargs or {}
        self.include_stats = include_stats

        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize components
        self.vocab = get_vocab()
        self.t2s = Team2Seq(include_stats=include_stats)

        # Create model (pass include_stats for seq length calculation)
        model_kwargs_with_stats = {"include_stats": include_stats, **self.model_kwargs}
        self._model = self.model_class(**model_kwargs_with_stats).to(self.device)

        # Create decoders (lazy - only when needed)
        self._iterative_decoder: Optional[IterativeTeamDecoder] = None
        self._oneshot_decoder: Optional[OneShotDecoder] = None

    @property
    def model(self) -> nn.Module:
        """The underlying nn.Module."""
        return self._model

    @property
    def iterative_decoder(self) -> IterativeTeamDecoder:
        """The iterative decoder (created lazily)."""
        if self._iterative_decoder is None:
            self._iterative_decoder = IterativeTeamDecoder(
                model=self._model,
                include_stats=self.include_stats,
                **self.iterative_decoder_kwargs,
            )
        return self._iterative_decoder

    @property
    def oneshot_decoder(self) -> OneShotDecoder:
        """The one-shot decoder (created lazily)."""
        if self._oneshot_decoder is None:
            self._oneshot_decoder = OneShotDecoder(
                model=self._model,
                include_stats=self.include_stats,
                **self.oneshot_decoder_kwargs,
            )
        return self._oneshot_decoder

    def forward(
        self,
        x_tokens: torch.Tensor,
        type_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the underlying model.

        Used during training to get logits for loss computation.

        Args:
            x_tokens: Input token IDs (batch_size, seq_len)
            type_ids: Type IDs for each position (batch_size, seq_len)

        Returns:
            logits: Unnormalized scores (batch_size, seq_len, vocab_size)
        """
        return self._model(x_tokens, type_ids)

    def predict(
        self,
        team: TeamSet,
        return_stats: bool = False,
    ) -> Union[TeamSet, tuple[TeamSet, IterativeDecodingStats]]:
        """
        Predict missing attributes for a single team.

        This is the user-facing inference method. Takes a TeamSet with
        missing values (indicated by $missing_*$ tokens) and returns
        a complete TeamSet with predictions filled in.

        Args:
            team: Input TeamSet with some attributes masked as missing
            return_stats: If True, also return decoding statistics

        Returns:
            predicted_team: TeamSet with missing values filled in
            stats: (optional) IterativeDecodingStats if return_stats=True
        """
        return self.predict_batch([team], return_stats=return_stats)[0]

    def predict_batch(
        self,
        teams: List[TeamSet],
        return_stats: bool = False,
    ) -> Union[List[TeamSet], List[tuple[TeamSet, IterativeDecodingStats]]]:
        """
        Predict missing attributes for a batch of teams.

        Args:
            teams: List of input TeamSets
            return_stats: If True, also return decoding statistics per team

        Returns:
            List of predicted TeamSets (or tuples with stats if return_stats=True)
        """
        self._model.eval()

        if not teams:
            return []

        # Encode all teams
        batch_x, batch_type_ids, batch_mask = [], [], []
        for team in teams:
            x_tokens, type_ids, pred_mask = self.t2s.encode(team)
            batch_x.append(x_tokens)
            batch_type_ids.append(type_ids)
            batch_mask.append(pred_mask)

        # Stack into batches
        x_tokens = torch.stack(batch_x).to(self.device)
        type_ids = torch.stack(batch_type_ids).to(self.device)
        pred_mask = torch.stack(batch_mask).to(self.device)

        # Run iterative decoding
        with torch.no_grad():
            pred_tokens, stats = self.iterative_decoder.decode(
                x_tokens, type_ids, pred_mask, track_tokens=return_stats
            )

        # Decode back to TeamSets
        results = [self.t2s.decode(pred_tokens[i]) for i in range(len(teams))]

        if return_stats:
            # Return list of (team, stats) tuples
            # Note: stats is shared across batch, individual tracking would need changes
            return [(team, stats) for team in results]
        return results

    def iterative_forward(
        self,
        x_tokens: torch.Tensor,
        type_ids: torch.Tensor,
        pred_mask: torch.Tensor,
        track_tokens: bool = False,
    ) -> tuple[torch.Tensor, IterativeDecodingStats]:
        """
        Iterative decoding for evaluation with known (x, y) pairs.

        Unlike predict(), this works directly with tensors and is used
        during evaluation when we have pre-encoded (x, y) pairs.

        Args:
            x_tokens: Input token IDs (batch_size, seq_len)
            type_ids: Type IDs (batch_size, seq_len)
            pred_mask: Boolean mask of positions to predict (batch_size, seq_len)
            track_tokens: Whether to track tokens at each iteration for visualization

        Returns:
            pred_tokens: Predicted token IDs (batch_size, seq_len)
            stats: Decoding statistics
        """
        return self.iterative_decoder.decode(
            x_tokens, type_ids, pred_mask, track_tokens
        )

    def oneshot_forward(
        self,
        x_tokens: torch.Tensor,
        type_ids: torch.Tensor,
        pred_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        One-shot prediction with configured temperature and nucleus sampling.

        Used for comparison with iterative decoding during evaluation.

        Args:
            x_tokens: Input token IDs (batch_size, seq_len)
            type_ids: Type IDs (batch_size, seq_len)
            pred_mask: Boolean mask of positions to predict (batch_size, seq_len)

        Returns:
            pred_tokens: Predicted token IDs (batch_size, seq_len)
        """
        return self.oneshot_decoder.decode(x_tokens, type_ids, pred_mask)

    def save_checkpoint(
        self,
        path: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
            optimizer: Optional optimizer to save state
            extra_state: Optional additional state to save (e.g., step, epoch)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self._model.state_dict(),
            "model_class": self.model_class.__name__,
            "model_kwargs": self.model_kwargs,
            "iterative_decoder_kwargs": self.iterative_decoder_kwargs,
            "oneshot_decoder_kwargs": self.oneshot_decoder_kwargs,
            "include_stats": self.include_stats,
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        torch.save(checkpoint, path)

    def load_checkpoint(
        self,
        path: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint file
            optimizer: Optional optimizer to load state into
            strict: Whether to strictly enforce state_dict keys match

        Returns:
            extra_state: Any additional state that was saved, or empty dict
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self._model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Reset decoders since model changed
        self._iterative_decoder = None
        self._oneshot_decoder = None

        return checkpoint.get("extra_state", {})

    def train(self) -> "TeamPredictionModel":
        """Set model to training mode."""
        self._model.train()
        return self

    def eval(self) -> "TeamPredictionModel":
        """Set model to evaluation mode."""
        self._model.eval()
        return self

    def to(self, device: Union[str, torch.device]) -> "TeamPredictionModel":
        """Move model to device."""
        self.device = torch.device(device)
        self._model.to(self.device)
        # Reset decoders since they hold reference to model
        self._iterative_decoder = None
        self._oneshot_decoder = None
        return self

    def parameters(self):
        """Return model parameters (for optimizer)."""
        return self._model.parameters()

    def state_dict(self):
        """Return model state dict."""
        return self._model.state_dict()

    def load_state_dict(self, state_dict, strict=True):
        """Load model state dict."""
        self._model.load_state_dict(state_dict, strict=strict)
        self._iterative_decoder = None
        self._oneshot_decoder = None


def create_model(
    model_type: str = "TeamTransformer",
    d_model: int = 512,
    nhead: int = 8,
    num_layers: int = 6,
    dim_feedforward: int = 2048,
    dropout: float = 0.1,
    # Iterative decoder settings
    num_iterations: int = 8,
    iterative_temperature: float = 1.0,
    iterative_top_p: float = 0.9,
    iterative_deterministic: bool = False,
    # One-shot decoder settings
    oneshot_temperature: float = 1.0,
    oneshot_top_p: float = 0.9,
    oneshot_deterministic: bool = True,  # Argmax by default for one-shot
    include_stats: bool = False,
    device: Optional[str] = None,
) -> TeamPredictionModel:
    """
    Factory function to create a TeamPredictionModel with common defaults.

    Args:
        model_type: Model architecture ("TeamTransformer")
        d_model: Embedding dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        dim_feedforward: FFN inner dimension
        dropout: Dropout rate
        num_iterations: Number of iterative decoding steps
        iterative_temperature: Temperature for iterative decoder
        iterative_top_p: Nucleus threshold for iterative decoder
        iterative_deterministic: Use argmax in iterative decoder
        oneshot_temperature: Temperature for one-shot decoder
        oneshot_top_p: Nucleus threshold for one-shot decoder
        oneshot_deterministic: Use argmax in one-shot decoder
        include_stats: Whether to include nature/EVs/IVs (also determines seq length)
        device: Device to use

    Returns:
        Configured TeamPredictionModel
    """
    model_classes = {
        "TeamTransformer": TeamTransformer,
    }

    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}")

    model_kwargs = {
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": num_layers,
        "dim_feedforward": dim_feedforward,
        "dropout": dropout,
    }

    iterative_decoder_kwargs = {
        "num_iterations": num_iterations,
        "temperature": iterative_temperature,
        "top_p": iterative_top_p,
        "deterministic": iterative_deterministic,
    }

    oneshot_decoder_kwargs = {
        "temperature": oneshot_temperature,
        "top_p": oneshot_top_p,
        "deterministic": oneshot_deterministic,
    }

    return TeamPredictionModel(
        model_class=model_classes[model_type],
        model_kwargs=model_kwargs,
        iterative_decoder_kwargs=iterative_decoder_kwargs,
        oneshot_decoder_kwargs=oneshot_decoder_kwargs,
        include_stats=include_stats,
        device=device,
    )
