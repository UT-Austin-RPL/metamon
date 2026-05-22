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

##################################
## Neural Network Architectures ##
##################################


class TeamTransformer(nn.Module):
    """Transformer encoder for team prediction (token + type + position embeddings)."""

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
            activation="gelu",
            batch_first=True,
            norm_first=norm_first,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model) if norm_first else None,
        )

        self.output_layer = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    @torch.compile
    def forward(
        self,
        x_tokens: torch.LongTensor,
        type_ids: torch.LongTensor,
    ) -> torch.Tensor:
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


class LocalGlobalTeamTransformer(nn.Module):
    """
    Alternating local (per-pokemon) and global (full-team) attention.

    Local attention: Each pokemon attends to its own tokens + format token.
    Global attention: Full sequence attention over the entire team.

    The format token embedding is kept constant throughout - it provides
    context but is never updated since it never needs predicting.
    """

    NUM_POKEMON = 6

    def __init__(
        self,
        include_stats: bool = False,
        d_model: int = 512,
        nhead: int = 8,
        num_blocks: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        norm_first: bool = True,
    ):
        super().__init__()
        self.include_stats = include_stats
        self.seq_len = Team2Seq.seq_len(include_stats)
        self.attrs_per_pokemon = (
            Team2Seq.ATTRS_PER_POKEMON_WITH_STATS
            if include_stats
            else Team2Seq.ATTRS_PER_POKEMON_BASE
        )
        self.num_blocks = num_blocks
        self.vocab = Vocabulary()
        vocab_size = len(self.vocab.tokenizer)
        type_vocab_size = max(self.vocab.type_ids.values()) + 1
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.type_embedding = nn.Embedding(type_vocab_size, d_model)
        self.position_embedding = nn.Embedding(self.seq_len, d_model)

        self.local_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=norm_first,
                )
                for _ in range(num_blocks)
            ]
        )

        self.global_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=norm_first,
                )
                for _ in range(num_blocks)
            ]
        )

        self.final_norm = nn.LayerNorm(d_model) if norm_first else None
        self.output_layer = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def _fold_for_local(
        self, pokemon_emb: torch.Tensor, format_emb: torch.Tensor
    ) -> torch.Tensor:
        """(batch, 6*attrs, d) + format -> (batch*6, 1+attrs, d)."""
        batch_size = pokemon_emb.size(0)
        pokemon_emb = pokemon_emb.view(
            batch_size, self.NUM_POKEMON, self.attrs_per_pokemon, self.d_model
        )
        format_expanded = format_emb.unsqueeze(1).expand(
            batch_size, self.NUM_POKEMON, 1, self.d_model
        )
        local_seq = torch.cat([format_expanded, pokemon_emb], dim=2)
        local_seq = local_seq.view(
            batch_size * self.NUM_POKEMON, 1 + self.attrs_per_pokemon, self.d_model
        )
        return local_seq

    def _unfold_from_local(
        self, local_out: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """(batch*6, 1+attrs, d) -> (batch, 6*attrs, d), dropping format token."""
        local_out = local_out.view(
            batch_size, self.NUM_POKEMON, 1 + self.attrs_per_pokemon, self.d_model
        )
        pokemon_emb = local_out[:, :, 1:, :]
        pokemon_emb = pokemon_emb.reshape(
            batch_size, self.NUM_POKEMON * self.attrs_per_pokemon, self.d_model
        )
        return pokemon_emb

    @torch.compile
    def forward(
        self,
        x_tokens: torch.LongTensor,
        type_ids: torch.LongTensor,
    ) -> torch.Tensor:
        batch_size, seq_len = x_tokens.size()
        if seq_len > self.seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds expected {self.seq_len}."
            )

        # standard embedding (tokens, position, types)
        position_ids = torch.arange(seq_len, device=x_tokens.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)
        token_emb = self.token_embedding(x_tokens)
        type_emb = self.type_embedding(type_ids)
        pos_emb = self.position_embedding(position_ids)
        x = token_emb + type_emb + pos_emb
        x = self.dropout(x)

        # split format token (constant) from pokemon tokens (updated)
        format_emb = x[:, 0:1, :]  # (batch, 1, d_model) - kept constant
        pokemon_emb = x[:, 1:, :]  # (batch, 6*attrs, d_model) - gets updated

        # alternating local and global attention
        for local_layer, global_layer in zip(self.local_layers, self.global_layers):
            # local attention: each pokemon sees format + its own tokens
            local_in = self._fold_for_local(pokemon_emb, format_emb)
            local_out = local_layer(local_in)
            pokemon_emb = self._unfold_from_local(local_out, batch_size)
            # global attention: full sequence
            global_in = torch.cat([format_emb, pokemon_emb], dim=1)
            global_out = global_layer(global_in)
            pokemon_emb = global_out[:, 1:, :]  # discard format output

        output = torch.cat([format_emb, pokemon_emb], dim=1)
        if self.final_norm is not None:
            output = self.final_norm(output)
        logits = self.output_layer(output)
        return logits


##############################
## High-Level Model Wrapper ##
##############################


class TeamPredictionModel:
    """
    High-level wrapper: TeamSet <-> tensors for training and inference.

    Decoder options go in iterative_decoder_kwargs / oneshot_decoder_kwargs.
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
        """Training forward: logits for loss computation."""
        return self._model(x_tokens, type_ids)

    def predict(
        self,
        team: TeamSet,
        return_stats: bool = False,
    ) -> Union[TeamSet, tuple[TeamSet, IterativeDecodingStats]]:
        """Fill $missing_*$ tokens for one team (user-facing inference)."""
        return self.predict_batch([team], return_stats=return_stats)[0]

    def predict_batch(
        self,
        teams: List[TeamSet],
        return_stats: bool = False,
    ) -> Union[List[TeamSet], List[tuple[TeamSet, IterativeDecodingStats]]]:
        """Fill $missing_*$ tokens for a batch of teams."""
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
        Tensor-level iterative decoding for eval (pre-encoded x/y pairs).

        Unlike predict(), does not round-trip through TeamSet.
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
        """One-shot decode for comparison with iterative decoding during eval."""
        return self.oneshot_decoder.decode(x_tokens, type_ids, pred_mask)

    def save_checkpoint(
        self,
        path: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save checkpoint (optional optimizer and extra_state)."""
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
        """Load checkpoint; returns saved extra_state (or {})."""
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
    """Factory for TeamPredictionModel with gin-friendly defaults."""
    model_classes = {
        "TeamTransformer": TeamTransformer,
        "LocalGlobalTeamTransformer": LocalGlobalTeamTransformer,
    }

    if model_type not in model_classes:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available: {list(model_classes.keys())}"
        )

    # Build model kwargs based on architecture
    if model_type == "LocalGlobalTeamTransformer":
        model_kwargs = {
            "d_model": d_model,
            "nhead": nhead,
            "num_blocks": num_layers,  # LocalGlobal uses num_blocks
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
        }
    else:
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
