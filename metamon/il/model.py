import math
from abc import ABC, abstractmethod
from typing import Type

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat
import gin

from metamon.tokenizer import PokemonTokenizer


class MetamonILModel(nn.Module, ABC):
    """
    If you want to build a custom model for replay data:
        1. Inherit from this class
        2. Define the forward pass with `inner_forward`
        3. You should be able to use it with our training pipeline and online baseline system w/o any changes.
    """

    def __init__(
        self,
        tokenizer: PokemonTokenizer,
        text_features: int,
        numerical_features: int,
        num_actions: int,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.text_features = text_features
        self.num_actions = num_actions
        self.numerical_features = numerical_features

    def forward(self, token_inputs, numerical_inputs, hidden_state=None):
        return self.inner_forward(
            token_inputs, numerical_inputs, hidden_state=hidden_state
        )

    @abstractmethod
    def inner_forward(sef, token_inputs, numerical_inputs, hidden_state=None):
        pass


class TokenEmbedding(nn.Module):
    """
    Map integer tokens to a sequence of vector representations.

    Just an `nn.Embedding` with a +1 shift to account for the "unknown"
    (-1) token. Separated out like this to make it easy to use
    this layer as an initialization for online RL policies.
    """

    def __init__(self, tokenizer: PokemonTokenizer, emb_dim: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer) + 1
        self.text_emb = nn.Embedding(self.vocab_size, embedding_dim=emb_dim)
        self.output_dim = emb_dim

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        shifted_tokens = (tokens + 1).clamp(0, self.vocab_size)
        text_emb = self.text_emb(shifted_tokens)
        return text_emb


class MultiModalEmbedding(nn.Module):
    """
    Take the text embedding and add on a representation of the numerical
    part of our observation, creating a multimodal sequence.

    We do not need embeddings to indicate which timestep of this new sequence
    is from text vs. numerical because they are padded to be a consistent length.
    """

    def __init__(
        self,
        token_emb_dim: int,
        numerical_d_inp: int,
        output_dim: int,
        numerical_tokens: int,
        dropout: float,
    ):
        super().__init__()
        self.text_emb = nn.Linear(token_emb_dim, output_dim)
        self.num_emb = nn.Linear(numerical_d_inp, numerical_tokens * output_dim)
        self.numerical_tokens = numerical_tokens
        self.dropout = nn.Dropout(dropout)
        self.output_dim = output_dim

    def forward(
        self, text_emb: torch.Tensor, numerical_features: torch.Tensor
    ) -> torch.Tensor:
        text_emb = F.leaky_relu(self.dropout(self.text_emb(text_emb)))
        num_emb = F.leaky_relu(self.dropout(self.num_emb(numerical_features)))
        num_emb = num_emb.unflatten(-1, (self.numerical_tokens, -1))
        seq = torch.cat((text_emb, num_emb), dim=-2)
        return seq


class FixedPosEmb(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, pos_idxs: torch.LongTensor):
        B, L = pos_idxs.shape
        emb = torch.zeros(
            (B, L, self.d_model), device=pos_idxs.device, dtype=torch.float32
        )
        coeff = torch.exp(
            (
                torch.arange(0, self.d_model, 2, device=emb.device, dtype=torch.float32)
                * -(math.log(10000.0) / self.d_model)
            )
        )
        emb[..., 0::2] = torch.sin(pos_idxs.float().unsqueeze(-1) * coeff)
        emb[..., 1::2] = torch.cos(pos_idxs.float().unsqueeze(-1) * coeff)
        return emb


class LearnablePosEmb(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.max_len = max_len

    def forward(self, pos_idxs: torch.LongTensor) -> torch.Tensor:
        return self.pos_emb(pos_idxs.clamp_min(0).clamp_max(self.max_len - 1))


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm_ff = nn.LayerNorm(d_model)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(kv)
        attn_out, _ = self.attn(
            query=q_norm, key=kv_norm, value=kv_norm, need_weights=False
        )
        q = q + attn_out
        q = q + self.ff(self.norm_ff(q))
        return q


class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm_attn(x)
        attn_out, _ = self.attn(
            query=x_norm, key=x_norm, value=x_norm, need_weights=False
        )
        x = x + attn_out
        x = x + self.ff(self.norm_ff(x))
        return x


class PerceiverEncoder(nn.Module):
    def __init__(
        self,
        latent_tokens: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(latent_tokens, d_model) * 0.02)
        self.cross_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(d_model=d_model, n_heads=n_heads, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.self_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(d_model=d_model, n_heads=n_heads, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.output_dim = latent_tokens * d_model

    def forward(self, x: torch.Tensor, flatten: bool = True) -> torch.Tensor:
        B, L, D = x.shape
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)
        for cross, self_attn in zip(self.cross_blocks, self.self_blocks):
            latents = cross(latents, x)
            latents = self_attn(latents)
        if flatten:
            return rearrange(latents, "b n d -> b 1 (n d)")
        return latents  # (B, latent_tokens, d_model)


class TimestepTransformer(nn.Module):
    """
    Take the multimodal sequence, add on a few blank ("scratch") tokens
    like it's a Vision Transformer, pass the new seq through a small Transformer,
    then treat the scratch tokens as the output.
    """

    def __init__(
        self,
        scratch_tokens: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.scratch_tokens = scratch_tokens
        self.d_model = d_model
        self.pos_emb = FixedPosEmb(d_model=d_model)
        emb_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            activation=F.leaky_relu,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.emb_transformer = nn.TransformerEncoder(
            emb_layer, num_layers=n_layers, norm=nn.LayerNorm(d_model)
        )
        self.output_dim = scratch_tokens * d_model

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        # create scratch token sequence
        B, Global_L, Local_L, D_emb = emb.shape
        assert self.d_model == D_emb
        device = emb.device
        scratch_tokens = torch.zeros(
            B, Global_L, self.scratch_tokens, D_emb, device=device
        )
        seq = torch.cat((emb, scratch_tokens), dim=-2)
        # this sequence dimension is really the length of each turn,
        # so you fold the length of the battle into the batch dimension
        seq = rearrange(seq, "b l l2 d -> (b l) l2 d")
        B, L, _ = seq.shape
        pos_idxs = torch.arange(0, L, device=seq.device).long().unsqueeze(0)
        # add positional embeddings manually
        pos_embs = self.pos_emb(pos_idxs)
        emb = self.emb_transformer(seq + pos_embs)
        # keep only scratch tokens, undoing the previous length fold
        emb = emb[:, -self.scratch_tokens :, :]
        emb = rearrange(
            emb, "(b l) l2 d -> b l (l2 d)", l=Global_L, l2=self.scratch_tokens
        )
        return emb


class TurnEmbedding(nn.Module, ABC):
    """
    Map the multimodal (text, numerical) features of each turn in a battle to a
    fixed-size representation.

    Separated into an abstract class to make it more organized to pull pretrained
    versions into online/offline RL policies.
    """

    def __init__(
        self,
        tokenizer: PokemonTokenizer,
        text_features: int,
        token_embedding_dim: int,
        numerical_features: int,
    ):
        super().__init__()
        self.token_embedding = TokenEmbedding(
            tokenizer=tokenizer, emb_dim=token_embedding_dim
        )
        self.text_features = text_features
        self.numerical_features = numerical_features
        self.tokenizer = tokenizer

    @property
    @abstractmethod
    def output_dim(self):
        pass

    @abstractmethod
    def forward(
        self, token_inputs: torch.Tensor, numerical_inputs: torch.Tensor
    ) -> torch.Tensor:
        pass


@gin.configurable
class TransformerTurnEmbedding(TurnEmbedding):
    def __init__(
        self,
        tokenizer: PokemonTokenizer,
        token_embedding_dim: int,
        text_features: int,
        numerical_features: int,
        numerical_tokens: int = gin.REQUIRED,
        scratch_tokens: int = gin.REQUIRED,
        d_model: int = gin.REQUIRED,
        n_heads: int = gin.REQUIRED,
        n_layers: int = gin.REQUIRED,
        dropout: float = gin.REQUIRED,
    ):
        super().__init__(
            tokenizer,
            token_embedding_dim=token_embedding_dim,
            text_features=text_features,
            numerical_features=numerical_features,
        )
        self.tformer_embedding = TimestepTransformer(
            scratch_tokens=scratch_tokens,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.multimodal_fuse = MultiModalEmbedding(
            token_emb_dim=self.token_embedding.output_dim,
            numerical_d_inp=self.numerical_features,
            output_dim=self.tformer_embedding.d_model,
            numerical_tokens=numerical_tokens,
            dropout=dropout,
        )

    @property
    def output_dim(self):
        return self.tformer_embedding.output_dim

    def forward(self, token_inputs, numerical_inputs):
        token_emb = self.token_embedding(token_inputs)
        tstep_seq = self.multimodal_fuse(token_emb, numerical_features=numerical_inputs)
        tstep_emb = self.tformer_embedding(tstep_seq)
        return tstep_emb


@gin.configurable
class FFTurnEmbedding(TurnEmbedding):
    def __init__(
        self,
        tokenizer: PokemonTokenizer,
        token_embedding_dim: int,
        text_features: int,
        numerical_features: int,
        d_hidden: int = gin.REQUIRED,
        d_emb: int = gin.REQUIRED,
        dropout: float = gin.REQUIRED,
    ):
        super().__init__(
            tokenizer,
            token_embedding_dim=token_embedding_dim,
            text_features=text_features,
            numerical_features=numerical_features,
        )
        self.d_emb = d_emb
        inp_dim = (
            self.token_embedding.output_dim * self.text_features
            + self.numerical_features
        )
        self.merge1 = nn.Linear(inp_dim, d_hidden)
        self.merge2 = nn.Linear(d_hidden, d_hidden)
        self.merge3 = nn.Linear(d_hidden, d_emb)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_emb)

    @property
    def output_dim(self):
        return self.d_emb

    def forward(self, token_inputs, numerical_inputs):
        token_emb = self.token_embedding(token_inputs)
        token_emb = rearrange(token_emb, "b l1 l2 d -> b l1 (l2 d)")
        inp = torch.cat((token_emb, numerical_inputs), dim=-1)
        emb = F.leaky_relu(self.dropout(self.merge1(inp)))
        emb = F.leaky_relu(self.dropout(self.merge2(emb)))
        emb = self.norm(self.merge3(emb))
        return emb


@gin.configurable
class PerceiverTurnEmbedding(TurnEmbedding):
    def __init__(
        self,
        tokenizer: PokemonTokenizer,
        token_embedding_dim: int,
        text_features: int,
        numerical_features: int,
        numerical_tokens: int = gin.REQUIRED,
        latent_tokens: int = gin.REQUIRED,
        d_model: int = gin.REQUIRED,
        n_heads: int = gin.REQUIRED,
        n_layers: int = gin.REQUIRED,
        dropout: float = gin.REQUIRED,
        max_tokens_per_turn: int = 128,
    ):
        super().__init__(
            tokenizer,
            token_embedding_dim=token_embedding_dim,
            text_features=text_features,
            numerical_features=numerical_features,
        )
        self.learned_pos = LearnablePosEmb(max_len=max_tokens_per_turn, d_model=d_model)
        self.perceiver = PerceiverEncoder(
            latent_tokens=latent_tokens,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.multimodal_fuse = MultiModalEmbedding(
            token_emb_dim=self.token_embedding.output_dim,
            numerical_d_inp=self.numerical_features,
            output_dim=d_model,
            numerical_tokens=numerical_tokens,
            dropout=dropout,
        )

    @property
    def output_dim(self):
        return self.perceiver.output_dim

    def forward(self, token_inputs, numerical_inputs):
        token_emb = self.token_embedding(token_inputs)
        seq = self.multimodal_fuse(token_emb, numerical_features=numerical_inputs)
        B, Global_L, Local_L, D = seq.shape
        seq = rearrange(seq, "b l1 l2 d -> (b l1) l2 d")
        L = seq.shape[1]
        pos = (
            torch.arange(0, L, device=seq.device)
            .long()
            .unsqueeze(0)
            .expand(seq.shape[0], -1)
        )
        seq = seq + self.learned_pos(pos)
        latents = self.perceiver(seq)
        latents = rearrange(latents, "(b l1) 1 d -> b l1 d", b=B, l1=Global_L)
        return latents


@gin.configurable
class GRUModel(MetamonILModel):
    """
    Use a basic GRU to process sequences of Turn representations to produce
    action logits.
    """

    def __init__(
        self,
        tokenizer: PokemonTokenizer,
        text_features: int,
        numerical_features: int,
        num_actions: int,
        token_embedding_dim: int = gin.REQUIRED,
        turn_embedding_Cls: Type[TurnEmbedding] = gin.REQUIRED,
        rnn_d_hidden: int = gin.REQUIRED,
        rnn_layers: int = gin.REQUIRED,
        rnn_dropout: float = gin.REQUIRED,
    ):
        super().__init__(
            tokenizer=tokenizer,
            text_features=text_features,
            numerical_features=numerical_features,
            num_actions=num_actions,
        )
        self.turn_embedding = turn_embedding_Cls(
            tokenizer=tokenizer,
            text_features=text_features,
            numerical_features=numerical_features,
            token_embedding_dim=token_embedding_dim,
        )
        self.rnn = nn.GRU(
            input_size=self.turn_embedding.output_dim,
            hidden_size=rnn_d_hidden,
            num_layers=rnn_layers,
            dropout=rnn_dropout,
            batch_first=True,
        )
        self.out = nn.Linear(rnn_d_hidden, num_actions)

    @torch.compile
    def inner_forward(self, token_inputs, numerical_inputs, hidden_state=None):
        turn_emb = self.turn_embedding(
            token_inputs=token_inputs, numerical_inputs=numerical_inputs
        )
        rnn_out, new_hidden_state = self.rnn(turn_emb, hidden_state)
        out = self.out(rnn_out)
        return out, new_hidden_state
