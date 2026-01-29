from typing import Optional, Any, Type
import os
import warnings

import gin
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


from metamon.interface import (
    ObservationSpace,
    RewardFunction,
    ActionSpace,
    UniversalAction,
)
from metamon.il.model import (
    TransformerTurnEmbedding,
    PerceiverTurnEmbedding,
    TokenEmbedding,
    MultiModalEmbedding,
    LearnablePosEmb,
    PerceiverEncoder,
)
from metamon.tokenizer import PokemonTokenizer, UNKNOWN_TOKEN
from metamon.data import ParsedReplayDataset
from metamon.env import (
    TeamSet,
    PokeEnvWrapper,
    BattleAgainstBaseline,
    QueueOnLocalLadder,
    PokeAgentLadder,
)


try:
    import amago
except ImportError:
    raise ImportError(
        "Must install `amago` RL package. Visit: https://ut-austin-rpl.github.io/amago/ "
    )
else:
    assert (
        hasattr(amago, "__version__") and amago.__version__ >= "3.1.1"
    ), "Update to the latest AMAGO version!"
    from amago.envs import AMAGOEnv
    from amago.nets.utils import symlog, add_activation_log
    from amago.loading import RLData, RLDataset, Batch
    from amago.envs.amago_env import AMAGO_ENV_LOG_PREFIX
    from amago.nets.ff import Normalization


def _block_warnings():
    """Suppress common gymnasium warnings during environment creation."""
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=amago.utils.AmagoWarning)


def make_placeholder_env(
    observation_space: ObservationSpace, action_space: ActionSpace
) -> AMAGOEnv:
    """
    Create an environment that does nothing. Can be used to initialize a policy
    """
    _block_warnings()

    class _PlaceholderShowdown(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = observation_space.gym_space
            self.metamon_action_space = action_space
            self.action_space = action_space.gym_space
            self.observation_space["illegal_actions"] = gym.spaces.Box(
                low=0, high=1, shape=(self.action_space.n,), dtype=bool
            )
            self.metamon_battle_format = "PlaceholderShowdown"
            self.metamon_opponent_name = "PlaceholderOpponent"

        def reset(self, *args, **kwargs):
            obs = {
                key: np.zeros(value.shape, dtype=value.dtype)
                for key, value in self.observation_space.items()
            }
            return obs, {"legal_actions": []}

        def take_long_break(self):
            pass

        def resume_from_break(self):
            pass

    penv = _PlaceholderShowdown()
    return MetamonAMAGOWrapper(penv)


def make_local_ladder_env(*args, **kwargs):
    """
    Battle on the local Showdown ladder!
    """
    _block_warnings()
    menv = QueueOnLocalLadder(*args, **kwargs)
    print("Made Local Ladder Env")
    return PSLadderAMAGOWrapper(menv)


def make_pokeagent_ladder_env(*args, **kwargs):
    """
    Battle on the NeurIPS 2025 PokéAgent Challenge ladder!
    """
    _block_warnings()
    menv = PokeAgentLadder(*args, **kwargs)
    print("Made PokeAgent Ladder Env")
    return PSLadderAMAGOWrapper(menv)


def make_baseline_env(*args, **kwargs):
    """
    Battle against a built-in baseline opponent
    """
    _block_warnings()
    menv = BattleAgainstBaseline(*args, **kwargs)
    print("Made Baseline Env")
    return MetamonAMAGOWrapper(menv)


def make_placeholder_experiment(
    ckpt_base_dir: str,
    run_name: str,
    log: bool,
    observation_space: ObservationSpace,
    action_space: ActionSpace,
):
    """
    Initialize an AMAGO experiment that will be used to load a pretrained checkpoint
    and manage agent/env interaction.
    """
    # the environment is only used to initialize the network
    # before loading the correct checkpoint
    penv = make_placeholder_env(
        observation_space=observation_space,
        action_space=action_space,
    )
    dummy_dset = amago.loading.DoNothingDataset()
    dummy_env = lambda: penv
    experiment = MetamonAMAGOExperiment(
        # assumes that positional args
        # agent_type, tstep_encoder_type,
        # traj_encoder_type, and max_seq_len
        # are set in the gin file
        ckpt_base_dir=ckpt_base_dir,
        run_name=run_name,
        dataset=dummy_dset,
        make_train_env=dummy_env,
        make_val_env=dummy_env,
        env_mode="sync",
        async_env_mp_context="spawn",
        parallel_actors=1,
        exploration_wrapper_type=None,
        epochs=0,
        start_learning_at_epoch=float("inf"),
        start_collecting_at_epoch=float("inf"),
        train_timesteps_per_epoch=0,
        stagger_traj_file_lengths=False,
        train_batches_per_epoch=0,
        val_interval=None,
        val_timesteps_per_epoch=0,
        ckpt_interval=None,
        always_save_latest=False,
        always_load_latest=False,
        log_interval=1,
        batch_size=1,
        dloader_workers=0,
        log_to_wandb=log,
        wandb_project=os.environ.get("METAMON_WANDB_PROJECT"),
        wandb_entity=os.environ.get("METAMON_WANDB_ENTITY"),
        verbose=True,
    )
    return experiment


class MetamonAMAGOWrapper(amago.envs.AMAGOEnv):
    """AMAGOEnv wrapper for poke-env gymnasium environments.

    - Extends the observation space with an illegal action mask, which will
        be passed along to the actor network.
    - Adds success rate and valid action rate logging.
    """

    def __init__(self, metamon_env: PokeEnvWrapper):
        self.metamon_action_space = metamon_env.metamon_action_space
        super().__init__(
            env=metamon_env,
            env_name="metamon",
            batched_envs=1,
        )
        assert isinstance(self.action_space, gym.spaces.Discrete)
        self.observation_space["illegal_actions"] = gym.spaces.Box(
            low=0, high=1, shape=(self.action_space.n,), dtype=bool
        )

    def add_illegal_action_mask_to_obs(self, obs: dict, info: dict):
        # move legal action from info to obs
        legal_actions = info["legal_actions"]
        illegal_actions = np.ones((self.action_space.n,), dtype=bool)
        for agent_legal_action in legal_actions:
            illegal_actions[agent_legal_action] = False
        obs["illegal_actions"] = illegal_actions

    def inner_reset(self, *args, **kwargs):
        # move legal action from info to obs
        obs, info = self.env.reset(*args, **kwargs)
        self.add_illegal_action_mask_to_obs(obs, info)
        return obs, info

    def inner_step(self, action):
        # move legal action from info to obs
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.add_illegal_action_mask_to_obs(obs, info)
        return obs, reward, terminated, truncated, info

    def step(self, action):
        try:
            next_tstep, reward, terminated, truncated, info = super().step(action)
            # amago will average these stats over episodes, devices, and parallel actors.
            if "won" in info:
                info[f"{AMAGO_ENV_LOG_PREFIX} Win Rate"] = info["won"]
            if "valid_action_count" in info and "invalid_action_count" in info:
                info[f"{AMAGO_ENV_LOG_PREFIX} Valid Actions"] = info[
                    "valid_action_count"
                ] / (info["valid_action_count"] + info["invalid_action_count"])
            return next_tstep, reward, terminated, truncated, info
        except Exception as e:
            print(e)
            print("Force resetting due to long-tail error")
            self.reset()
            next_tstep, reward, terminated, truncated, info = self.step(action)
            reward *= 0.0
            terminated[:] = False
            truncated[:] = True  # force a proper reset asap
            return next_tstep, reward, terminated, truncated, info

    @property
    def env_name(self):
        return f"{self.env.metamon_battle_format}_vs_{self.env.metamon_opponent_name}"


@gin.configurable
class MetamonDiscrete(amago.nets.policy_dists.Discrete):
    """Discrete policy with temperature-based sampling.

    Extends AMAGO's Discrete PolicyOutput to add temperature scaling to the logits.
    High-temperature sampling is a better alternative to epsilon-greedy exploration
    for self-play in metamon due to illegal action masking.

    Args:
        d_action: Dimension of the action space.
        temperature: Temperature for scaling logits. Default is 1.0 (no scaling).
        clip_prob_low: Clips action probabilities to this value before
            renormalizing. Default is 0.001.
        clip_prob_high: Clips action probabilities to this value before
            renormalizing. Default is 0.99.
    """

    def __init__(
        self,
        d_action: int,
        clip_prob_low: float = 0.001,
        clip_prob_high: float = 0.99,
        temperature: float = 1.0,
    ):
        super().__init__(
            d_action=d_action,
            clip_prob_low=clip_prob_low,
            clip_prob_high=clip_prob_high,
        )
        self.temperature = temperature

    def forward(
        self, vec: torch.Tensor, log_dict: Optional[dict] = None
    ) -> amago.nets.policy_dists._Categorical:
        scaled_logits = vec / self.temperature

        dist = amago.nets.policy_dists._Categorical(logits=scaled_logits)
        probs = dist.probs
        clip_probs = probs.clamp(self.clip_prob_low, self.clip_prob_high)
        safe_probs = clip_probs / clip_probs.sum(-1, keepdims=True).detach()
        safe_dist = amago.nets.policy_dists._Categorical(probs=safe_probs)

        if log_dict is not None:
            from amago.nets.utils import add_activation_log

            add_activation_log("MetamonDiscrete-probs", probs, log_dict)
            add_activation_log(
                "MetamonDiscrete-temperature", torch.tensor(self.temperature), log_dict
            )

        return safe_dist


@gin.configurable
class MetamonMaskedActor(amago.nets.actor_critic.Actor):
    """
    Default AMAGO Actor with optional logit masking of illegal actions.

    Note that all the original models were trained with the equivalent of
    mask_illegal_actions=False... the dataset would not have illegal actions,
    and in self-play data an illegal action triggers a random one to be taken,
    so it's always a bad idea, and critic nets have no problem learning this.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        discrete: bool,
        gammas: torch.Tensor,
        n_layers: int = 2,
        d_hidden: int = 256,
        activation: str = "leaky_relu",
        dropout_p: float = 0.0,
        continuous_dist_type=None,
        mask_illegal_actions: bool = True,
    ):
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            discrete=discrete,
            gammas=gammas,
            n_layers=n_layers,
            d_hidden=d_hidden,
            activation=activation,
            dropout_p=dropout_p,
            continuous_dist_type=continuous_dist_type,
            discrete_dist_type=MetamonDiscrete,
        )
        self.mask_illegal_actions = mask_illegal_actions

    def actor_network_forward(
        self,
        state: torch.Tensor,
        log_dict: Optional[dict[str, Any]] = None,
        straight_from_obs: Optional[dict[str, torch.Tensor]] = None,
    ):
        dist_params = super().actor_network_forward(
            state, log_dict=log_dict, straight_from_obs=straight_from_obs
        )
        if self.mask_illegal_actions:
            Batch, Len, Gammas, N = dist_params.shape
            mask = straight_from_obs["illegal_actions"]
            no_options = mask.all(dim=-1, keepdim=True)
            # TODO: having no legal options should be considered a problem
            # with action masking / action space, but seems to happen
            # for two reasons: 1) battle is over and there's nothing left to do
            # (harmless) and 2) gen 9 revival blessing edge case (need to revisit).
            # prevent crash by letting agent pick its own action and dealing with
            # legality on the env side (probably falling back to a default choice).
            mask = torch.logical_and(mask, ~no_options)
            mask = einops.repeat(mask, f"b l n -> b l {Gammas} n")
            dist_params.masked_fill_(mask, -float("inf"))
        return dist_params


@gin.configurable
class MetamonMaskedResidualActor(amago.nets.actor_critic.ResidualActor):
    """ResidualActor with optional masking of illegal actions in logits.

    Mirrors `MetamonMaskedActor` but for AMAGO's ResidualActor head.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        discrete: bool,
        gammas: torch.Tensor,
        feature_dim: int = 256,
        residual_ff_dim: int = 512,
        residual_blocks: int = 2,
        activation: str = "leaky_relu",
        normalization: str = "layer",
        dropout_p: float = 0.0,
        continuous_dist_type=None,
        mask_illegal_actions: bool = True,
    ):
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            discrete=discrete,
            gammas=gammas,
            feature_dim=feature_dim,
            residual_ff_dim=residual_ff_dim,
            residual_blocks=residual_blocks,
            activation=activation,
            normalization=normalization,
            dropout_p=dropout_p,
            continuous_dist_type=continuous_dist_type,
            discrete_dist_type=MetamonDiscrete,
        )
        self.mask_illegal_actions = mask_illegal_actions

    def actor_network_forward(
        self,
        state: torch.Tensor,
        log_dict: Optional[dict[str, Any]] = None,
        straight_from_obs: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        dist_params = super().actor_network_forward(
            state, log_dict=log_dict, straight_from_obs=straight_from_obs
        )
        if self.mask_illegal_actions and straight_from_obs is not None:
            Batch, Len, Gammas, N = dist_params.shape
            mask = straight_from_obs["illegal_actions"]
            no_options = mask.all(dim=-1, keepdim=True)
            mask = torch.logical_and(mask, ~no_options)
            mask = einops.repeat(mask, f"b l n -> b l {Gammas} n")
            dist_params.masked_fill_(mask, -float("inf"))
        return dist_params


class PSLadderAMAGOWrapper(MetamonAMAGOWrapper):
    def __init__(self, env):
        assert isinstance(env, QueueOnLocalLadder)
        self.placeholder_obs = None
        self.battle_counter = 0
        super().__init__(env)

    def inner_reset(self, *args, **kwargs):
        if self.battle_counter >= self.env.num_battles:
            # quirk of amago's parallel actor auto-resets that matters
            # for online ladder.
            warnings.warn(
                "Blocking auto-reset to avoid creating a battle that will not be completed!"
            )
            return self.placeholder_obs, {}
        obs, info = self.env.reset(*args, **kwargs)
        self.battle_counter += 1
        if self.placeholder_obs is None:
            self.placeholder_obs = obs
        # move legal action from info to obs
        self.add_illegal_action_mask_to_obs(obs, info)
        return obs, info

    @property
    def env_name(self):
        return f"psladder_{self.env.env.username}"


def unknown_token_mask(tokens, skip_prob: float = 0.5, batch_max_prob: float = 0.2):
    """Randomly set entries in the text component of the observation space to UNKNOWN_TOKEN.

    Args:
        skip_prob: Probability of entirely skipping the mask for any given sequence
        batch_max_prob: For each sequence, randomly mask tokens with [0, batch_max_prob) prob
            (if not skipped).
    """
    B, L, tok = tokens.shape
    dev = tokens.device
    batch_mask = torch.rand(B) < (1.0 - skip_prob)  # mask tokens from this batch index
    batch_thresh = (
        torch.rand(B) * batch_max_prob
    )  # mask this % of tokens from the sequence
    thresh = (
        batch_mask * batch_thresh
    )  # 0 if batch index isn't masked, % to mask otherwise
    mask = torch.rand(tokens.shape) < thresh.view(-1, 1, 1)
    tokens[mask.to(dev)] = UNKNOWN_TOKEN
    return tokens.to(dev)


@gin.configurable
class MetamonTstepEncoder(amago.nets.tstep_encoders.TstepEncoder):
    """
    Token + numerical embedding for Metamon.

    Fuses multi-modal input with attention and summary tokens.
    Visualized on the README and in the paper architecture figure.
    """

    def __init__(
        self,
        obs_space,
        rl2_space,
        tokenizer: PokemonTokenizer,
        extra_emb_dim: int = 18,
        d_model: int = 100,
        n_layers: int = 3,
        n_heads: int = 5,
        scratch_tokens: int = 4,
        numerical_tokens: int = 6,
        token_mask_aug: bool = False,
        dropout: float = 0.05,
    ):
        super().__init__(obs_space=obs_space, rl2_space=rl2_space)
        self.token_mask_aug = token_mask_aug
        self.extra_emb = nn.Linear(rl2_space.shape[-1], extra_emb_dim)
        base_numerical_features = obs_space["numbers"].shape[0]
        base_text_features = obs_space["text_tokens"].shape[0]
        self.turn_embedding = TransformerTurnEmbedding(
            tokenizer=tokenizer,
            token_embedding_dim=d_model,
            text_features=base_text_features,
            numerical_features=base_numerical_features + extra_emb_dim,
            numerical_tokens=numerical_tokens,
            scratch_tokens=scratch_tokens,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )

    @property
    def emb_dim(self):
        return self.turn_embedding.output_dim

    @torch.compile
    def inner_forward(self, obs, rl2s, log_dict=None):
        if self.training and self.token_mask_aug:
            obs["text_tokens"] = unknown_token_mask(obs["text_tokens"])
        extras = F.leaky_relu(self.extra_emb(symlog(rl2s)))
        add_activation_log("MetamonTstepEncoder/extra_emb", extras, log_dict)
        numerical = torch.cat((obs["numbers"], extras), dim=-1)
        add_activation_log("MetamonTstepEncoder/numerical", numerical, log_dict)
        turn_emb = self.turn_embedding(
            token_inputs=obs["text_tokens"], numerical_inputs=numerical
        )
        add_activation_log("MetamonTstepEncoder/turn_emb", turn_emb, log_dict)
        return turn_emb


@gin.configurable
class MetamonPerceiverTstepEncoder(amago.nets.tstep_encoders.TstepEncoder):
    """
    Efficient attention scheme for processing turn token inputs.

    Uses latent cross-/self-attention with learnable positional embeddings.
    """

    def __init__(
        self,
        obs_space,
        rl2_space,
        tokenizer: PokemonTokenizer,
        extra_emb_dim: int = 18,
        d_model: int = 100,
        n_layers: int = 3,
        n_heads: int = 5,
        latent_tokens: int = 8,
        numerical_tokens: int = 6,
        token_mask_aug: bool = False,
        dropout: float = 0.05,
        max_tokens_per_turn: int = 128,
    ):
        super().__init__(obs_space=obs_space, rl2_space=rl2_space)
        self.token_mask_aug = token_mask_aug
        self.extra_emb = nn.Linear(rl2_space.shape[-1], extra_emb_dim)
        base_numerical_features = obs_space["numbers"].shape[0]
        base_text_features = obs_space["text_tokens"].shape[0]
        self.turn_embedding = PerceiverTurnEmbedding(
            tokenizer=tokenizer,
            token_embedding_dim=d_model,
            text_features=base_text_features,
            numerical_features=base_numerical_features + extra_emb_dim,
            numerical_tokens=numerical_tokens,
            latent_tokens=latent_tokens,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            max_tokens_per_turn=max_tokens_per_turn,
        )

    @property
    def emb_dim(self):
        return self.turn_embedding.output_dim

    @torch.compile
    def inner_forward(self, obs, rl2s, log_dict=None):
        if self.training and self.token_mask_aug:
            obs["text_tokens"] = unknown_token_mask(obs["text_tokens"])
        extras = F.leaky_relu(self.extra_emb(symlog(rl2s)))
        add_activation_log("MetamonPerceiverTstepEncoder/extra_emb", extras, log_dict)
        numerical = torch.cat((obs["numbers"], extras), dim=-1)
        add_activation_log(
            "MetamonPerceiverTstepEncoder/numerical", numerical, log_dict
        )
        turn_emb = self.turn_embedding(
            token_inputs=obs["text_tokens"], numerical_inputs=numerical
        )
        add_activation_log("MetamonPerceiverTstepEncoder/turn_emb", turn_emb, log_dict)
        return turn_emb


@gin.configurable
class MetamonGroupedTstepEncoder(amago.nets.tstep_encoders.TstepEncoder):
    """
    Timestep encoder for GroupedObservationSpace.

    Three-stage architecture:
    1. Pokemon encoder (shared): encodes each of 7 Pokemon independently
    2. Global encoder: encodes misc features (format, conditions, etc.) + rl2
    3. Fusion encoder: combines 8 entity embeddings into final turn representation
    """

    # Hardcoded to GroupedObservationSpace dimensions
    POKEMON_TEXT_LEN = 12
    POKEMON_NUM_LEN = 31
    MISC_TEXT_LEN = 20
    MISC_NUM_LEN = 4
    NUM_POKEMON = 7

    def __init__(
        self,
        obs_space,
        rl2_space,
        tokenizer: PokemonTokenizer,
        # Pokemon encoder
        d_pokemon: int = 64,
        n_heads_pokemon: int = 4,
        n_layers_pokemon: int = 2,
        latent_tokens_pokemon: int = 4,
        numerical_tokens_pokemon: int = 4,
        pokemon_out_norm: str = "layer",
        # Global encoder
        d_global: int = 64,
        n_heads_global: int = 4,
        n_layers_global: int = 2,
        latent_tokens_global: int = 4,
        numerical_tokens_global: int = 2,
        global_out_norm: str = "layer",
        # Fusion encoder
        d_fusion: int = 128,
        n_heads_fusion: int = 4,
        n_layers_fusion: int = 2,
        latent_tokens_fusion: int = 4,
        fusion_out_norm: str = "layer",
        # General
        extra_emb_dim: int = 16,
        dropout: float = 0.05,
    ):
        super().__init__(obs_space=obs_space, rl2_space=rl2_space)

        self.extra_emb = nn.Linear(rl2_space.shape[-1], extra_emb_dim)

        # pokemon encoder (shared for all 7)
        self.pokemon_token_emb = TokenEmbedding(tokenizer, d_pokemon)
        self.pokemon_fuse = MultiModalEmbedding(
            token_emb_dim=d_pokemon,
            numerical_d_inp=self.POKEMON_NUM_LEN,
            output_dim=d_pokemon,
            numerical_tokens=numerical_tokens_pokemon,
            dropout=dropout,
        )
        self.pokemon_pos = LearnablePosEmb(
            max_len=self.POKEMON_TEXT_LEN + numerical_tokens_pokemon,
            d_model=d_pokemon,
        )
        self.pokemon_perceiver = PerceiverEncoder(
            latent_tokens=latent_tokens_pokemon,
            d_model=d_pokemon,
            n_heads=n_heads_pokemon,
            n_layers=n_layers_pokemon,
            dropout=dropout,
        )
        pokemon_out_dim = latent_tokens_pokemon * d_pokemon
        self.pokemon_out_norm = Normalization(pokemon_out_norm, d_pokemon)
        self.pokemon_proj = nn.Linear(pokemon_out_dim, d_fusion)

        # global encoder
        self.global_token_emb = TokenEmbedding(tokenizer, d_global)
        self.global_fuse = MultiModalEmbedding(
            token_emb_dim=d_global,
            numerical_d_inp=self.MISC_NUM_LEN + extra_emb_dim,
            output_dim=d_global,
            numerical_tokens=numerical_tokens_global,
            dropout=dropout,
        )
        self.global_pos = LearnablePosEmb(
            max_len=self.MISC_TEXT_LEN + numerical_tokens_global,
            d_model=d_global,
        )
        self.global_perceiver = PerceiverEncoder(
            latent_tokens=latent_tokens_global,
            d_model=d_global,
            n_heads=n_heads_global,
            n_layers=n_layers_global,
            dropout=dropout,
        )
        global_out_dim = latent_tokens_global * d_global
        self.global_out_norm = Normalization(global_out_norm, d_global)
        self.global_proj = nn.Linear(global_out_dim, d_fusion)

        # fusion encoder
        self.entity_type_emb = nn.Embedding(self.NUM_POKEMON + 1, d_fusion)
        self.fusion = PerceiverEncoder(
            latent_tokens=latent_tokens_fusion,
            d_model=d_fusion,
            n_heads=n_heads_fusion,
            n_layers=n_layers_fusion,
            dropout=dropout,
        )
        self.fusion_out_norm = Normalization(fusion_out_norm, d_fusion)

        self._emb_dim = self.fusion.output_dim

    @property
    def emb_dim(self):
        return self._emb_dim

    def _encode_pokemon(
        self, text_tokens: torch.Tensor, numerical: torch.Tensor, log_dict=None
    ) -> torch.Tensor:
        B = text_tokens.size(0)

        # batch all Pokemon together: (B*7, 12), (B*7, 31)
        text_flat = einops.rearrange(text_tokens, "b n l -> (b n) l")
        nums_flat = einops.rearrange(numerical, "b n d -> (b n) d")

        # embed tokens: (B*7, 12, d_pokemon)
        tok_emb = self.pokemon_token_emb(text_flat)

        # multimodal fuse (needs dummy seq dim): (B*7, 1, L, d) → (B*7, L+num, d)
        tok_emb = tok_emb.unsqueeze(1)
        nums_flat = nums_flat.unsqueeze(1)
        seq = self.pokemon_fuse(tok_emb, nums_flat).squeeze(1)

        L = seq.size(1)
        pos_ids = (
            torch.arange(L, device=seq.device).unsqueeze(0).expand(seq.size(0), -1)
        )
        seq = seq + self.pokemon_pos(pos_ids)

        # Perceiver: (B*7, L, d) → (B*7, latent_tokens, d_pokemon)
        emb = self.pokemon_perceiver(seq, flatten=False)
        add_activation_log(
            "MetamonGroupedTstepEncoder/pokemon_perceiver", emb, log_dict
        )

        emb = self.pokemon_out_norm(emb)
        add_activation_log("MetamonGroupedTstepEncoder/pokemon_out_norm", emb, log_dict)

        emb = einops.rearrange(emb, "b t d -> b (t d)")
        emb = self.pokemon_proj(emb)
        add_activation_log("MetamonGroupedTstepEncoder/pokemon_proj", emb, log_dict)

        return einops.rearrange(emb, "(b n) d -> b n d", b=B, n=self.NUM_POKEMON)

    def _encode_global(
        self, text_tokens: torch.Tensor, numerical: torch.Tensor, log_dict=None
    ) -> torch.Tensor:
        tok_emb = self.global_token_emb(text_tokens)

        tok_emb = tok_emb.unsqueeze(1)
        numerical = numerical.unsqueeze(1)
        seq = self.global_fuse(tok_emb, numerical).squeeze(1)

        L = seq.size(1)
        pos_ids = (
            torch.arange(L, device=seq.device).unsqueeze(0).expand(seq.size(0), -1)
        )
        seq = seq + self.global_pos(pos_ids)

        # Perceiver: (B, L, d) → (B, latent_tokens, d_global)
        emb = self.global_perceiver(seq, flatten=False)
        add_activation_log("MetamonGroupedTstepEncoder/global_perceiver", emb, log_dict)

        emb = self.global_out_norm(emb)
        add_activation_log("MetamonGroupedTstepEncoder/global_out_norm", emb, log_dict)

        emb = einops.rearrange(emb, "b t d -> b (t d)")
        emb = self.global_proj(emb)
        add_activation_log("MetamonGroupedTstepEncoder/global_proj", emb, log_dict)

        return emb

    @torch.compile
    def inner_forward(self, obs, rl2s, log_dict=None):
        pokemon_text = torch.stack(
            [
                obs["text_active_pokemon_tokens"],
                obs["text_switch_0_tokens"],
                obs["text_switch_1_tokens"],
                obs["text_switch_2_tokens"],
                obs["text_switch_3_tokens"],
                obs["text_switch_4_tokens"],
                obs["text_opponent_active_pokemon_tokens"],
            ],
            dim=2,
        )

        pokemon_nums = torch.stack(
            [
                obs["numbers_active_pokemon"],
                obs["numbers_switch_0"],
                obs["numbers_switch_1"],
                obs["numbers_switch_2"],
                obs["numbers_switch_3"],
                obs["numbers_switch_4"],
                obs["numbers_opponent_active_pokemon"],
            ],
            dim=2,
        )

        B, L = pokemon_text.shape[:2]
        pokemon_text = einops.rearrange(pokemon_text, "b l n f -> (b l) n f")
        pokemon_nums = einops.rearrange(pokemon_nums, "b l n f -> (b l) n f")

        pokemon_embs = self._encode_pokemon(pokemon_text, pokemon_nums, log_dict)

        rl2s_flat = einops.rearrange(rl2s, "b l d -> (b l) d")
        extras = F.leaky_relu(self.extra_emb(symlog(rl2s_flat)))
        global_nums_flat = einops.rearrange(obs["numbers_misc"], "b l d -> (b l) d")
        global_nums = torch.cat([global_nums_flat, extras], dim=-1)
        global_text_flat = einops.rearrange(obs["text_misc_tokens"], "b l d -> (b l) d")
        global_emb = self._encode_global(global_text_flat, global_nums, log_dict)
        all_embs = torch.cat([pokemon_embs, global_emb.unsqueeze(1)], dim=1)

        type_ids = torch.arange(8, device=all_embs.device)
        all_embs = all_embs + self.entity_type_emb(type_ids)

        # Fusion perceiver: (B, 8, d_fusion) → (B, latent_tokens, d_fusion)
        emb = self.fusion(all_embs, flatten=False)
        add_activation_log("MetamonGroupedTstepEncoder/fusion", emb, log_dict)

        emb = self.fusion_out_norm(emb)
        add_activation_log("MetamonGroupedTstepEncoder/fusion_out_norm", emb, log_dict)

        emb = einops.rearrange(emb, "b t d -> b (t d)")
        emb = einops.rearrange(emb, "(b l) d -> b l d", b=B, l=L)

        return emb


class MetamonAMAGODataset(RLDataset):
    """A wrapper around the ParsedReplayDataset that converts to an AMAGO RLDataset.

    Args:
        parsed_replay_dset: The ParsedReplayDataset to wrap.
        dset_name: Give the dataset an arbitrary name for logging. Defaults to class name.
        refresh_files_every_epoch: Whether to find newly written replay files at the end of each epoch.
            This imitates the behavior of the main AMAGO disk replay buffer. Would be necessary for
            online RL. Defaults to False.
    """

    def __init__(
        self,
        parsed_replay_dset: ParsedReplayDataset,
        dset_name: Optional[str] = None,
        refresh_files_every_epoch: bool = False,
    ):
        super().__init__(dset_name=dset_name)
        self.parsed_replay_dset = parsed_replay_dset
        self.refresh_files_every_epoch = refresh_files_every_epoch

    @property
    def save_new_trajs_to(self):
        # disables AMAGO's trajetory saving; metamon
        # will handle this in its own replay format.
        return None

    def on_end_of_collection(self, experiment) -> dict[str, Any]:
        # TODO: implement FIFO replay buffer
        if self.refresh_files_every_epoch:
            self.parsed_replay_dset.refresh_files()
        return {"Num Replays": len(self.parsed_replay_dset)}

    def get_description(self) -> str:
        return f"Metamon Replay Dataset ({self.dset_name})"

    def sample_random_trajectory(self) -> RLData:
        data = self.parsed_replay_dset.random_sample()
        return self._process_data(data)

    def _process_data(self, data):
        obs, action_infos, rewards, dones = data
        # amago expects discrete actions to be one-hot encoded
        num_actions = self.parsed_replay_dset.action_space.gym_space.n
        actions_torch = F.one_hot(
            torch.tensor(action_infos["chosen"]).long().clamp(min=0),
            num_classes=num_actions,
        ).float()

        # set all illegal. needs to be one timestep longer than the actions to match the size of observations
        illegal_actions = torch.ones(
            (len(action_infos["chosen"]) + 1, num_actions)
        ).bool()
        for i, legal_actions in enumerate(action_infos["legal"]):
            for legal_action in legal_actions:
                legal_universal_action = UniversalAction(action_idx=legal_action)
                # discrete action spaces don't need a state input...
                legal_agent_action = (
                    self.parsed_replay_dset.action_space.action_to_agent_output(
                        state=None, action=legal_universal_action
                    )
                )
                # set the action legal
                illegal_actions[i, legal_agent_action] = False

        # a bit of a hack: put action info in the amago observation dict, let the network ignore it,
        # and make it accessible to mask the actor/critic loss later on.
        obs_torch = {k: torch.from_numpy(np.stack(v, axis=0)) for k, v in obs.items()}
        # add a final missing action to match the size of observations
        missing_acts = torch.tensor(action_infos["missing"] + [True]).unsqueeze(-1)
        obs_torch["missing_action_mask"] = missing_acts
        # the environment wrappers also add illegal_actions to the obs
        obs_torch["illegal_actions"] = illegal_actions
        rewards_torch = torch.from_numpy(rewards).unsqueeze(-1)
        dones_torch = torch.from_numpy(dones).unsqueeze(-1)
        time_idxs = torch.arange(len(action_infos["chosen"]) + 1).long().unsqueeze(-1)
        rl_data = RLData(
            obs=obs_torch,
            actions=actions_torch,
            rews=rewards_torch,
            dones=dones_torch,
            time_idxs=time_idxs,
        )
        return rl_data


@gin.configurable
class MetamonAMAGOExperiment(amago.Experiment):
    """
    Adds actions masking to the main AMAGO experiment, and leaves room for further tweaks.
    """

    def start(self):
        super().start()

    def init_envs(self):
        out = super().init_envs()
        amago.utils.call_async_env(self.val_envs, "take_long_break")
        return out

    def evaluate_val(self):
        amago.utils.call_async_env(self.val_envs, "resume_from_break")
        out = super().evaluate_val()
        amago.utils.call_async_env(self.val_envs, "take_long_break")
        return out

    def edit_actor_mask(
        self, batch: Batch, actor_loss: torch.FloatTensor, pad_mask: torch.BoolTensor
    ) -> torch.BoolTensor:
        B, L, G, _ = actor_loss.shape
        # missing_action_mask is one timestep too long to match the size of observations
        # True where the action is missing, False where it's provided.
        # pad_mask is True where the timestep should count towards loss, False where it shouldn't.
        missing_action_mask = einops.repeat(
            ~batch.obs["missing_action_mask"][:, :-1], "b l 1 -> b l g 1", g=G
        )
        return pad_mask & missing_action_mask

    def edit_critic_mask(
        self, batch: Batch, critic_loss: torch.FloatTensor, pad_mask: torch.BoolTensor
    ) -> torch.BoolTensor:
        B, L, C, G, _ = pad_mask.shape
        missing_action_mask = einops.repeat(
            ~batch.obs["missing_action_mask"][:, :-1], "b l 1 -> b l c g 1", g=G, c=C
        )
        return pad_mask & missing_action_mask
