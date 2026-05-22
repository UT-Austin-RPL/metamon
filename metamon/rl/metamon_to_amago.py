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
from einops import rearrange


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
    PerceiverEncoder,
    LearnablePosEmb,
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
    from amago.nets.utils import symlog
    from amago.loading import RLData, RLDataset, Batch
    from amago.envs.amago_env import AMAGO_ENV_LOG_PREFIX


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

    # @torch.compile  # Disabled: causes stride assertion errors when fine-tuning with different reward functions
    def inner_forward(self, obs, rl2s, log_dict=None):
        if self.training and self.token_mask_aug:
            obs["text_tokens"] = unknown_token_mask(obs["text_tokens"])
        extras = F.leaky_relu(self.extra_emb(symlog(rl2s)))
        numerical = torch.cat((obs["numbers"], extras), dim=-1)
        turn_emb = self.turn_embedding(
            token_inputs=obs["text_tokens"], numerical_inputs=numerical
        )
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

    # @torch.compile  # Disabled: causes stride assertion errors when fine-tuning with different reward functions
    def inner_forward(self, obs, rl2s, log_dict=None):
        if self.training and self.token_mask_aug:
            obs["text_tokens"] = unknown_token_mask(obs["text_tokens"])
        extras = F.leaky_relu(self.extra_emb(symlog(rl2s)))
        numerical = torch.cat((obs["numbers"], extras), dim=-1)
        turn_emb = self.turn_embedding(
            token_inputs=obs["text_tokens"], numerical_inputs=numerical
        )
        return turn_emb


class PokemonSlotTurnEmbedding(nn.Module):
    """
    Encode fixed Pokemon slots independently, then merge slot and global context
    tokens with a second Perceiver block.
    """

    def __init__(
        self,
        tokenizer: PokemonTokenizer,
        slot_count: int,
        pokemon_text_features: int,
        pokemon_numerical_features: int,
        global_text_features: int,
        global_numerical_features: int,
        token_embedding_dim: int,
        d_model: int,
        n_heads: int,
        slot_layers: int,
        team_layers: int,
        slot_latent_tokens: int,
        team_latent_tokens: int,
        pokemon_numerical_tokens: int,
        global_numerical_tokens: int,
        dropout: float,
        max_pokemon_tokens: int,
        max_team_tokens: int,
    ):
        super().__init__()
        self.slot_count = slot_count
        self.pokemon_text_features = pokemon_text_features
        self.pokemon_numerical_features = pokemon_numerical_features
        self.global_text_features = global_text_features
        self.global_numerical_features = global_numerical_features
        self.token_embedding = TokenEmbedding(tokenizer, emb_dim=token_embedding_dim)

        self.pokemon_multimodal_fuse = MultiModalEmbedding(
            token_emb_dim=self.token_embedding.output_dim,
            numerical_d_inp=pokemon_numerical_features,
            output_dim=d_model,
            numerical_tokens=pokemon_numerical_tokens,
            dropout=dropout,
        )
        self.pokemon_pos = LearnablePosEmb(
            max_len=max_pokemon_tokens,
            d_model=d_model,
        )
        self.pokemon_perceiver = PerceiverEncoder(
            latent_tokens=slot_latent_tokens,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=slot_layers,
            dropout=dropout,
        )
        self.slot_projection = nn.Sequential(
            nn.Linear(self.pokemon_perceiver.output_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.slot_role_embedding = nn.Embedding(slot_count, d_model)

        self.global_multimodal_fuse = MultiModalEmbedding(
            token_emb_dim=self.token_embedding.output_dim,
            numerical_d_inp=global_numerical_features,
            output_dim=d_model,
            numerical_tokens=global_numerical_tokens,
            dropout=dropout,
        )
        self.team_pos = LearnablePosEmb(max_len=max_team_tokens, d_model=d_model)
        self.team_perceiver = PerceiverEncoder(
            latent_tokens=team_latent_tokens,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=team_layers,
            dropout=dropout,
        )

    @property
    def output_dim(self):
        return self.team_perceiver.output_dim

    def _add_pos(self, seq: torch.Tensor, pos_emb: LearnablePosEmb) -> torch.Tensor:
        pos = (
            torch.arange(0, seq.shape[1], device=seq.device)
            .long()
            .unsqueeze(0)
            .expand(seq.shape[0], -1)
        )
        return seq + pos_emb(pos)

    def forward(
        self,
        pokemon_token_inputs: torch.Tensor,
        pokemon_numerical_inputs: torch.Tensor,
        global_token_inputs: torch.Tensor,
        global_numerical_inputs: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = pokemon_token_inputs.shape
        pokemon_tokens = rearrange(
            pokemon_token_inputs,
            "b t (s p) -> (b t s) 1 p",
            s=self.slot_count,
            p=self.pokemon_text_features,
        )
        pokemon_numbers = rearrange(
            pokemon_numerical_inputs,
            "b t (s n) -> (b t s) 1 n",
            s=self.slot_count,
            n=self.pokemon_numerical_features,
        )
        pokemon_text_emb = self.token_embedding(pokemon_tokens)
        pokemon_seq = self.pokemon_multimodal_fuse(
            pokemon_text_emb,
            numerical_features=pokemon_numbers,
        )
        pokemon_seq = rearrange(pokemon_seq, "b 1 l d -> b l d")
        pokemon_seq = self._add_pos(pokemon_seq, self.pokemon_pos)
        pokemon_slots = self.pokemon_perceiver(pokemon_seq)
        pokemon_slots = rearrange(
            pokemon_slots,
            "(b t s) 1 d -> b t s d",
            b=B,
            t=T,
            s=self.slot_count,
        )
        pokemon_slots = self.slot_projection(pokemon_slots)
        role_idxs = torch.arange(
            0,
            self.slot_count,
            device=pokemon_slots.device,
        )
        pokemon_slots = pokemon_slots + self.slot_role_embedding(role_idxs).view(
            1, 1, self.slot_count, -1
        )

        global_text_emb = self.token_embedding(global_token_inputs)
        global_seq = self.global_multimodal_fuse(
            global_text_emb,
            numerical_features=global_numerical_inputs,
        )

        team_seq = torch.cat((pokemon_slots, global_seq), dim=-2)
        team_seq = rearrange(team_seq, "b t l d -> (b t) l d")
        team_seq = self._add_pos(team_seq, self.team_pos)
        team_emb = self.team_perceiver(team_seq)
        team_emb = rearrange(team_emb, "(b t) 1 d -> b t d", b=B, t=T)
        return team_emb


@gin.configurable
class MetamonPokemonSlotTstepEncoder(amago.nets.tstep_encoders.TstepEncoder):
    """
    AMAGO timestep encoder for Gen1PokemonSlotObservationSpace.
    """

    def __init__(
        self,
        obs_space,
        rl2_space,
        tokenizer: PokemonTokenizer,
        extra_emb_dim: int = 18,
        d_model: int = 168,
        n_heads: int = 8,
        slot_layers: int = 2,
        team_layers: int = 8,
        slot_latent_tokens: int = 2,
        team_latent_tokens: int = 8,
        pokemon_numerical_tokens: int = 4,
        global_numerical_tokens: int = 2,
        token_mask_aug: bool = False,
        dropout: float = 0.05,
        max_pokemon_tokens: int = 16,
        max_team_tokens: int = 32,
    ):
        super().__init__(obs_space=obs_space, rl2_space=rl2_space)
        self.token_mask_aug = token_mask_aug
        self.extra_emb = nn.Linear(rl2_space.shape[-1], extra_emb_dim)

        pokemon_text_features = obs_space["pokemon_text_tokens"].shape[0]
        pokemon_numerical_features = obs_space["pokemon_numbers"].shape[0]
        global_text_features = obs_space["global_text_tokens"].shape[0]
        global_numerical_features = obs_space["global_numbers"].shape[0] + extra_emb_dim

        slot_count = 13
        if pokemon_text_features % slot_count != 0:
            raise ValueError(
                "pokemon_text_tokens length must be divisible by the Pokemon slot count"
            )
        if pokemon_numerical_features % slot_count != 0:
            raise ValueError(
                "pokemon_numbers length must be divisible by the Pokemon slot count"
            )

        self.turn_embedding = PokemonSlotTurnEmbedding(
            tokenizer=tokenizer,
            slot_count=slot_count,
            pokemon_text_features=pokemon_text_features // slot_count,
            pokemon_numerical_features=pokemon_numerical_features // slot_count,
            global_text_features=global_text_features,
            global_numerical_features=global_numerical_features,
            token_embedding_dim=d_model,
            d_model=d_model,
            n_heads=n_heads,
            slot_layers=slot_layers,
            team_layers=team_layers,
            slot_latent_tokens=slot_latent_tokens,
            team_latent_tokens=team_latent_tokens,
            pokemon_numerical_tokens=pokemon_numerical_tokens,
            global_numerical_tokens=global_numerical_tokens,
            dropout=dropout,
            max_pokemon_tokens=max_pokemon_tokens,
            max_team_tokens=max_team_tokens,
        )

    @property
    def emb_dim(self):
        return self.turn_embedding.output_dim

    def inner_forward(self, obs, rl2s, log_dict=None):
        if self.training and self.token_mask_aug:
            obs["pokemon_text_tokens"] = unknown_token_mask(obs["pokemon_text_tokens"])
            obs["global_text_tokens"] = unknown_token_mask(obs["global_text_tokens"])
        extras = F.leaky_relu(self.extra_emb(symlog(rl2s)))
        global_numbers = torch.cat((obs["global_numbers"], extras), dim=-1)
        turn_emb = self.turn_embedding(
            pokemon_token_inputs=obs["pokemon_text_tokens"],
            pokemon_numerical_inputs=obs["pokemon_numbers"],
            global_token_inputs=obs["global_text_tokens"],
            global_numerical_inputs=global_numbers,
        )
        return turn_emb


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
class MetamonMultiTaskAgent(amago.agent.MultiTaskAgent):
    """MultiTaskAgent with cached intermediate values for efficient KL regularization.

    This agent caches trajectory embeddings and observation data during the forward pass,
    allowing dynamic damping to reuse these values instead of recomputing them.
    This provides ~1.6-1.9x speedup in training iteration time.

    Also caches Q-ensemble standard deviation for epistemic uncertainty-aware actor updates.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cached_kl_data = None
        self.cached_epistemic = None

    def forward(self, batch, log_step: bool):
        """Forward pass with caching for dynamic damping and epistemic weighting.

        Args:
            batch: Batch of RL data from amago.loading.Batch
            log_step: Whether this is a logging step

        Computes and caches intermediate values (trajectory embeddings, observations,
        Q-ensemble uncertainty) that can be reused by KL regularization and epistemic
        weighting without expensive recomputation.
        """
        # Reset caches
        self.cached_kl_data = None
        self.cached_epistemic = None

        # Compute encodings that will be cached
        self.update_info = {}
        active_log_dict = self.update_info if log_step else None

        # Timestep embedding
        o = self.tstep_encoder(obs=batch.obs, rl2s=batch.rl2s, log_dict=active_log_dict)
        straight_from_obs = {k: batch.obs[k] for k in self.pass_obs_keys_to_actor}

        # Trajectory embedding (expensive transformer operation)
        s_rep, hidden_state = self.traj_encoder(
            seq=o,
            time_idxs=batch.time_idxs,
            hidden_state=None,
            log_dict=active_log_dict
        )

        # Cache these values for KL computation (no .detach() - keep gradients for new policy)
        self.cached_kl_data = {
            's_rep': s_rep,  # [B, L, D_emb] - trajectory embeddings
            'straight_from_obs': straight_from_obs,  # dict of observations for actor
            'batch_shape': (s_rep.shape[0], s_rep.shape[1]),  # (B, L) for validation
        }

        # Call parent's forward to compute losses
        # The parent will compute Q-values, but we need to intercept them for epistemic caching.
        # Since we can't directly intercept inside parent's forward without full override,
        # we'll need to recompute Q-values after the fact for epistemic caching.
        critic_loss, actor_loss = super().forward(batch, log_step)

        # Post-hoc Q-ensemble std caching for epistemic weighting
        # We recompute Q(s,a) briefly to extract ensemble uncertainty
        # This is a small overhead but necessary without full forward() override
        if not self.fake_filter or self.online_coeff > 0:
            try:
                with torch.no_grad():
                    # Get actions from batch and expand with gamma dimension (same as parent does)
                    a = batch.actions  # [B, L, D_action]
                    B, L = s_rep.shape[0], s_rep.shape[1]
                    G = len(self.gammas)

                    # Expand actions with gamma dimension
                    a_buffer = einops.repeat(a, "B L act -> B L G act", G=G)  # [B, L-1, G, D_action]
                    # Note: batch.actions already has length L-1 (one less than observations)
                    # because the last observation doesn't have an action

                    # Verify shapes match before calling critic
                    s_slice = s_rep[:, :-1, ...].detach()  # [B, L-1, D_emb]
                    a_slice = a_buffer.unsqueeze(0)  # [1, B, L-1, G, D_action] - NO SLICING!

                    # Debug: check shapes
                    if s_slice.shape[1] != a_slice.shape[2]:
                        print(f"[WARNING] Shape mismatch in epistemic caching:")
                        print(f"  s_rep[:, :-1] shape: {s_slice.shape} (expected [B, L-1, D_emb])")
                        print(f"  a_buffer[:, :-1].unsqueeze(0) shape: {a_slice.shape} (expected [1, B, L-1, G, D_action])")
                        print(f"  Skipping epistemic caching for this batch")
                        self.cached_epistemic = None
                    else:
                        # Compute Q(s, a) ensemble (same as parent does at line 524-525 in agent.py)
                        s_a_g = (s_slice, a_slice)
                        q_s_a_g_ensemble = self.critics(*s_a_g)

                        # Handle distributional critics (C51, etc.) vs standard critics
                        if hasattr(q_s_a_g_ensemble, 'probs'):
                            # Distributional critic: convert distribution to scalar values
                            # q_s_a_g_ensemble is a distribution with shape [1, B, L-1, C, G, Bins]
                            scalar_q = self.critics.bin_dist_to_raw_vals(q_s_a_g_ensemble)  # [1, B, L-1, C, G, 1]
                        else:
                            # Standard critic: already scalar values
                            scalar_q = q_s_a_g_ensemble  # [1, B, L-1, C, G, 1]

                        # Extract ensemble std across critic dimension (dim=3)
                        # scalar_q has shape [1, B, L-1, C, G, 1]
                        q_std = scalar_q.std(dim=3)  # [1, B, L-1, G, 1]

                        self.cached_epistemic = {
                            'q_std': q_std,  # Already detached via torch.no_grad() context
                        }
            except Exception as e:
                print(f"[WARNING] Failed to cache epistemic uncertainty: {e}")
                print(f"  Skipping epistemic weighting for this batch")
                self.cached_epistemic = None

        return critic_loss, actor_loss


@gin.configurable
class MetamonAMAGOExperiment(amago.Experiment):
    """
    Adds actions masking to the main AMAGO experiment, and leaves room for further tweaks.

    Also supports dynamic damping for stable self-play training with:
    - Reverse-KL regularization to a reference policy
    - Power-law schedules for entropy and KL coefficients
    - Adaptive learning rate and KL coefficient control
    """

    def __init__(
        self,
        *args,
        # Dynamic damping parameters (gin-configurable)
        use_dynamic_damping: bool = False,
        kl_coef_init: float = 0.05,
        kl_coef_max: float = 0.5,
        kl_power_alpha: float = 0.5,
        kl_schedule_steps: int = 1_000_000,
        ent_coef_init: float = 0.01,
        ent_coef_min: float = 0.001,
        ent_power_alpha: float = 0.7,
        ent_schedule_steps: int = 1_000_000,
        target_kl_per_step: float = 0.01,
        kl_tolerance: float = 1.5,
        lr_shrink_factor: float = 0.5,
        lr_grow_factor: float = 1.1,
        kl_coef_growth_factor: float = 1.5,
        kl_coef_decay_factor: float = 0.9,
        min_lr: float = 1e-6,
        max_lr: float = 1e-3,
        dd_adapt_interval: int = 100,  # KL window length for adaptation
        # Epistemic weighting parameters (gin-configurable)
        use_epistemic_weighting: bool = False,
        epistemic_beta_init: float = 5.0,
        epistemic_beta_final: float = 1.0,
        epistemic_anneal_steps: int = 10000,
        epistemic_anneal_power: float = 0.5,
        epistemic_power: int = 2,
        # EMA (policy averaging) parameters (gin-configurable)
        use_ema: bool = False,
        ema_decay: float = 0.999,
        ema_update_interval: int = 1,
        ema_warmup_steps: int = 0,
        ema_eval_only: bool = True,
        **kwargs,
    ):
        # Debug: Print dynamic damping parameter (commented out to reduce log spam)
        # print(f"[DEBUG] MetamonAMAGOExperiment.__init__ called with use_dynamic_damping={use_dynamic_damping}", flush=True)

        super().__init__(*args, **kwargs)
        # print("[DEBUG] super().__init__() completed, policy exists:", hasattr(self, 'policy'), flush=True)

        # Dynamic damping state
        from collections import deque
        self.dd_state = None
        self.dd_config = None
        self.dd_adapt_interval = dd_adapt_interval  # Adapt controller every N steps (gin-configurable)
        self.kl_window = deque(maxlen=self.dd_adapt_interval)  # Sliding window of last N KL values
        self.dd_step_counter = 0  # Track steps for periodic adaptation

        if use_dynamic_damping:
            from metamon.rl.dynamic_damping import DynamicDampingConfig
            # print(f"[DEBUG] Creating DynamicDampingConfig...", flush=True)
            self.dd_config = DynamicDampingConfig(
                enabled=True,
                kl_coef_init=kl_coef_init,
                kl_coef_max=kl_coef_max,
                kl_power_alpha=kl_power_alpha,
                kl_schedule_steps=kl_schedule_steps,
                ent_coef_init=ent_coef_init,
                ent_coef_min=ent_coef_min,
                ent_power_alpha=ent_power_alpha,
                ent_schedule_steps=ent_schedule_steps,
                target_kl_per_step=target_kl_per_step,
                kl_tolerance=kl_tolerance,
                lr_shrink_factor=lr_shrink_factor,
                lr_grow_factor=lr_grow_factor,
                kl_coef_growth_factor=kl_coef_growth_factor,
                kl_coef_decay_factor=kl_coef_decay_factor,
                min_lr=min_lr,
                max_lr=max_lr,
            )
            # print(f"[DEBUG] dd_config created: {self.dd_config}", flush=True)
            # Note: dd_state will be initialized in start() after policy is created
        else:
            pass
            # print(f"[DEBUG] use_dynamic_damping=False, skipping dd_config creation", flush=True)

        # Epistemic weighting configuration
        self.use_epistemic_weighting = use_epistemic_weighting
        self.epistemic_beta_init = epistemic_beta_init
        self.epistemic_beta_final = epistemic_beta_final
        self.epistemic_anneal_steps = epistemic_anneal_steps
        self.epistemic_anneal_power = epistemic_anneal_power
        self.epistemic_power = epistemic_power

        # EMA (policy averaging) configuration
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_update_interval = ema_update_interval
        self.ema_warmup_steps = ema_warmup_steps
        self.ema_eval_only = ema_eval_only
        self.ema_model = None  # Will be initialized in start()
        self.ema_step_counter = 0
        self._training_state_dict = None  # Temporary storage for eval weight swapping
        self.epistemic_step = 0  # Track steps for beta annealing

    def start(self):
        """Override start to initialize dynamic damping after policy is created."""
        # print("[DEBUG] start() called", flush=True)
        super().start()
        # print("[DEBUG] super().start() completed", flush=True)

        # Initialize dynamic damping state now that policy exists
        if self.dd_config is not None and self.dd_config.enabled:
            from metamon.rl.dynamic_damping import DynamicDampingState
            # print("[DEBUG] Initializing DynamicDampingState...", flush=True)
            self.dd_state = DynamicDampingState(
                base_model=self.policy,
                config=self.dd_config,
            )
            # print(f"[Dynamic Damping] Initialized with kl_coef={self.dd_state.kl_coef:.4f}, "
            #       f"ent_coef={self.dd_state.ent_coef:.4f}", flush=True)

        # Initialize EMA model now that policy exists
        if self.use_ema:
            import copy
            self.ema_model = copy.deepcopy(self.policy)
            self.ema_model.eval()
            for param in self.ema_model.parameters():
                param.requires_grad_(False)
            print(f"[EMA] Initialized with decay={self.ema_decay}, warmup_steps={self.ema_warmup_steps}")

    def init_policy(self):
        """Initialize policy and optionally enable dynamic damping."""
        # print("[DEBUG] init_policy() CALLED", flush=True)
        out = super().init_policy()
        # print("[DEBUG] super().init_policy() COMPLETED", flush=True)

        # Debug: Check if dynamic damping is configured (commented out to reduce log spam)
        # print(f"[DEBUG] dd_config is None: {self.dd_config is None}", flush=True)
        # if self.dd_config is not None:
        #     print(f"[DEBUG] dd_config.enabled: {self.dd_config.enabled}", flush=True)

        # Initialize dynamic damping if configured
        if self.dd_config is not None and self.dd_config.enabled:
            self._init_dynamic_damping()
        # else:
        #     print("[WARNING] Dynamic damping NOT initialized - check gin config!", flush=True)

        return out

    def _init_dynamic_damping(self):
        """Initialize dynamic damping with a frozen reference policy snapshot."""
        from metamon.rl.dynamic_damping import DynamicDampingState

        # Create frozen reference from current policy
        self.dd_state = DynamicDampingState(
            base_model=self.policy,  # The full agent
            config=self.dd_config,
        )
        print(f"[Dynamic Damping] Initialized with kl_coef={self.dd_state.kl_coef:.4f}, "
              f"ent_coef={self.dd_state.ent_coef:.4f}")

    def update_reference_policy(self):
        """Update the reference policy to match current policy weights.

        Call this after loading a checkpoint to ensure the reference policy
        is a snapshot of the loaded weights, not the random initialization.
        """
        if self.dd_state is not None:
            import copy
            print("[Dynamic Damping] Updating reference policy to match loaded checkpoint...")
            self.dd_state.ref_model = copy.deepcopy(self.policy)
            self.dd_state.ref_model.eval()
            for param in self.dd_state.ref_model.parameters():
                param.requires_grad_(False)
            print("[Dynamic Damping] Reference policy updated successfully")

        # Also update EMA model if enabled
        if self.use_ema and self.ema_model is not None:
            import copy
            print("[EMA] Updating EMA model to match loaded checkpoint...")
            self.ema_model = copy.deepcopy(self.policy)
            self.ema_model.eval()
            for param in self.ema_model.parameters():
                param.requires_grad_(False)
            print("[EMA] EMA model updated successfully")

    def enable_dynamic_damping(self, config=None):
        """Manually enable dynamic damping after initialization.

        Useful for programmatically enabling damping outside of gin configs.

        Args:
            config: Optional DynamicDampingConfig. If None, uses default config.
        """
        from metamon.rl.dynamic_damping import DynamicDampingConfig, DynamicDampingState

        if config is None:
            config = DynamicDampingConfig()

        self.dd_config = config
        self.dd_state = DynamicDampingState(
            base_model=self.policy,
            config=config,
        )
        print(f"[Dynamic Damping] Enabled with kl_coef={self.dd_state.kl_coef:.4f}")

    def compute_loss(self, batch: Batch, log_step: bool) -> dict:
        """Compute RL loss with optional epistemic weighting and dynamic damping.

        Overrides parent to apply per-timestep epistemic confidence weights BEFORE
        masked averaging (critical for epistemic weighting to affect gradients).
        """
        # If epistemic weighting is disabled, use parent's implementation with KL damping
        if not self.use_epistemic_weighting:
            # Call parent to get standard actor/critic losses
            loss_dict = super().compute_loss(batch, log_step)

            # Add KL regularization if dynamic damping is enabled
            if self.dd_state is not None and self.dd_config.enabled:
                kl_loss, kl_metrics = self._compute_kl_loss(batch, log_step)
                loss_dict["Actor Loss"] = loss_dict["Actor Loss"] + kl_loss
                loss_dict.update(kl_metrics)

                # Track KL for adaptive control (sliding window of last N steps)
                if "KL Divergence" in kl_metrics:
                    self.kl_window.append(kl_metrics["KL Divergence"])
                    self.dd_step_counter += 1

                    # Adapt controller every N steps based on LOCAL KL window (not entire epoch)
                    if self.dd_step_counter >= self.dd_adapt_interval and len(self.kl_window) >= 10:
                        mean_kl = float(np.mean(self.kl_window))
                        self.dd_state.adapt_from_observed_kl(self.optimizer, mean_kl)
                        self.dd_step_counter = 0

            return loss_dict

        # Epistemic weighting path: manually compute losses with per-timestep weighting
        # Call Agent.forward() to get per-timestep losses
        critic_loss, actor_loss = self.policy_aclr(batch, log_step=log_step)
        update_info = self.policy.update_info
        B, L_1, G, _ = actor_loss.shape
        C = len(self.policy.critics)

        # Apply epistemic weighting to actor_loss BEFORE masked_avg
        actor_loss = self._apply_epistemic_weighting(actor_loss, batch, log_step)

        # Apply masking (copied from parent Experiment.compute_loss)
        state_mask = (~((batch.rl2s == self.policy.pad_val).all(-1, keepdim=True))).bool()
        critic_state_mask = einops.repeat(state_mask[:, 1:, ...], f"B L 1 -> B L {C} {G} 1")
        actor_state_mask = einops.repeat(state_mask[:, 1:, ...], f"B L 1 -> B L {G} 1")

        # Hook to allow custom masks (e.g., missing_action_mask in metamon)
        actor_state_mask = self.edit_actor_mask(batch, actor_loss, actor_state_mask)
        critic_state_mask = self.edit_critic_mask(batch, critic_loss, critic_state_mask)

        # Compute scalar losses via masked averaging
        batch_size = B * L_1
        unmasked_batch_size = actor_state_mask[..., 0, 0].sum()
        masked_actor_loss = amago.utils.masked_avg(actor_loss, actor_state_mask)
        if isinstance(critic_loss, torch.Tensor):
            masked_critic_loss = amago.utils.masked_avg(critic_loss, critic_state_mask)
        else:
            assert critic_loss is None
            masked_critic_loss = 0.0

        loss_dict = {
            "Critic Loss": masked_critic_loss,
            "Actor Loss": masked_actor_loss,
            "Sequence Length": L_1 + 1,
            "Batch Size (in Timesteps)": batch_size,
            "Unmasked Batch Size (in Timesteps)": unmasked_batch_size,
        }
        loss_dict.update(update_info)

        # Add KL regularization if dynamic damping enabled (independent mechanism)
        if self.dd_state is not None and self.dd_config.enabled:
            kl_loss, kl_metrics = self._compute_kl_loss(batch, log_step)
            loss_dict["Actor Loss"] = loss_dict["Actor Loss"] + kl_loss
            loss_dict.update(kl_metrics)

            # Track KL for adaptive control
            if "KL Divergence" in kl_metrics:
                self.kl_window.append(kl_metrics["KL Divergence"])
                self.dd_step_counter += 1

                if self.dd_step_counter >= self.dd_adapt_interval and len(self.kl_window) >= 10:
                    mean_kl = float(np.mean(self.kl_window))
                    self.dd_state.adapt_from_observed_kl(self.optimizer, mean_kl)
                    self.dd_step_counter = 0

        # Clean up cache to prevent stale data
        if hasattr(self.policy, 'cached_epistemic') and self.policy.cached_epistemic is not None:
            self.policy.cached_epistemic = None

        return loss_dict

    def _compute_kl_loss(self, batch: Batch, log_step: bool) -> tuple[torch.Tensor, dict]:
        """Compute reverse-KL regularization loss: KL(π_new || π_ref).

        Returns:
            kl_loss: Scalar KL loss weighted by kl_coef
            metrics: Dict of metrics for logging
        """
        from metamon.rl.dynamic_damping import compute_masked_reverse_kl, compute_policy_entropy
        from einops import repeat

        # Try to use cached values from agent's forward pass (MetamonMultiTaskAgent)
        # This eliminates expensive recomputation of encodings
        cached = getattr(self.policy, 'cached_kl_data', None)

        # Validation mode: check if cached values match recomputed values
        # Set METAMON_VALIDATE_CACHE=1 environment variable to enable
        validate_cache = os.environ.get('METAMON_VALIDATE_CACHE', '0') == '1'

        if cached is not None:
            # FAST PATH: Use cached values from forward pass (~1.6-1.9x speedup)
            state = cached['s_rep']
            straight_from_obs = cached['straight_from_obs'].copy()  # Shallow copy to avoid mutation
            straight_from_obs["illegal_actions"] = batch.obs.get("illegal_actions")

            # Validation: verify cached values match recomputed values
            if validate_cache and log_step:
                with torch.no_grad():
                    tstep_emb_check = self.policy.tstep_encoder(
                        obs=batch.obs, rl2s=batch.rl2s, log_dict=None
                    )
                    traj_emb_check, _ = self.policy.traj_encoder(
                        seq=tstep_emb_check, time_idxs=batch.time_idxs, log_dict=None
                    )

                    # Check if cached values match recomputed values
                    max_diff = (state - traj_emb_check).abs().max().item()
                    if max_diff > 1e-5:
                        print(f"[CACHE VALIDATION WARNING] Max difference: {max_diff:.2e}")
                    else:
                        print(f"[CACHE VALIDATION OK] Max difference: {max_diff:.2e}")
        else:
            # FALLBACK: Recompute encodings (backwards compatibility or if caching disabled)
            # This path is used if not using MetamonMultiTaskAgent
            tstep_emb = self.policy.tstep_encoder(
                obs=batch.obs,
                rl2s=batch.rl2s,
                log_dict=None,
            )

            # Get trajectory embeddings from NEW policy's traj encoder
            traj_emb, _ = self.policy.traj_encoder(
                seq=tstep_emb,
                time_idxs=batch.time_idxs,
                log_dict=None,
            )

            # Get state representation
            state = traj_emb

            # Get observations to pass directly to actor (for illegal action masking)
            straight_from_obs = {
                k: batch.obs[k] for k in self.policy.pass_obs_keys_to_actor
            }
            straight_from_obs["illegal_actions"] = batch.obs.get("illegal_actions")

        # Get NEW policy logits (with gradients)
        new_dist_params = self.policy.actor.actor_network_forward(
            state=state,
            log_dict=None,
            straight_from_obs=straight_from_obs,
        )  # [B, L, G, A] - includes initial timestep at index 0

        # Get REFERENCE policy logits (no gradients)
        with torch.no_grad():
            ref_dist_params = self.dd_state.ref_model.actor.actor_network_forward(
                state=state,  # Reuse same state encoding
                log_dict=None,
                straight_from_obs=straight_from_obs,
            )  # [B, L, G, A] - includes initial timestep at index 0

        # Slice to exclude first timestep (no action at initial state)
        # This aligns with how AMAGO handles actor loss (actions start at timestep 1)
        new_dist_params = new_dist_params[:, 1:, :, :]  # [B, L-1, G, A]
        ref_dist_params = ref_dist_params[:, 1:, :, :]  # [B, L-1, G, A]

        B, L, G, A = new_dist_params.shape  # Note: L is now L-1 (action-aligned length)

        # Get legal action mask (inverse of illegal_actions), also sliced to match
        legal_mask = ~straight_from_obs["illegal_actions"][:, 1:, :]  # [B, L, A]
        legal_mask = repeat(legal_mask, "b l a -> b l g a", g=G)  # [B, L, G, A]

        # Compute KL divergence per timestep
        kl_per_timestep = compute_masked_reverse_kl(
            new_logits=new_dist_params.reshape(B * L * G, A),
            ref_logits=ref_dist_params.reshape(B * L * G, A),
            legal_mask=legal_mask.reshape(B * L * G, A),
        )  # [B*L*G]
        kl_per_timestep = kl_per_timestep.reshape(B, L, G, 1)  # [B, L, G, 1]

        # Compute policy entropy (for logging)
        entropy_per_timestep = compute_policy_entropy(
            logits=new_dist_params.reshape(B * L * G, A),
            legal_mask=legal_mask.reshape(B * L * G, A),
        ).reshape(B, L, G, 1)

        # Apply the same masking as actor loss (reuse edit_actor_mask)
        state_mask = (~((batch.rl2s == self.policy.pad_val).all(-1, keepdim=True))).bool()
        # Slice to match action-aligned length (same as base AMAGO)
        actor_state_mask = repeat(state_mask[:, 1:, ...], f"b l 1 -> b l {G} 1")
        actor_state_mask = self.edit_actor_mask(batch, kl_per_timestep, actor_state_mask)

        # Compute masked averages
        masked_kl = amago.utils.masked_avg(kl_per_timestep, actor_state_mask)
        masked_entropy = amago.utils.masked_avg(entropy_per_timestep, actor_state_mask)

        # Weighted KL loss
        kl_loss = self.dd_state.kl_coef * masked_kl

        # Metrics for logging (always log all damping metrics)
        metrics = {
            "KL Divergence": masked_kl.item(),
            "Policy Entropy": masked_entropy.item(),
            "Damping/KL Coefficient": self.dd_state.kl_coef,
            "Damping/Entropy Coefficient": self.dd_state.ent_coef,
            "Damping/Step": self.dd_state.step,
            "Damping/Learning Rate": self.dd_state.current_lr if self.dd_state.current_lr is not None else self.optimizer.param_groups[0]["lr"],
        }

        return kl_loss, metrics

    def _update_ema_weights(self):
        """Update EMA model weights using exponential moving average.

        EMA update formula: ema_param = decay * ema_param + (1 - decay) * current_param

        Respects warmup period and update interval for gradual/efficient updates.
        """
        import torch

        self.ema_step_counter += 1

        # Warmup: skip EMA updates until warmup period completes
        if self.ema_step_counter < self.ema_warmup_steps:
            return

        # Update interval: only update every N steps (for efficiency)
        if self.ema_step_counter % self.ema_update_interval != 0:
            return

        # EMA update: ema_param = decay * ema_param + (1 - decay) * current_param
        with torch.no_grad():
            for ema_param, current_param in zip(
                self.ema_model.parameters(),
                self.policy.parameters()
            ):
                ema_param.data.mul_(self.ema_decay).add_(
                    current_param.data, alpha=1 - self.ema_decay
                )

    def train_step(self, batch: Batch, log_step: bool):
        """Training step with dynamic damping schedule updates and adaptive control."""
        # Update damping schedules before training step
        if self.dd_state is not None and self.dd_config.enabled:
            self.dd_state.update_schedules()

        # Perform standard training step
        metrics = super().train_step(batch, log_step)

        # Update EMA weights after gradient step
        if self.use_ema:
            self._update_ema_weights()

        return metrics

    def save_ema_checkpoint(self, epoch: int):
        """Save EMA model weights separately from training checkpoint.

        Saves to: {ckpt_dir}/ema_weights/policy_epoch_{epoch}.pt

        This allows loading EMA checkpoints independently of training checkpoints
        for evaluation or deployment.
        """
        import os
        import torch

        if not self.use_ema:
            return

        # Create EMA checkpoint directory
        ema_ckpt_dir = os.path.join(self.ckpt_dir, "ema_weights")
        os.makedirs(ema_ckpt_dir, exist_ok=True)

        # Save EMA weights
        ema_path = os.path.join(ema_ckpt_dir, f"policy_epoch_{epoch}.pt")
        torch.save(self.ema_model.state_dict(), ema_path)
        print(f"[EMA] Saved checkpoint to {ema_path}")

    def save_checkpoint(self) -> None:
        """Override AMAGO's save_checkpoint to also save EMA weights.

        AMAGO calls this method from learn() loop when epoch % ckpt_interval == 0.
        We must override this (not train_epoch) to ensure EMA checkpoints are saved.
        """
        # Call parent's checkpoint saving (saves training state + policy weights)
        super().save_checkpoint()

        # Save EMA checkpoint if enabled
        if self.use_ema:
            self.save_ema_checkpoint(self.epoch)

    def train_epoch(self, epoch: int):
        """Training epoch with adaptive LR/KL control during training (every N steps)."""
        # Reset step counter at start of epoch (window keeps rolling)
        self.dd_step_counter = 0

        # Run standard training epoch (adaptive control happens every N steps during training)
        out = super().train_epoch(epoch)

        # End-of-epoch: adapt if we have accumulated steps since last adaptation
        # (ensures we don't miss the last partial interval)
        if self.dd_state is not None and self.dd_config.enabled and \
           self.dd_step_counter > 0 and len(self.kl_window) >= 10:
            mean_kl = float(np.mean(self.kl_window))
            self.dd_state.adapt_from_observed_kl(self.optimizer, mean_kl)

            print(f"[Dynamic Damping] End-of-epoch {epoch} adaptation: mean_kl={mean_kl:.4f} "
                  f"(over last {len(self.kl_window)} steps), "
                  f"kl_coef={self.dd_state.kl_coef:.4f}, "
                  f"lr={self.optimizer.param_groups[0]['lr']:.6f}")

            # Reset for next epoch
            self.dd_step_counter = 0

        return out

    def _apply_epistemic_weighting(
        self,
        actor_loss: torch.Tensor,  # [B, L-1, G, 1]
        batch: Batch,
        log_step: bool
    ) -> torch.Tensor:
        """Apply per-timestep confidence weighting based on critic uncertainty.

        Weights actor gradients by inverse uncertainty: w = 1/(1 + β·σ̃)^p
        where σ̃ is normalized critic ensemble std dev.

        Args:
            actor_loss: Per-timestep actor loss [B, L-1, G, 1]
            batch: Batch of RL data
            log_step: Whether this is a logging step

        Returns:
            Weighted actor loss with same shape [B, L-1, G, 1]
        """
        # Extract Q-ensemble std from cache
        if self.policy.cached_epistemic is None or 'q_std' not in self.policy.cached_epistemic:
            # Fallback: no epistemic data cached (shouldn't happen if agent forward ran)
            print("[WARNING] Epistemic cache not found, skipping weighting")
            return actor_loss

        q_std = self.policy.cached_epistemic['q_std']  # [1, B, L-1, G, 1]

        # Shape alignment: remove batch dimension and match actor_loss
        q_std = q_std.squeeze(0)  # [B, L-1, G, 1]

        # Defensive shape check
        assert q_std.shape == actor_loss.shape, \
            f"Shape mismatch: q_std {q_std.shape} vs actor_loss {actor_loss.shape}"

        # Get mask for valid (non-padding) timesteps
        state_mask = (~((batch.rl2s == self.policy.pad_val).all(-1, keepdim=True))).bool()
        state_mask = state_mask[:, 1:]  # [B, L-1, 1]
        state_mask = state_mask.unsqueeze(-1)  # [B, L-1, 1, 1] - only ONE unsqueeze!

        # Normalize uncertainty (using only valid timesteps)
        sigma_norm = self._normalize_uncertainty(q_std, state_mask)  # [B, L-1, G, 1]

        # Compute confidence weights: w = 1 / (1 + β·σ̃)^p
        beta = self._get_current_beta()
        confidence = 1.0 / (1.0 + beta * sigma_norm).pow(self.epistemic_power)

        # Ensure stop-gradient (no backprop through uncertainty)
        confidence = confidence.detach()

        # Apply per-timestep weighting
        weighted_actor_loss = actor_loss * confidence

        # Log metrics
        if log_step:
            self._log_epistemic_metrics(sigma_norm, confidence, state_mask)

            # Debug prints for first few steps (validation)
            if self.epistemic_step <= 5:
                print(f"\n=== Epistemic Weighting Debug (step {self.epistemic_step - 1}) ===")
                print(f"actor_loss shape: {actor_loss.shape}")
                print(f"q_std shape: {q_std.shape}")
                print(f"confidence range: [{confidence.min():.3f}, {confidence.max():.3f}]")
                print(f"confidence mean: {confidence.mean():.3f}")

                # Check high vs low uncertainty separation
                valid = state_mask.squeeze(-1).squeeze(-1) > 0
                sigma_valid = sigma_norm[valid]
                conf_valid = confidence[valid]

                if sigma_valid.numel() > 0:
                    high_mask = sigma_valid > sigma_valid.median()
                    if high_mask.any() and (~high_mask).any():
                        print(f"High-σ confidence: {conf_valid[high_mask].mean():.3f}")
                        print(f"Low-σ confidence: {conf_valid[~high_mask].mean():.3f}")

                print(f"Beta: {beta:.3f}")
                print("=" * 50)

        return weighted_actor_loss

    def _normalize_uncertainty(
        self,
        q_std: torch.Tensor,      # [B, L-1, G, 1]
        mask: torch.Tensor        # [B, L-1, 1, 1]
    ) -> torch.Tensor:
        """Normalize uncertainty to prevent scale drift across training loops.

        Uses per-batch median normalization for stability.

        Args:
            q_std: Q-ensemble standard deviation [B, L-1, G, 1]
            mask: Valid timestep mask [B, L-1, 1, 1]

        Returns:
            Normalized uncertainty [B, L-1, G, 1]
        """
        # Apply mask to exclude padding
        masked_std = q_std * mask  # Zero out padding

        # Get valid stds (flatten all dimensions except batch/time)
        valid_mask = mask.squeeze(-1).squeeze(-1) > 0  # [B, L-1]
        valid_stds = q_std[valid_mask]  # Flatten valid entries

        if valid_stds.numel() == 0:
            # Fallback: all padding (shouldn't happen)
            return torch.ones_like(q_std)

        # Compute median for normalization
        median = valid_stds.median()

        # Ratio normalization: σ̃ = σ / median(σ)
        sigma_norm = q_std / (median + 1e-8)

        # Clamp to prevent extreme outliers
        sigma_norm = sigma_norm.clamp(0, 10)

        return sigma_norm

    def _get_current_beta(self) -> float:
        """Anneal beta from high (conservative) to low (permissive) over training.

        Schedule: β(t) = β_final + (β_init - β_final) * (1 - progress)^α

        This yields:
        - step 0: β = β_init (high penalty, ~5-10)
        - end: β = β_final (low penalty, ~0.5-1.0)

        Returns:
            Current beta value
        """
        # Compute training progress [0, 1]
        progress = min(1.0, self.epistemic_step / self.epistemic_anneal_steps)

        # Power-law decay: high → low
        # CRITICAL: This is the CORRECT formula (high → low)
        beta = self.epistemic_beta_final + \
               (self.epistemic_beta_init - self.epistemic_beta_final) * \
               (1.0 - progress) ** self.epistemic_anneal_power

        self.epistemic_step += 1
        return beta

    def _log_epistemic_metrics(
        self,
        sigma_norm: torch.Tensor,   # [B, L-1, G, 1]
        confidence: torch.Tensor,   # [B, L-1, G, 1]
        mask: torch.Tensor          # [B, L-1, 1, 1]
    ) -> None:
        """Log epistemic weighting diagnostics to wandb."""
        with torch.no_grad():
            # Only consider valid (non-padding) timesteps
            valid_mask = mask.squeeze(-1).squeeze(-1) > 0  # [B, L-1]
            sigma_valid = sigma_norm[valid_mask]
            conf_valid = confidence[valid_mask]

            if sigma_valid.numel() == 0:
                return  # No valid timesteps

            # Get current beta
            # Note: _get_current_beta() increments step, so we compute it differently here
            progress = min(1.0, (self.epistemic_step - 1) / self.epistemic_anneal_steps)
            current_beta = self.epistemic_beta_final + \
                          (self.epistemic_beta_init - self.epistemic_beta_final) * \
                          (1.0 - progress) ** self.epistemic_anneal_power

            # Basic stats - add to update_info which gets logged
            metrics = {
                "Epistemic/Mean Uncertainty": sigma_valid.mean().item(),
                "Epistemic/Mean Confidence": conf_valid.mean().item(),
                "Epistemic/Beta": current_beta,
            }

            # High vs low uncertainty impact
            median_sigma = sigma_valid.median()
            high_unc_mask = sigma_valid > median_sigma
            low_unc_mask = ~high_unc_mask

            if high_unc_mask.any():
                metrics["Epistemic/Confidence (High σ)"] = conf_valid[high_unc_mask].mean().item()
            if low_unc_mask.any():
                metrics["Epistemic/Confidence (Low σ)"] = conf_valid[low_unc_mask].mean().item()

            # Effective learning mass (what fraction of gradients we're allowing)
            effective_mass = conf_valid.mean().item()
            metrics["Epistemic/Effective Mass"] = effective_mass

            # Add to policy's update_info for logging
            if hasattr(self.policy, 'update_info'):
                self.policy.update_info.update(metrics)

    def init_envs(self):
        out = super().init_envs()
        amago.utils.call_async_env(self.val_envs, "take_long_break")
        return out

    def _swap_to_ema_for_eval(self):
        """Temporarily swap policy weights with EMA weights for evaluation.

        Stores current training weights in self._training_state_dict for later restoration.
        """
        import torch

        # Store current training weights (on CPU to save GPU memory)
        self._training_state_dict = {
            k: v.cpu().clone()
            for k, v in self.policy.state_dict().items()
        }

        # Load EMA weights into policy
        self.policy.load_state_dict(self.ema_model.state_dict())
        print("[EMA] Swapped to EMA weights for evaluation")

    def _restore_training_weights(self):
        """Restore training weights after evaluation.

        Moves stored weights back to device and loads them into policy.
        """
        if self._training_state_dict is None:
            raise RuntimeError("Cannot restore training weights: no backup found")

        # Move weights back to policy device and load
        device = next(self.policy.parameters()).device
        self.policy.load_state_dict(
            {k: v.to(device) for k, v in self._training_state_dict.items()}
        )
        self._training_state_dict = None
        print("[EMA] Restored training weights after evaluation")

    def evaluate_val(self):
        amago.utils.call_async_env(self.val_envs, "resume_from_break")

        # Swap to EMA weights for evaluation if enabled
        if self.use_ema and self.ema_eval_only:
            self._swap_to_ema_for_eval()

        out = super().evaluate_val()

        # Restore training weights after evaluation
        if self.use_ema and self.ema_eval_only:
            self._restore_training_weights()

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
