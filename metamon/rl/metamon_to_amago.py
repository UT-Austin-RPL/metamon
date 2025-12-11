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
from metamon.il.model import TransformerTurnEmbedding, PerceiverTurnEmbedding
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


def unknown_token_mask(tokens, skip_prob: float = 0.2, batch_max_prob: float = 0.33):
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
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cached_kl_data = None

    def forward(self, batch, log_step: bool):
        """Forward pass with caching for dynamic damping efficiency.

        Args:
            batch: Batch of RL data from amago.loading.Batch
            log_step: Whether this is a logging step

        Computes and caches intermediate values (trajectory embeddings, observations)
        that can be reused by KL regularization without expensive recomputation.
        """
        # Reset cache
        self.cached_kl_data = None

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

        # Now call parent's forward, which will use these same values
        # Note: Parent will recompute o and s_rep, but this is unavoidable without
        # copying the entire forward() logic. The key win is that _compute_kl_loss()
        # can reuse our cached values, eliminating its expensive recomputation.
        critic_loss, actor_loss = super().forward(batch, log_step)

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
        """Compute RL loss with optional dynamic damping (KL regularization)."""
        # Call parent to get standard actor/critic losses
        loss_dict = super().compute_loss(batch, log_step)

        # Add KL regularization if dynamic damping is enabled
        if self.dd_state is not None and self.dd_config.enabled:
            kl_loss, kl_metrics = self._compute_kl_loss(batch, log_step)

            # Add KL loss to actor loss
            loss_dict["Actor Loss"] = loss_dict["Actor Loss"] + kl_loss

            # Add KL metrics to loss dict for logging
            loss_dict.update(kl_metrics)

            # Debug: Print metrics being logged (commented out to reduce log spam)
            # if log_step:
            #     print(f"[DEBUG] Damping metrics: KL={kl_metrics.get('KL Divergence', 'N/A'):.4f}, "
            #           f"Entropy={kl_metrics.get('Policy Entropy', 'N/A'):.4f}, "
            #           f"Keys in loss_dict: {list(kl_metrics.keys())}")

            # Track KL for adaptive control (sliding window of last N steps)
            if "KL Divergence" in kl_metrics:
                self.kl_window.append(kl_metrics["KL Divergence"])
                self.dd_step_counter += 1

                # Adapt controller every N steps based on LOCAL KL window (not entire epoch)
                if self.dd_step_counter >= self.dd_adapt_interval and len(self.kl_window) >= 10:
                    mean_kl = float(np.mean(self.kl_window))
                    self.dd_state.adapt_from_observed_kl(self.optimizer, mean_kl)

                    # Adaptation print (commented out to reduce log spam)
                    # print(f"[Dynamic Damping] Adapted at step {self.dd_step_counter}: "
                    #       f"mean_kl={mean_kl:.4f}, kl_coef={self.dd_state.kl_coef:.4f}, "
                    #       f"lr={self.optimizer.param_groups[0]['lr']:.6f}", flush=True)

                    # Reset step counter, keep window rolling (deque auto-manages size)
                    self.dd_step_counter = 0
        # else:
        #     if log_step:
        #         print(f"[DEBUG] Damping NOT enabled: dd_state={self.dd_state is not None}, "
        #               f"config.enabled={self.dd_config.enabled if self.dd_config else 'N/A'}")

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

    def train_step(self, batch: Batch, log_step: bool):
        """Training step with dynamic damping schedule updates and adaptive control."""
        # Update damping schedules before training step
        if self.dd_state is not None and self.dd_config.enabled:
            self.dd_state.update_schedules()

        # Perform standard training step
        metrics = super().train_step(batch, log_step)

        return metrics

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
