import json
import os
from pathlib import Path
import warnings
from typing import Optional, Type

warnings.filterwarnings("ignore")


import huggingface_hub
import torch
import amago

import metamon
from metamon.rl.metamon_to_amago import (
    make_placeholder_experiment,
    MetamonDiscrete,
)
from metamon.rl.experimental.ensemble import (
    EnsembleMemberSpec,
    build_heuristic_ensemble_experiment,
)
from metamon.interface import (
    ObservationSpace,
    RewardFunction,
    get_observation_space,
    get_reward_function,
    get_action_space,
    TokenizedObservationSpace,
    ActionSpace,
)
from metamon.tokenizer import PokemonTokenizer, get_tokenizer

from metamon.config import METAMON_CACHE_DIR

if METAMON_CACHE_DIR is None:
    raise ValueError("Set METAMON_CACHE_DIR environment variable")
# downloads checkpoints to the metamon cache dir where we're putting all the other data
MODEL_DOWNLOAD_DIR = os.path.join(METAMON_CACHE_DIR, "pretrained_models")

# registry for pretrained models
ALL_PRETRAINED_MODELS = {}


ENSEMBLE_PRESETS_PATH = (
    Path(__file__).parent / "experimental/ensemble/ensemble_presets.json"
)


def _load_ensemble_member_presets():
    raw_presets = json.loads(ENSEMBLE_PRESETS_PATH.read_text())
    return {
        name: [
            EnsembleMemberSpec(
                **{
                    **spec,
                    "proposal_roles": tuple(spec.get("proposal_roles", [])),
                }
            )
            for spec in member_specs
        ]
        for name, member_specs in raw_presets.items()
    }


def pretrained_model(name: Optional[str] = None):
    """
    Decorator to register pretrained model classes.

    Args:
        name: Optional custom name for the model. If not provided, uses the class name.

    Usage:
        @pretrained_model()
        class MyModel(PretrainedModel):
            pass

        @pretrained_model("CustomName")
        class AnotherModel(PretrainedModel):
            pass
    """

    def _register(cls):
        model_name = name if name is not None else cls.__name__
        if model_name in ALL_PRETRAINED_MODELS:
            raise ValueError(f"Pretrained model '{model_name}' is already registered!")
        ALL_PRETRAINED_MODELS[model_name] = cls
        return cls

    return _register


def get_pretrained_model_names():
    return sorted(ALL_PRETRAINED_MODELS.keys())


def get_pretrained_model(name: str):
    if name not in ALL_PRETRAINED_MODELS:
        raise ValueError(
            f"Unknown pretrained model '{name}' (available models: {get_pretrained_model_names()})"
        )
    return ALL_PRETRAINED_MODELS[name]()


class PretrainedModel:
    """
    Create an AMAGO agent and load a pretrained checkpoint from the HuggingFace Hub.

    This class handles downloading pretrained model weights from HuggingFace Hub,
    configuring the model architecture using gin files, and initializing the
    evaluation experiment.

    Args:
        model_gin_config: Path to gin config file that modifies the model architecture
            (layers, size, etc.)
        train_gin_config: Path to training gin config file. Does not have to be 1:1
            with training, but should match any architecture changes that were used.
        model_name: Model identifier used to locate the model in the HuggingFace Hub.
        tokenizer: Tokenizer for the text component of the observation space.
        observation_space: Observation space configuration. Uses original paper
            observation space by default.
        action_space: Action space configuration. The paper action space is now
            called MinimalActionSpace.
        reward_function: Reward function configuration. Uses original paper reward
            function by default.
        hf_cache_dir: Cache directory for HuggingFace Hub downloads. Note that
            these checkpoint files are large.
        default_checkpoint: Default checkpoint epoch to load. 40 corresponds to
            approximately 1M gradient steps with original paper training settings.
        gin_overrides: Optional dictionary of one-off gin overrides if there's a small tweak to an existing config file.
        battle_backend: The correct default battle backend to use during evaluations of this agent.
            Should indicate a version of the backend pokemon logic that mostly closely replicates the
            version used to collect data and/or reconstruct replays for training.
                'poke-env' is deprecated; maintains the original paper's models.
                'metamon' is the lateset version
                'pokeagent' maintains policies trained (and used as the organizer baselines) during the PokéAgent Challenge
        dataset_config: Path to the dataset config YAML that describes the training data
            composition (replay weights, self-play subsets, custom replay dirs). None for
            HuggingFace models where the dataset composition is lost to time...
        action_temperature: Temperature for temperature-based sampling. Higher temperature means more exploration. Default is 1.0 (no scaling).
    """

    HF_REPO_ID = "jakegrigsby/metamon"

    def __init__(
        self,
        model_gin_config: str,
        train_gin_config: str,
        model_name: str,
        tokenizer: PokemonTokenizer = get_tokenizer("allreplays-v3"),
        observation_space: ObservationSpace = get_observation_space(
            "DefaultObservationSpace"
        ),
        action_space: ActionSpace = get_action_space("DefaultActionSpace"),
        reward_function: RewardFunction = get_reward_function("DefaultShapedReward"),
        hf_cache_dir: Optional[str] = None,
        default_checkpoint: int = 40,
        gin_overrides: Optional[dict] = None,
        battle_backend: str = "metamon",
        dataset_config: Optional[str] = None,
    ):
        self.model_name = model_name
        self.model_gin_config = model_gin_config
        self.train_gin_config = train_gin_config
        self.battle_backend = battle_backend
        self.dataset_config = dataset_config
        self.model_gin_config_path = os.path.join(
            metamon.rl.MODEL_CONFIG_DIR, self.model_gin_config
        )
        self.train_gin_config_path = os.path.join(
            metamon.rl.TRAINING_CONFIG_DIR, self.train_gin_config
        )
        self.hf_cache_dir = hf_cache_dir or MODEL_DOWNLOAD_DIR
        self.tokenizer = tokenizer
        self.observation_space = TokenizedObservationSpace(
            base_obs_space=observation_space,
            tokenizer=tokenizer,
        )
        self.action_space = action_space
        self.reward_function = reward_function
        self.default_checkpoint = default_checkpoint
        self.gin_overrides = gin_overrides
        os.makedirs(self.hf_cache_dir, exist_ok=True)

    @property
    def base_config(self) -> dict:
        """
        Override to set one-off changes to the gin config files

        By default, sets the tokenizer and enables faster initialization.
        """
        config = {
            "MetamonTstepEncoder.tokenizer": self.tokenizer,
            # skip cpu-intensive init, because we're going to be replacing the weights
            # with a checkpoint anyway....
            "amago.nets.transformer.SigmaReparam.fast_init": True,
        }
        if self.gin_overrides is not None:
            config.update(self.gin_overrides)
        return config

    def get_path_to_checkpoint(self, checkpoint: int) -> str:
        # Download checkpoint from HF Hub
        checkpoint_path = huggingface_hub.hf_hub_download(
            repo_id=self.HF_REPO_ID,
            filename=f"{self.model_name}/ckpts/policy_weights/policy_epoch_{checkpoint}.pt",
            cache_dir=self.hf_cache_dir,
        )
        return checkpoint_path

    def initialize_agent(
        self,
        checkpoint: Optional[int] = None,
        log: bool = False,
        action_temperature: float = 1.0,
    ) -> amago.Experiment:
        # use the base config and the gin file to configure the model
        amago.cli_utils.use_config(
            self.base_config | {"MetamonDiscrete.temperature": action_temperature},
            [self.model_gin_config_path, self.train_gin_config_path],
            finalize=False,
        )
        checkpoint = checkpoint if checkpoint is not None else self.default_checkpoint
        ckpt_path = self.get_path_to_checkpoint(checkpoint)
        ckpt_base_dir = str(Path(ckpt_path).parents[2])
        # build an experiment
        experiment = make_placeholder_experiment(
            ckpt_base_dir=ckpt_base_dir,
            run_name=self.model_name,
            log=log,
            observation_space=self.observation_space,
            action_space=self.action_space,
        )
        # starting the experiment will build the initial model
        experiment.start()
        if checkpoint > 0:
            ckpt_state = torch.load(ckpt_path, map_location="cpu")
            model_state = experiment.policy.state_dict()
            self._validate_checkpoint(ckpt_state, model_state)
            experiment.policy.load_state_dict(ckpt_state, strict=True)
            experiment.policy.on_checkpoint_loaded(is_resume=False)
        return experiment

    @staticmethod
    def _validate_checkpoint(ckpt_state: dict, model_state: dict) -> None:
        ckpt_keys = set(ckpt_state.keys())
        model_keys = set(model_state.keys())
        missing = model_keys - ckpt_keys
        unexpected = ckpt_keys - model_keys
        if missing:
            raise RuntimeError(
                f"Checkpoint is missing {len(missing)} keys expected by the model:\n"
                + "\n".join(f"  {k}" for k in sorted(missing))
            )
        if unexpected:
            raise RuntimeError(
                f"Checkpoint has {len(unexpected)} unexpected keys not in the model:\n"
                + "\n".join(f"  {k}" for k in sorted(unexpected))
            )
        shape_mismatches = []
        for k in model_keys:
            if model_state[k].shape != ckpt_state[k].shape:
                shape_mismatches.append(
                    f"  {k}: model={list(model_state[k].shape)} vs ckpt={list(ckpt_state[k].shape)}"
                )
        if shape_mismatches:
            raise RuntimeError(
                f"Shape mismatch for {len(shape_mismatches)} parameters:\n"
                + "\n".join(shape_mismatches)
            )
        ckpt_params = sum(p.numel() for p in ckpt_state.values())
        model_params = sum(p.numel() for p in model_state.values())
        print(
            f"Checkpoint validated: {len(model_keys)} keys, "
            f"{model_params:,} params (model) == {ckpt_params:,} params (ckpt)"
        )


class LocalPretrainedModel(PretrainedModel):
    """
    Evaluate a model from a custom training run.

    Args:
        amago_ckpt_dir: Path to the AMAGO checkpoint directory (e.g. --save_dir from the training script)
        model_name: The name of the training run (e.g. --run_name from the training script)
        Additional arguments follow the PretrainedModel
    """

    def __init__(self, amago_ckpt_dir: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_ckpt_dir = os.path.join(amago_ckpt_dir, self.model_name, "ckpts")
        if not os.path.exists(self.local_ckpt_dir):
            raise FileNotFoundError(
                f"Checkpoint directory {self.local_ckpt_dir} was not found. Check the amago_ckpt_dir and model_name arguments."
            )

    def get_path_to_checkpoint(self, checkpoint: int) -> str:
        return os.path.join(
            self.local_ckpt_dir,
            "policy_weights",
            f"policy_epoch_{checkpoint}.pt",
        )


class LocalFinetunedModel(LocalPretrainedModel):
    """
    Evaluate a model from a finetuning run.

    Same as LocalPretrainedModel but takes care of setting the config files.
    If you used a custom train_gin_config or reward_function, pass them here.

    Args:
        base_model: The base model type that was finetuned.
        amago_ckpt_dir: Path to the AMAGO checkpoint directory (e.g. --save_dir from the training script)
        model_name: The name of the training run (e.g. --run_name from the training script)
        default_checkpoint: The checkpoint number to load by default (e.g., the last epoch number)
        train_gin_config: The gin config file to use for training. Defaults to the same as used by the base model (like the finetuning script does).
        reward_function: The reward function to use. Defaults to the same as used by the base model (like the finetuning script does).
        dataset_config: Path to the dataset config YAML used for this finetuning run (or None if unknown).
    """

    def __init__(
        self,
        base_model: Type[PretrainedModel],
        amago_ckpt_dir: str,
        model_name: str,
        default_checkpoint: int,
        train_gin_config: Optional[str] = None,
        reward_function: Optional[RewardFunction] = None,
        battle_backend: Optional[str] = None,
        dataset_config: Optional[str] = None,
    ):
        base_model = base_model()
        train_gin_config = train_gin_config or base_model.train_gin_config
        reward_function = reward_function or base_model.reward_function
        battle_backend = battle_backend or base_model.battle_backend
        super().__init__(
            amago_ckpt_dir=amago_ckpt_dir,
            model_name=model_name,
            train_gin_config=train_gin_config,
            default_checkpoint=default_checkpoint,
            model_gin_config=base_model.model_gin_config,
            tokenizer=base_model.tokenizer,
            observation_space=base_model.observation_space,
            action_space=base_model.action_space,
            reward_function=reward_function,
            battle_backend=battle_backend,
            dataset_config=dataset_config,
        )


####################################################
## Paper Policies (Nov 2024 - Feb 2025) Gens 1-4 ###
####################################################


@pretrained_model()
class SmallIL(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="small-il",
            model_gin_config="small_agent.gin",
            train_gin_config="il.gin",
            default_checkpoint=40,
            action_space=get_action_space("MinimalActionSpace"),
            battle_backend="poke-env",
        )


@pretrained_model()
class SmallILFA(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="small-il-filled-actions",
            model_gin_config="small_agent.gin",
            train_gin_config="il.gin",
            default_checkpoint=40,
            action_space=get_action_space("MinimalActionSpace"),
            battle_backend="poke-env",
        )


@pretrained_model()
class SmallRL(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="small-rl",
            model_gin_config="small_agent.gin",
            train_gin_config="exp_rl.gin",
            default_checkpoint=40,
            action_space=get_action_space("MinimalActionSpace"),
            battle_backend="poke-env",
        )


@pretrained_model()
class SmallRL_ExtremeFilter(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="small-rl-exp-extreme",
            model_gin_config="small_agent.gin",
            train_gin_config="exp_rl.gin",
            default_checkpoint=38,
            action_space=get_action_space("MinimalActionSpace"),
            gin_overrides={
                "amago.agent.exp_filter.beta": 5.0,
                "amago.agent.exp_filter.clip_weights_high": 100.0,
            },
        )


@pretrained_model()
class SmallRL_BinaryFilter(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="small-rl-binary",
            model_gin_config="small_agent.gin",
            train_gin_config="binary_rl.gin",
            default_checkpoint=40,
            action_space=get_action_space("MinimalActionSpace"),
            battle_backend="poke-env",
        )


@pretrained_model()
class SmallRL_Aug(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="small-rl-aug",
            model_gin_config="small_agent.gin",
            train_gin_config="binary_rl.gin",
            default_checkpoint=40,
            action_space=get_action_space("MinimalActionSpace"),
            battle_backend="poke-env",
        )


@pretrained_model()
class SmallRL_MaxQ(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="small-rl-maxq",
            model_gin_config="small_agent.gin",
            train_gin_config="binary_maxq_rl.gin",
            default_checkpoint=40,
            action_space=get_action_space("MinimalActionSpace"),
            battle_backend="poke-env",
        )


@pretrained_model()
class MediumIL(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="medium-il",
            model_gin_config="medium_agent.gin",
            train_gin_config="il.gin",
            default_checkpoint=40,
            action_space=get_action_space("MinimalActionSpace"),
            battle_backend="poke-env",
        )


@pretrained_model()
class MediumRL(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="medium-rl",
            model_gin_config="medium_agent.gin",
            train_gin_config="exp_rl.gin",
            default_checkpoint=40,
            action_space=get_action_space("MinimalActionSpace"),
            battle_backend="poke-env",
        )


@pretrained_model()
class MediumRL_Aug(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="medium-rl-aug",
            model_gin_config="medium_agent.gin",
            train_gin_config="exp_rl.gin",
            default_checkpoint=40,
            action_space=get_action_space("MinimalActionSpace"),
            battle_backend="poke-env",
        )


@pretrained_model()
class MediumRL_MaxQ(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="medium-rl-maxq",
            model_gin_config="medium_agent.gin",
            train_gin_config="binary_maxq_rl.gin",
            default_checkpoint=40,
            action_space=get_action_space("MinimalActionSpace"),
            battle_backend="poke-env",
        )


@pretrained_model()
class LargeRL(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="large-rl",
            model_gin_config="large_agent.gin",
            train_gin_config="exp_rl.gin",
            default_checkpoint=40,
            action_space=get_action_space("MinimalActionSpace"),
            battle_backend="poke-env",
        )


@pretrained_model()
class LargeIL(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="large-il",
            model_gin_config="large_agent.gin",
            train_gin_config="il.gin",
            default_checkpoint=40,
            action_space=get_action_space("MinimalActionSpace"),
            battle_backend="poke-env",
        )


@pretrained_model()
class SyntheticRLV0(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="synthetic-rl-v0",
            model_gin_config="synthetic_agent.gin",
            train_gin_config="binary_rl.gin",
            default_checkpoint=40,
            action_space=get_action_space("MinimalActionSpace"),
            battle_backend="poke-env",
        )


@pretrained_model()
class SyntheticRLV1(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="synthetic-rl-v1",
            model_gin_config="synthetic_agent.gin",
            train_gin_config="binary_rl.gin",
            default_checkpoint=40,
            action_space=get_action_space("MinimalActionSpace"),
            battle_backend="poke-env",
        )


@pretrained_model()
class SyntheticRLV1_SelfPlay(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="synthetic-rl-v1+sp",
            model_gin_config="synthetic_agent.gin",
            train_gin_config="binary_rl.gin",
            default_checkpoint=48,
            action_space=get_action_space("MinimalActionSpace"),
            battle_backend="poke-env",
        )


@pretrained_model()
class SyntheticRLV1_PlusPlus(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="synthetic-rl-v1++",
            model_gin_config="synthetic_agent.gin",
            train_gin_config="binary_maxq_rl.gin",
            default_checkpoint=38,
            action_space=get_action_space("MinimalActionSpace"),
            battle_backend="poke-env",
        )


@pretrained_model()
class SyntheticRLV2(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="synthetic-rl-v2",
            model_gin_config="synthetic_multitaskagent.gin",
            train_gin_config="binary_rl.gin",
            default_checkpoint=48,
            action_space=get_action_space("MinimalActionSpace"),
            battle_backend="poke-env",
        )


###########################################################################
## PokéAgent Challenge Policies (June 2025 - November 2025) Gens1-4 & 9 ###
###########################################################################


"""
"PAC-" prefixed observation spaces trigger a hack to reintroduce a bug that impacted models trained during the challenge.
For now, this lets these policies continue to collect reasonable trajectories for the "metamon" battle backend.
"""


@pretrained_model()
class SmallRLGen9Beta(PretrainedModel):
    """
    Prototype for Gen9 agents. Trained entirely on human replays (parsed-replays v3). Was finetuned
    from a previous Gen9 attempt in order to switch from "ExpandedObservationSpace" to "TeamPreviewObservationSpace".
    TeamPreviewObservationSpace adds the opponent's species names if revealed before the start of the battle, which
    is only relevant to Gen 9.

    Few formal evals done, but it appears roughly equivalent to the original replays-only policies from the paper
    (e.g., LargeRL), except that it also plays Gen9 at about that same level.
    """

    def __init__(self):
        super().__init__(
            model_name="small-rl-gen9beta",
            model_gin_config="small_multitaskagent.gin",
            train_gin_config="exp_rl.gin",
            # this model was finetuned from a previous gen9 attempt and has
            # trained for more than 24 total epochs...
            default_checkpoint=24,
            action_space=get_action_space("DefaultActionSpace"),
            observation_space=get_observation_space("PAC-TeamPreviewObservationSpace"),
            tokenizer=get_tokenizer("DefaultObservationSpace-v1"),
            # temporarily forced to flash attention until we can verify numerical stability
            # of a switch to a standard pytorch sliding window inference alternative
            gin_overrides={
                "amago.nets.traj_encoders.TformerTrajEncoder.attention_type": amago.nets.transformer.FlashAttention,
                "amago.nets.transformer.FlashAttention.window_size": (32, 0),
            },
            battle_backend="pokeagent",
        )


@pretrained_model()
class Minikazam(PretrainedModel):
    """
    An attempt to create an affordable starting point for finetuning.

    Small RNN trained on parsed-replays v4 and ~5M self-play battles.

    Detailed evals compiled here: https://docs.google.com/spreadsheets/d/1GU7-Jh0MkIKWhiS1WNQiPfv49WIajanUF4MjKeghMAc/edit?usp=sharing
    """

    def __init__(self):
        super().__init__(
            model_name="minikazam",
            model_gin_config="minikazam.gin",
            train_gin_config="binary_rl.gin",
            default_checkpoint=40,
            action_space=get_action_space("DefaultActionSpace"),
            observation_space=get_observation_space("PAC-OpponentMoveObservationSpace"),
            reward_function=get_reward_function("AggressiveShapedReward"),
            tokenizer=get_tokenizer("DefaultObservationSpace-v1"),
            battle_backend="pokeagent",
            gin_overrides={
                "MetamonPerceiverTstepEncoder.tokenizer": get_tokenizer(
                    "DefaultObservationSpace-v1"
                ),
            },
        )


@pretrained_model()
class Abra(PretrainedModel):
    """
    First of a new series of training runs replicating the "Synthetic" agents from the paper *with Gen 9*.

    Trained on parsed-replays v3 with ~100k self-play battles per OU generation. Gen 9 battles collected amongst checkpoints
    of SmallRLGen9Beta and a previous Gen 9 test. Gen 1-4 used battles from the stronger Synthetic agents. Most of these were
    played on the PokéAgent Challenge ladder, at a time when the organizer baselines made up 99%+ of active battles.

    Performance in Gen1-4 is comparable to early Synthetic policies like SyntheticRLV1, but nowhere close to SyntheticRLV2.

    50% GXE in Gen9OU playing with sample teams ("competitive" TeamSet) on the human ladder.
    """

    def __init__(self):
        super().__init__(
            model_name="abra",
            model_gin_config="medium_multitaskagent.gin",
            train_gin_config="binary_rl.gin",
            default_checkpoint=40,
            action_space=get_action_space("DefaultActionSpace"),
            observation_space=get_observation_space("PAC-TeamPreviewObservationSpace"),
            tokenizer=get_tokenizer("DefaultObservationSpace-v1"),
            gin_overrides={
                "amago.nets.traj_encoders.TformerTrajEncoder.attention_type": amago.nets.transformer.FlashAttention,
                "amago.nets.transformer.FlashAttention.window_size": (32, 0),
            },
            battle_backend="pokeagent",
        )


@pretrained_model()
class Kadabra(PretrainedModel):
    """
    A second attempt at self-play on gens1-4 & 9 that was featured in the PokéAgent Challenge.

    This policy held the top organizer gen9ou rank for most of the "practice ladder" period in Summer 2025.
    """

    def __init__(self):
        super().__init__(
            model_name="kadabra",
            model_gin_config="medium_multitaskagent.gin",
            train_gin_config="binary_rl.gin",
            default_checkpoint=46,
            action_space=get_action_space("DefaultActionSpace"),
            observation_space=get_observation_space("TeamPreviewObservationSpace"),
            tokenizer=get_tokenizer("DefaultObservationSpace-v1"),
            battle_backend="pokeagent",
            gin_overrides={
                "amago.nets.traj_encoders.TformerTrajEncoder.attention_type": amago.nets.transformer.FlashAttention,
                "amago.nets.transformer.FlashAttention.window_size": (32, 0),
            },
        )


@pretrained_model()
class Kadabra2(PretrainedModel):
    """
    A third attempt at self-play on gens1-4 & 9 that was featured in the PokéAgent Challenge.

    Confusingly, this policy played under the username "PAC-MM-Alakazam" for most of the challange, and held
    the top organizer gen9ou rank at the end of the Summer 2025 practice ladder. Checkpoints have been renamed
    for public release such that the best policy with this architecture gets to be "Alakazam" :)

    This marks the first time where performance of policies *trained on Gen9OU* roughly match the paper policies in Gens1-4;
    all policies below can play Gen9OU without sacrificing significant performance in Gens1-4.
    """

    def __init__(self):
        super().__init__(
            model_name="kadabra2",
            model_gin_config="alakazam2.gin",
            train_gin_config="alakazam2.gin",
            default_checkpoint=44,
            action_space=get_action_space("DefaultActionSpace"),
            observation_space=get_observation_space("PAC-OpponentMoveObservationSpace"),
            reward_function=get_reward_function("AggressiveShapedReward"),
            tokenizer=get_tokenizer("DefaultObservationSpace-v1"),
            battle_backend="pokeagent",
            gin_overrides={
                "MetamonPerceiverTstepEncoder.tokenizer": get_tokenizer(
                    "DefaultObservationSpace-v1"
                ),
            },
        )


@pretrained_model()
class Kadabra3(PretrainedModel):
    """
    A fourth attempt at self-play on gens1-4 & 9 that was featured in the PokéAgent Challenge.

    This policy played under the username "PAC-MM-Wildcard" or "PAC-MM-Mystery" during the qualification period.
    If it had been pubilcly available, it would have qualified as the #2 seed in Gen1OU and #3 seed in Gen9OU.
    """

    def __init__(self):
        super().__init__(
            model_name="kadabra3",
            model_gin_config="alakazam2.gin",
            train_gin_config="alakazam3.gin",
            default_checkpoint=20,
            action_space=get_action_space("DefaultActionSpace"),
            observation_space=get_observation_space("PAC-OpponentMoveObservationSpace"),
            reward_function=get_reward_function("AggressiveShapedReward"),
            tokenizer=get_tokenizer("DefaultObservationSpace-v1"),
            battle_backend="pokeagent",
            gin_overrides={
                "MetamonPerceiverTstepEncoder.tokenizer": get_tokenizer(
                    "DefaultObservationSpace-v1"
                ),
            },
        )


@pretrained_model()
class Kadabra4(PretrainedModel):
    """
    A fifth attempt at self-play on gens1-4 & 9 that was featured in the PokéAgent Challenge.

    The final PokéAgent Challenge era dataset was 11.6M self-play battles + parsed-replays-v4.

    This policy played under the username "PAC-MM-Mystery" or "PAC-MM-Wildcard" during the qualification period.
    If it had been pubilcally available, it would have qualified as the #1 seed in Gen1OU and #2 seed in Gen9OU
    (behind FoulPlay).

    Most of the performance gains from Kadabra2 --> Kadabra4 are seen in diverse team evaluations (i.e., "modern_replays_v2" TeamSet).
    """

    def __init__(self):
        super().__init__(
            model_name="kadabra4",
            model_gin_config="alakazam4.gin",
            train_gin_config="alakazam3.gin",
            default_checkpoint=50,
            action_space=get_action_space("DefaultActionSpace"),
            observation_space=get_observation_space("PAC-OpponentMoveObservationSpace"),
            reward_function=get_reward_function("AggressiveShapedReward"),
            tokenizer=get_tokenizer("DefaultObservationSpace-v1"),
            battle_backend="pokeagent",
            gin_overrides={
                "MetamonPerceiverTstepEncoder.tokenizer": get_tokenizer(
                    "DefaultObservationSpace-v1"
                ),
            },
        )


@pretrained_model()
class Alakazam(PretrainedModel):
    """
    This policy patches a bug (https://github.com/UT-Austin-RPL/metamon/pull/54) that impacted all PokéAgnet Challenge training runs.

    We finetuned Kadabra4 on a new version of the self-play dataset that was patched to include tera types.
    The "Kadabra*" policies now intentionally *preserve* the bug for backwards compatibility, so this policy gains a slight
    edge when evaluated today (after the bug was patched).

    This policy never appeared on the PokéAgent Challenge ladder but is called "Alakazam" because it is the last model
    of this size (~50M params) to be trained on the PokéAgent Challenge dataset.
    """

    def __init__(self):
        super().__init__(
            model_name="alakazam",
            model_gin_config="alakazam4.gin",
            train_gin_config="alakazam3.gin",
            default_checkpoint=8,
            action_space=get_action_space("DefaultActionSpace"),
            observation_space=get_observation_space("OpponentMoveObservationSpace"),
            reward_function=get_reward_function("AggressiveShapedReward"),
            tokenizer=get_tokenizer("DefaultObservationSpace-v1"),
            battle_backend="metamon",
            gin_overrides={
                "MetamonPerceiverTstepEncoder.tokenizer": get_tokenizer(
                    "DefaultObservationSpace-v1"
                ),
            },
        )


@pretrained_model()
class Superkazam(PretrainedModel):
    """
    Revisits the PokéAgent Challenge dataset at a model size closer to the paper's SyntheticRLV2 configuration (~140M params).

    - PokéAgent Challenge self-play dataset (11.6M battles)
    - (Human) parsed-replays-v4 (4M battles)

    Evals against the most important (modern) baselines are available here: https://docs.google.com/spreadsheets/d/1lU8tQ0tnnupY28kIyK6FVtvPmxLSVT9_slLShOhRsqg/edit?usp=sharing
    """

    def __init__(self):
        super().__init__(
            model_name="superkazam",
            model_gin_config="superkazam.gin",
            train_gin_config="alakazam3.gin",
            default_checkpoint=50,
            action_space=get_action_space("DefaultActionSpace"),
            observation_space=get_observation_space("OpponentMoveObservationSpace"),
            reward_function=get_reward_function("AggressiveShapedReward"),
            tokenizer=get_tokenizer("DefaultObservationSpace-v1"),
            battle_backend="metamon",
            gin_overrides={
                "MetamonPerceiverTstepEncoder.tokenizer": get_tokenizer(
                    "DefaultObservationSpace-v1"
                ),
            },
        )


@pretrained_model()
class Kakuna(PretrainedModel):
    """
    The current best Metamon policy.

    Superkazam, finetuned on a dataset of self-play battles collected at increased temperature for exploration and value learning (+7.8M battles).

    After > 700 total games played over a span of a month, we estimate GXEs vs. humans (with "competitive" TeamSet) of:

    gen1ou: ~82%
    gen2ou: ~70%
    gen3ou: ~63%
    gen4ou: ~64%
    gen9ou: ~71%

    Evals against the most important (modern) metamon baselines are available here: https://docs.google.com/spreadsheets/d/1lU8tQ0tnnupY28kIyK6FVtvPmxLSVT9_slLShOhRsqg/edit?usp=sharing
    """

    def __init__(self):
        super().__init__(
            model_name="kakuna",
            model_gin_config="superkazam.gin",
            train_gin_config="kakuna.gin",
            default_checkpoint=34,
            action_space=get_action_space("DefaultActionSpace"),
            observation_space=get_observation_space("OpponentMoveObservationSpace"),
            reward_function=get_reward_function("AggressiveShapedReward"),
            tokenizer=get_tokenizer("DefaultObservationSpace-v1"),
            battle_backend="metamon",
            gin_overrides={
                "MetamonPerceiverTstepEncoder.tokenizer": get_tokenizer(
                    "DefaultObservationSpace-v1"
                ),
            },
        )


#################################################
### Gen 1 Specialists (Feb 2026 - April 2026) ###
#################################################
"""
Post-PokéAgent Challenge effort to reach the top of the Gen 1 OU leaderboard.

All of these policies are trained to play Gen 1 OU specifically.

The many "V2A" runs are small-scale (~12-15M param) RL hparam ablations. 
They all have pretty similar performance, in the range between SyntheticRLV2
and Kakuna. There isn't much of a reason to use them aside from boosting self-play
diversity. Tauros-v0 scales up the findings on a fresh dataset and is the best
standalone Gen1OU policy in metamon to date.
"""


@pretrained_model()
class V2A(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="v2_small_rl_baseline",
            model_gin_config="smaller_multitaskagent.gin",
            train_gin_config="alakazam3.gin",
            default_checkpoint=26,
            action_space=get_action_space("DefaultActionSpace"),
            observation_space=get_observation_space("OpponentMoveObservationSpace"),
            reward_function=get_reward_function("AggressiveShapedReward"),
            tokenizer=get_tokenizer("DefaultObservationSpace-v1"),
            battle_backend="metamon",
            gin_overrides={
                "MetamonPerceiverTstepEncoder.tokenizer": get_tokenizer(
                    "DefaultObservationSpace-v1"
                ),
            },
        )


@pretrained_model()
class V2ASeed2(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="v2_small_rl_baseline_track_metrics",
            model_gin_config="smaller_multitaskagent.gin",
            train_gin_config="alakazam3.gin",
            default_checkpoint=26,
            action_space=get_action_space("DefaultActionSpace"),
            observation_space=get_observation_space("OpponentMoveObservationSpace"),
            reward_function=get_reward_function("AggressiveShapedReward"),
            tokenizer=get_tokenizer("DefaultObservationSpace-v1"),
            battle_backend="metamon",
            gin_overrides={
                "MetamonPerceiverTstepEncoder.tokenizer": get_tokenizer(
                    "DefaultObservationSpace-v1"
                ),
            },
        )


@pretrained_model()
class V2ABeta01(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="beta_01",
            model_gin_config="smaller_multitaskagent.gin",
            train_gin_config="alakazam_beta0.1.gin",
            default_checkpoint=20,
            action_space=get_action_space("DefaultActionSpace"),
            observation_space=get_observation_space("OpponentMoveObservationSpace"),
            reward_function=get_reward_function("AggressiveShapedReward"),
            tokenizer=get_tokenizer("DefaultObservationSpace-v1"),
            battle_backend="metamon",
            gin_overrides={
                "MetamonPerceiverTstepEncoder.tokenizer": get_tokenizer(
                    "DefaultObservationSpace-v1"
                ),
            },
        )


@pretrained_model()
class V2ABeta1(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="beta_1",
            model_gin_config="smaller_multitaskagent.gin",
            train_gin_config="alakazam_beta1.gin",
            default_checkpoint=20,
            action_space=get_action_space("DefaultActionSpace"),
            observation_space=get_observation_space("OpponentMoveObservationSpace"),
            reward_function=get_reward_function("AggressiveShapedReward"),
            tokenizer=get_tokenizer("DefaultObservationSpace-v1"),
            battle_backend="metamon",
            gin_overrides={
                "MetamonPerceiverTstepEncoder.tokenizer": get_tokenizer(
                    "DefaultObservationSpace-v1"
                ),
            },
        )


@pretrained_model()
class V2ABeta3(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="beta_3",
            model_gin_config="smaller_multitaskagent.gin",
            train_gin_config="alakazam_beta3.gin",
            default_checkpoint=20,
            action_space=get_action_space("DefaultActionSpace"),
            observation_space=get_observation_space("OpponentMoveObservationSpace"),
            reward_function=get_reward_function("AggressiveShapedReward"),
            tokenizer=get_tokenizer("DefaultObservationSpace-v1"),
            battle_backend="metamon",
            gin_overrides={
                "MetamonPerceiverTstepEncoder.tokenizer": get_tokenizer(
                    "DefaultObservationSpace-v1"
                ),
            },
        )


@pretrained_model()
class V2ABeta10(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="beta_10",
            model_gin_config="smaller_multitaskagent.gin",
            train_gin_config="alakazam_beta10.gin",
            default_checkpoint=20,
            action_space=get_action_space("DefaultActionSpace"),
            observation_space=get_observation_space("OpponentMoveObservationSpace"),
            reward_function=get_reward_function("AggressiveShapedReward"),
            tokenizer=get_tokenizer("DefaultObservationSpace-v1"),
            battle_backend="metamon",
            gin_overrides={
                "MetamonPerceiverTstepEncoder.tokenizer": get_tokenizer(
                    "DefaultObservationSpace-v1"
                ),
            },
        )


@pretrained_model()
class V2ABNBeta3HLGauss(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="bn_beta3_hlgauss",
            model_gin_config="smaller_multitaskagent.gin",
            train_gin_config="alakazam_bnorm_beta3_hlgauss.gin",
            default_checkpoint=26,
            action_space=get_action_space("DefaultActionSpace"),
            observation_space=get_observation_space("OpponentMoveObservationSpace"),
            reward_function=get_reward_function("AggressiveShapedReward"),
            tokenizer=get_tokenizer("DefaultObservationSpace-v1"),
            battle_backend="metamon",
            gin_overrides={
                "MetamonPerceiverTstepEncoder.tokenizer": get_tokenizer(
                    "DefaultObservationSpace-v1"
                ),
            },
        )


@pretrained_model()
class V2ABNBeta3HLGaussVanilla(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="bn_beta3_hlgauss_vanilla",
            model_gin_config="smaller_multitaskagent.gin",
            train_gin_config="alakazam_bnorm_beta3_hlgauss_vanilla.gin",
            default_checkpoint=26,
            action_space=get_action_space("DefaultActionSpace"),
            observation_space=get_observation_space("OpponentMoveObservationSpace"),
            reward_function=get_reward_function("AggressiveShapedReward"),
            tokenizer=get_tokenizer("DefaultObservationSpace-v1"),
            battle_backend="metamon",
            gin_overrides={
                "MetamonPerceiverTstepEncoder.tokenizer": get_tokenizer(
                    "DefaultObservationSpace-v1"
                ),
            },
        )


@pretrained_model()
class V2ABNBeta3(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="bn_beta3",
            model_gin_config="smaller_multitaskagent.gin",
            train_gin_config="alakazam_bnorm_beta3.gin",
            default_checkpoint=20,
            action_space=get_action_space("DefaultActionSpace"),
            observation_space=get_observation_space("OpponentMoveObservationSpace"),
            reward_function=get_reward_function("AggressiveShapedReward"),
            tokenizer=get_tokenizer("DefaultObservationSpace-v1"),
            battle_backend="metamon",
            gin_overrides={
                "MetamonPerceiverTstepEncoder.tokenizer": get_tokenizer(
                    "DefaultObservationSpace-v1"
                ),
            },
        )


@pretrained_model()
class V2AMixedGens(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="v2_small_rl_baseline_all_gens",
            model_gin_config="smaller_multitaskagent.gin",
            train_gin_config="alakazam3.gin",
            # this one trained all the way out to epoch 80!
            default_checkpoint=26,
            action_space=get_action_space("DefaultActionSpace"),
            observation_space=get_observation_space("OpponentMoveObservationSpace"),
            reward_function=get_reward_function("AggressiveShapedReward"),
            tokenizer=get_tokenizer("DefaultObservationSpace-v1"),
            battle_backend="metamon",
            gin_overrides={
                "MetamonPerceiverTstepEncoder.tokenizer": get_tokenizer(
                    "DefaultObservationSpace-v1"
                ),
            },
        )


@pretrained_model()
class V2ANoMG(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="v2_small_rl_baseline_nomg_g99",
            model_gin_config="smaller_multitaskagent.gin",
            train_gin_config="alakazam3.gin",
            default_checkpoint=26,
            action_space=get_action_space("DefaultActionSpace"),
            observation_space=get_observation_space("OpponentMoveObservationSpace"),
            reward_function=get_reward_function("AggressiveShapedReward"),
            tokenizer=get_tokenizer("DefaultObservationSpace-v1"),
            battle_backend="metamon",
            gin_overrides={
                "MetamonPerceiverTstepEncoder.tokenizer": get_tokenizer(
                    "DefaultObservationSpace-v1"
                ),
                "MultiTaskAgent.use_multigamma": False,
                "MultiTaskAgent.gamma": 0.99,
            },
        )


@pretrained_model()
class V2AIL(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="v2_small_il_baseline",
            model_gin_config="smaller_multitaskagent.gin",
            train_gin_config="il.gin",
            default_checkpoint=26,
            action_space=get_action_space("DefaultActionSpace"),
            observation_space=get_observation_space("OpponentMoveObservationSpace"),
            reward_function=get_reward_function("AggressiveShapedReward"),
            tokenizer=get_tokenizer("DefaultObservationSpace-v1"),
            battle_backend="metamon",
            gin_overrides={
                "MetamonPerceiverTstepEncoder.tokenizer": get_tokenizer(
                    "DefaultObservationSpace-v1"
                ),
                "MetamonAMAGOExperiment.learning_rate": 1.25e-4,
                "MetamonAMAGOExperiment.lr_warmup_steps": 2000,
            },
        )


@pretrained_model()
class V2AGroupedV2ISFilter(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="v2_small_rl_grouped_v2_isfilter",
            model_gin_config="smaller_multitaskagent_grouped_v2.gin",
            train_gin_config="alakazam3_isfilter.gin",
            default_checkpoint=88,
            action_space=get_action_space("DefaultActionSpace"),
            observation_space=get_observation_space("GroupedObservationSpace"),
            reward_function=get_reward_function("AggressiveShapedReward"),
            tokenizer=get_tokenizer("DefaultObservationSpace-v1"),
            battle_backend="metamon",
            gin_overrides={
                "MetamonGroupedTstepEncoderV2.tokenizer": get_tokenizer(
                    "DefaultObservationSpace-v1"
                ),
            },
        )


@pretrained_model()
class V2AGroupedV2Patched(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="v2_small_rl_grouped_v2_patched",
            model_gin_config="smaller_multitaskagent_grouped_v2.gin",
            train_gin_config="alakazam3.gin",
            default_checkpoint=26,
            action_space=get_action_space("DefaultActionSpace"),
            observation_space=get_observation_space("GroupedObservationSpace"),
            reward_function=get_reward_function("AggressiveShapedReward"),
            tokenizer=get_tokenizer("DefaultObservationSpace-v1"),
            battle_backend="metamon",
            gin_overrides={
                "MetamonGroupedTstepEncoderV2.tokenizer": get_tokenizer(
                    "DefaultObservationSpace-v1"
                ),
            },
        )


@pretrained_model()
class V2AGroupedV2ArchAblation(PretrainedModel):
    """Grouped V2 arch trained on the V2A baseline data mix (pac-base 60%, pac-exploratory 35%).

    Paired with V2AGroupedV2DataAblation to isolate the effect of the Tauros data mix
    vs. the architecture change (smaller_multitaskagent_grouped_v2_arch).
    """

    def __init__(self):
        super().__init__(
            model_name="v2_grouped_v2_arch_ablation",
            model_gin_config="smaller_multitaskagent_grouped_v2_arch.gin",
            train_gin_config="grouped_v2_large_isfilter.gin",
            default_checkpoint=32,
            action_space=get_action_space("DefaultActionSpace"),
            observation_space=get_observation_space("GroupedObservationSpace"),
            reward_function=get_reward_function("AggressiveShapedReward"),
            tokenizer=get_tokenizer("DefaultObservationSpace-v1"),
            battle_backend="metamon",
            gin_overrides={
                "MetamonGroupedTstepEncoderV2.tokenizer": get_tokenizer(
                    "DefaultObservationSpace-v1"
                ),
            },
        )


@pretrained_model()
class V2AGroupedV2DataAblation(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="v2_grouped_v2_arch_data_ablation",
            model_gin_config="smaller_multitaskagent_grouped_v2_arch.gin",
            train_gin_config="grouped_v2_large_isfilter.gin",
            default_checkpoint=90,
            action_space=get_action_space("DefaultActionSpace"),
            observation_space=get_observation_space("GroupedObservationSpace"),
            reward_function=get_reward_function("AggressiveShapedReward"),
            tokenizer=get_tokenizer("DefaultObservationSpace-v1"),
            battle_backend="metamon",
            gin_overrides={
                "MetamonGroupedTstepEncoderV2.tokenizer": get_tokenizer(
                    "DefaultObservationSpace-v1"
                ),
            },
        )


@pretrained_model()
class TaurosV0(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="tauros-v0",
            model_gin_config="grouped_v2_50m.gin",
            train_gin_config="grouped_v2_large_isfilter.gin",
            default_checkpoint=62,
            action_space=get_action_space("DefaultActionSpace"),
            observation_space=get_observation_space("GroupedObservationSpace"),
            reward_function=get_reward_function("AggressiveShapedReward"),
            tokenizer=get_tokenizer("DefaultObservationSpace-v1"),
            battle_backend="metamon",
            gin_overrides={
                "MetamonGroupedTstepEncoderV2.tokenizer": get_tokenizer(
                    "DefaultObservationSpace-v1"
                ),
            },
        )


@pretrained_model()
class KakunaEnsemble(PretrainedModel):
    """
    A prototype version of ensembling metamon policies.

    Notably became the first Metamon agent to reach #1 on the Showdown leaderboard.
    """

    # these gxe scores are incorrect
    MEMBER_SPECS = [
        EnsembleMemberSpec(
            model_name="Kakuna",
            checkpoint=34,
            gxe=0.75,
            proposer_bias=1.10,
            judge_bias=1.15,
            shortlist_k=3,
        ),
        EnsembleMemberSpec(
            model_name="Kakuna",
            checkpoint=28,
            gxe=0.78,
            proposer_bias=1.20,
            judge_bias=1.15,
            shortlist_k=3,
        ),
        EnsembleMemberSpec(
            model_name="Kakuna",
            checkpoint=30,
            gxe=0.72,
            proposer_bias=1.35,
            judge_bias=0.15,
            shortlist_k=2,
            proposal_roles=("move", "counter_anchor"),
        ),
        EnsembleMemberSpec(
            model_name="Alakazam",
            checkpoint=8,
            gxe=0.64,
            proposer_bias=1.45,
            judge_bias=0.05,
            shortlist_k=2,
            proposal_roles=("move", "counter_anchor"),
        ),
    ]
    MEMBER_PRESETS = _load_ensemble_member_presets()

    @classmethod
    def _parse_member_specs_raw(cls, raw: str) -> list[EnsembleMemberSpec]:
        specs: list[EnsembleMemberSpec] = []
        for chunk in raw.split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            parts = [part.strip() for part in chunk.split(",")]
            if len(parts) not in {1, 5, 6, 7}:
                raise ValueError(
                    "METAMON_ENSEMBLE_MEMBER_SPECS entries must look like "
                    "'Model@Checkpoint,gxe,proposer_bias,judge_bias,shortlist_k[,role1|role2][,action_temperature]'"
                )

            name_part = parts[0]
            if "@" in name_part:
                model_name, checkpoint_str = name_part.split("@", 1)
                checkpoint = int(checkpoint_str) if checkpoint_str else None
            else:
                model_name = name_part
                checkpoint = None

            if len(parts) == 1:
                specs.append(
                    EnsembleMemberSpec(model_name=model_name, checkpoint=checkpoint)
                )
                continue

            proposal_roles: tuple[str, ...] = ()
            action_temperature = 1.0
            if len(parts) >= 6:
                try:
                    action_temperature = float(parts[5])
                except ValueError:
                    proposal_roles = tuple(
                        role.strip() for role in parts[5].split("|") if role.strip()
                    )
            if len(parts) == 7:
                action_temperature = float(parts[6])

            specs.append(
                EnsembleMemberSpec(
                    model_name=model_name,
                    checkpoint=checkpoint,
                    gxe=float(parts[1]),
                    proposer_bias=float(parts[2]),
                    judge_bias=float(parts[3]),
                    shortlist_k=int(parts[4]),
                    proposal_roles=proposal_roles,
                    action_temperature=action_temperature,
                )
            )
        if not specs:
            raise ValueError("METAMON_ENSEMBLE_MEMBER_SPECS produced no members")
        return specs

    @classmethod
    def _member_specs_from_env(cls) -> list[EnsembleMemberSpec]:
        raw = os.environ.get("METAMON_ENSEMBLE_MEMBER_SPECS", "").strip()
        if raw:
            return cls._parse_member_specs_raw(raw)

        preset_name = os.environ.get("METAMON_ENSEMBLE_PRESET", "").strip()
        if preset_name:
            if preset_name not in cls.MEMBER_PRESETS:
                raise ValueError(
                    f"Unknown METAMON_ENSEMBLE_PRESET '{preset_name}' "
                    f"(available: {sorted(cls.MEMBER_PRESETS)})"
                )
            return cls.MEMBER_PRESETS[preset_name]

        return cls.MEMBER_SPECS

    def __init__(self):
        super().__init__(
            model_name="kakuna-ensemble",
            model_gin_config="superkazam.gin",
            train_gin_config="kakuna.gin",
            default_checkpoint=34,
            action_space=get_action_space("DefaultActionSpace"),
            observation_space=get_observation_space("OpponentMoveObservationSpace"),
            reward_function=get_reward_function("AggressiveShapedReward"),
            tokenizer=get_tokenizer("DefaultObservationSpace-v1"),
            battle_backend="metamon",
            gin_overrides={
                "MetamonPerceiverTstepEncoder.tokenizer": get_tokenizer(
                    "DefaultObservationSpace-v1"
                ),
            },
        )

    def initialize_agent(
        self,
        checkpoint: Optional[int] = None,
        log: bool = False,
        action_temperature: float = 1.0,
    ):
        member_specs = self._member_specs_from_env()
        return build_heuristic_ensemble_experiment(
            reference_model_name="Kakuna",
            member_specs=member_specs,
            expected_obs_space=self.observation_space,
            expected_action_space=self.action_space,
            log=log,
            action_temperature=action_temperature,
        )


@pretrained_model()
class TaurosEnsemble(KakunaEnsemble):
    """
    Followup to KakunaEnsemble that also reached #1 on the Showdown leaderboard.
    """

    MEMBER_SPECS = [
        EnsembleMemberSpec(
            model_name="TaurosV0",
            checkpoint=62,
            gxe=0.86,
            proposer_bias=1.05,
            judge_bias=1.65,
            shortlist_k=5,
            action_temperature=0.82,
        ),
        EnsembleMemberSpec(
            model_name="TaurosV0",
            checkpoint=62,
            gxe=0.86,
            proposer_bias=1.65,
            judge_bias=0.05,
            shortlist_k=4,
            proposal_roles=("move", "switch", "counter_anchor"),
            action_temperature=1.22,
        ),
        EnsembleMemberSpec(
            model_name="TaurosV0",
            checkpoint=66,
            gxe=0.83,
            proposer_bias=1.15,
            judge_bias=0.55,
            shortlist_k=3,
            proposal_roles=("move", "counter_anchor"),
            action_temperature=0.98,
        ),
        EnsembleMemberSpec(
            model_name="V2AGroupedV2DataAblation",
            checkpoint=90,
            gxe=0.79,
            proposer_bias=1.35,
            judge_bias=0.20,
            shortlist_k=3,
            proposal_roles=("move", "switch", "counter_anchor"),
            action_temperature=1.08,
        ),
        EnsembleMemberSpec(
            model_name="V2AGroupedV2ISFilter",
            checkpoint=88,
            gxe=0.77,
            proposer_bias=1.10,
            judge_bias=0.60,
            shortlist_k=3,
            proposal_roles=("move", "counter_anchor"),
            action_temperature=1.02,
        ),
    ]

    def __init__(self):
        PretrainedModel.__init__(
            self,
            model_name="tauros-ensemble",
            model_gin_config="grouped_v2_50m.gin",
            train_gin_config="grouped_v2_large_isfilter.gin",
            default_checkpoint=62,
            action_space=get_action_space("DefaultActionSpace"),
            observation_space=get_observation_space("GroupedObservationSpace"),
            reward_function=get_reward_function("AggressiveShapedReward"),
            tokenizer=get_tokenizer("DefaultObservationSpace-v1"),
            battle_backend="metamon",
            gin_overrides={
                "MetamonGroupedTstepEncoderV2.tokenizer": get_tokenizer(
                    "DefaultObservationSpace-v1"
                ),
            },
        )

    def initialize_agent(
        self,
        checkpoint: Optional[int] = None,
        log: bool = False,
        action_temperature: float = 1.0,
    ):
        member_specs = self._member_specs_from_env()
        return build_heuristic_ensemble_experiment(
            reference_model_name="TaurosV0",
            member_specs=member_specs,
            expected_obs_space=self.observation_space,
            expected_action_space=self.action_space,
            log=log,
            action_temperature=action_temperature,
        )


import metamon.rl.experimental.ensemble.register  # noqa: F401 — nickname ensemble agents
