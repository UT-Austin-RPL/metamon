import os
from pathlib import Path
import warnings
from typing import Optional, Type

warnings.filterwarnings("ignore")


def red_warning(msg: str):
    print(f"\033[91m{msg}\033[0m")


import huggingface_hub
import torch
import amago

import metamon
from metamon.rl.metamon_to_amago import (
    make_placeholder_experiment,
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


if metamon.METAMON_CACHE_DIR is None:
    raise ValueError("Set METAMON_CACHE_DIR environment variable")
# downloads checkpoints to the metamon cache dir where we're putting all the other data
MODEL_DOWNLOAD_DIR = os.path.join(metamon.METAMON_CACHE_DIR, "pretrained_models")

# registry for pretrained models
ALL_PRETRAINED_MODELS = {}


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
    ):
        self.model_name = model_name
        self.model_gin_config = model_gin_config
        self.train_gin_config = train_gin_config
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

        By default, adds ability to fallback to vanilla attention if flash attention is not available,
        sets the tokenizer, and enbables faster initialization.
        """
        has_gpu = torch.cuda.is_available()
        try:
            import flash_attn

            has_flash_attn = True
        except ImportError:
            has_flash_attn = False
        if has_flash_attn and has_gpu:
            attn_type = amago.nets.transformer.FlashAttention
        else:
            attn_type = amago.nets.transformer.VanillaAttention
            red_warning("Warning: Using unofficial VanillaAttention implementation")
        config = {
            "amago.nets.traj_encoders.TformerTrajEncoder.attention_type": attn_type,
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
        self, checkpoint: Optional[int] = None, log: bool = False
    ) -> amago.Experiment:
        # use the base config and the gin file to configure the model
        amago.cli_utils.use_config(
            self.base_config,
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
            # replace the weights with the pretrained checkpoint
            experiment.load_checkpoint_from_path(ckpt_path, is_accelerate_state=False)
        return experiment


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
    """

    def __init__(
        self,
        base_model: Type[PretrainedModel],
        amago_ckpt_dir: str,
        model_name: str,
        default_checkpoint: int,
        train_gin_config: Optional[str] = None,
        reward_function: Optional[RewardFunction] = None,
    ):
        base_model = base_model()
        train_gin_config = train_gin_config or base_model.train_gin_config
        reward_function = reward_function or base_model.reward_function
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
        )


#####################
## Paper Policies ###
#####################


@pretrained_model()
class SmallIL(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="small-il",
            model_gin_config="small_agent.gin",
            train_gin_config="il.gin",
            default_checkpoint=40,
            action_space=get_action_space("MinimalActionSpace"),
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
        )


@pretrained_model()
class SleepLoop4V1_Epoch0(LocalPretrainedModel):
    def __init__(self):
        super().__init__(
            amago_ckpt_dir="/home/eddie/metamon/models/gen1_aggressive_sleep_loop4_v1",
            model_name="sleep-loop4-v1",
            model_gin_config="synthetic_multitaskagent.gin",
            train_gin_config="sleep_selfplay_v1.gin",
            tokenizer=get_tokenizer("allreplays-v3"),
            observation_space=get_observation_space("DefaultObservationSpace"),
            action_space=get_action_space("MinimalActionSpace"),
            reward_function=get_reward_function("AggressiveShapedRewardSleep"),
            default_checkpoint=0,
        )


@pretrained_model()
class SleepLoop4V1_Epoch2(LocalPretrainedModel):
    def __init__(self):
        super().__init__(
            amago_ckpt_dir="/home/eddie/metamon/models/gen1_aggressive_sleep_loop4_v1",
            model_name="sleep-loop4-v1",
            model_gin_config="synthetic_multitaskagent.gin",
            train_gin_config="sleep_selfplay_v1.gin",
            tokenizer=get_tokenizer("allreplays-v3"),
            observation_space=get_observation_space("DefaultObservationSpace"),
            action_space=get_action_space("MinimalActionSpace"),
            reward_function=get_reward_function("AggressiveShapedRewardSleep"),
            default_checkpoint=2,
        )


@pretrained_model()
class SleepLoop4V1_Epoch4(LocalPretrainedModel):
    def __init__(self):
        super().__init__(
            amago_ckpt_dir="/home/eddie/metamon/models/gen1_aggressive_sleep_loop4_v1",
            model_name="sleep-loop4-v1",
            model_gin_config="synthetic_multitaskagent.gin",
            train_gin_config="sleep_selfplay_v1.gin",
            tokenizer=get_tokenizer("allreplays-v3"),
            observation_space=get_observation_space("DefaultObservationSpace"),
            action_space=get_action_space("MinimalActionSpace"),
            reward_function=get_reward_function("AggressiveShapedRewardSleep"),
            default_checkpoint=4,
        )


@pretrained_model()
class SleepLoop4V3_Epoch0(LocalPretrainedModel):
    def __init__(self):
        super().__init__(
            amago_ckpt_dir="/home/eddie/metamon/models/gen1_aggressive_sleep_loop4_v3",
            model_name="sleep-loop4-v3",
            model_gin_config="synthetic_multitaskagent.gin",
            train_gin_config="sleep_selfplay_v2_aggressive.gin",
            tokenizer=get_tokenizer("allreplays-v3"),
            observation_space=get_observation_space("DefaultObservationSpace"),
            action_space=get_action_space("MinimalActionSpace"),
            reward_function=get_reward_function("AggressiveShapedRewardSleep"),
            default_checkpoint=0,
        )


@pretrained_model()
class SleepLoop4V3_Epoch2(LocalPretrainedModel):
    def __init__(self):
        super().__init__(
            amago_ckpt_dir="/home/eddie/metamon/models/gen1_aggressive_sleep_loop4_v3",
            model_name="sleep-loop4-v3",
            model_gin_config="synthetic_multitaskagent.gin",
            train_gin_config="sleep_selfplay_v2_aggressive.gin",
            tokenizer=get_tokenizer("allreplays-v3"),
            observation_space=get_observation_space("DefaultObservationSpace"),
            action_space=get_action_space("MinimalActionSpace"),
            reward_function=get_reward_function("AggressiveShapedRewardSleep"),
            default_checkpoint=2,
        )


@pretrained_model()
class SleepLoop4V3_Epoch4(LocalPretrainedModel):
    def __init__(self):
        super().__init__(
            amago_ckpt_dir="/home/eddie/metamon/models/gen1_aggressive_sleep_loop4_v3",
            model_name="sleep-loop4-v3",
            model_gin_config="synthetic_multitaskagent.gin",
            train_gin_config="sleep_selfplay_v2_aggressive.gin",
            tokenizer=get_tokenizer("allreplays-v3"),
            observation_space=get_observation_space("DefaultObservationSpace"),
            action_space=get_action_space("MinimalActionSpace"),
            reward_function=get_reward_function("AggressiveShapedRewardSleep"),
            default_checkpoint=4,
        )


@pretrained_model()
class SleepLoop5V1_Epoch0(LocalPretrainedModel):
    def __init__(self):
        super().__init__(
            amago_ckpt_dir="/home/eddie/metamon/models/gen1_aggressive_sleep_loop5_v1",
            model_name="sleep-loop5-v1",
            model_gin_config="synthetic_multitaskagent.gin",
            train_gin_config="sleep_selfplay_v2_aggressive.gin",
            tokenizer=get_tokenizer("allreplays-v3"),
            observation_space=get_observation_space("DefaultObservationSpace"),
            action_space=get_action_space("MinimalActionSpace"),
            reward_function=get_reward_function("AggressiveShapedRewardSleep"),
            default_checkpoint=0,
        )


@pretrained_model()
class SleepLoop5V1_Epoch2(LocalPretrainedModel):
    def __init__(self):
        super().__init__(
            amago_ckpt_dir="/home/eddie/metamon/models/gen1_aggressive_sleep_loop5_v1",
            model_name="sleep-loop5-v1",
            model_gin_config="synthetic_multitaskagent.gin",
            train_gin_config="sleep_selfplay_v2_aggressive.gin",
            tokenizer=get_tokenizer("allreplays-v3"),
            observation_space=get_observation_space("DefaultObservationSpace"),
            action_space=get_action_space("MinimalActionSpace"),
            reward_function=get_reward_function("AggressiveShapedRewardSleep"),
            default_checkpoint=2,
        )


@pretrained_model()
class SleepLoop5V1_Epoch4(LocalPretrainedModel):
    def __init__(self):
        super().__init__(
            amago_ckpt_dir="/home/eddie/metamon/models/gen1_aggressive_sleep_loop5_v1",
            model_name="sleep-loop5-v1",
            model_gin_config="synthetic_multitaskagent.gin",
            train_gin_config="sleep_selfplay_v2_aggressive.gin",
            tokenizer=get_tokenizer("allreplays-v3"),
            observation_space=get_observation_space("DefaultObservationSpace"),
            action_space=get_action_space("MinimalActionSpace"),
            reward_function=get_reward_function("AggressiveShapedRewardSleep"),
            default_checkpoint=4,
        )


@pretrained_model()
class SleepLoop5Controller_Epoch0(LocalPretrainedModel):
    def __init__(self):
        super().__init__(
            amago_ckpt_dir="/home/eddie/metamon/models/gen1_control_sleep_loop5",
            model_name="sleep-loop5-control",
            model_gin_config="synthetic_multitaskagent.gin",
            train_gin_config="selfplay_controller_v1.gin",
            tokenizer=get_tokenizer("allreplays-v3"),
            observation_space=get_observation_space("DefaultObservationSpace"),
            action_space=get_action_space("MinimalActionSpace"),
            reward_function=get_reward_function("AggressiveShapedRewardSleep"),
            default_checkpoint=0,
        )


@pretrained_model()
class SleepLoop5Controller_Epoch2(LocalPretrainedModel):
    def __init__(self):
        super().__init__(
            amago_ckpt_dir="/home/eddie/metamon/models/gen1_control_sleep_loop5",
            model_name="sleep-loop5-control",
            model_gin_config="synthetic_multitaskagent.gin",
            train_gin_config="selfplay_controller_v1.gin",
            tokenizer=get_tokenizer("allreplays-v3"),
            observation_space=get_observation_space("DefaultObservationSpace"),
            action_space=get_action_space("MinimalActionSpace"),
            reward_function=get_reward_function("AggressiveShapedRewardSleep"),
            default_checkpoint=2,
        )


@pretrained_model()
class SleepLoop5Controller_Epoch4(LocalPretrainedModel):
    def __init__(self):
        super().__init__(
            amago_ckpt_dir="/home/eddie/metamon/models/gen1_control_sleep_loop5",
            model_name="sleep-loop5-control",
            model_gin_config="synthetic_multitaskagent.gin",
            train_gin_config="selfplay_controller_v1.gin",
            tokenizer=get_tokenizer("allreplays-v3"),
            observation_space=get_observation_space("DefaultObservationSpace"),
            action_space=get_action_space("MinimalActionSpace"),
            reward_function=get_reward_function("AggressiveShapedRewardSleep"),
            default_checkpoint=4,
        )


@pretrained_model()
class BinaryLoop4V1_Epoch0(LocalPretrainedModel):
    def __init__(self):
        super().__init__(
            amago_ckpt_dir="/home/eddie/metamon/models/gen1_binary_loop4_v1",
            model_name="binary-loop4-testv1",
            model_gin_config="synthetic_multitaskagent.gin",
            train_gin_config="vanilla_selfplay_damped_conservative.gin",
            tokenizer=get_tokenizer("allreplays-v3"),
            observation_space=get_observation_space("DefaultObservationSpace"),
            action_space=get_action_space("MinimalActionSpace"),
            reward_function=get_reward_function("BinaryReward"),
            default_checkpoint=0,
        )


@pretrained_model()
class BinaryLoop4V1_Epoch2(LocalPretrainedModel):
    def __init__(self):
        super().__init__(
            amago_ckpt_dir="/home/eddie/metamon/models/gen1_binary_loop4_v1",
            model_name="binary-loop4-testv1",
            model_gin_config="synthetic_multitaskagent.gin",
            train_gin_config="vanilla_selfplay_damped_conservative.gin",
            tokenizer=get_tokenizer("allreplays-v3"),
            observation_space=get_observation_space("DefaultObservationSpace"),
            action_space=get_action_space("MinimalActionSpace"),
            reward_function=get_reward_function("BinaryReward"),
            default_checkpoint=2,
        )


@pretrained_model()
class BinaryLoop4V1_Epoch4(LocalPretrainedModel):
    def __init__(self):
        super().__init__(
            amago_ckpt_dir="/home/eddie/metamon/models/gen1_binary_loop4_v1",
            model_name="binary-loop4-testv1",
            model_gin_config="synthetic_multitaskagent.gin",
            train_gin_config="vanilla_selfplay_damped_conservative.gin",
            tokenizer=get_tokenizer("allreplays-v3"),
            observation_space=get_observation_space("DefaultObservationSpace"),
            action_space=get_action_space("MinimalActionSpace"),
            reward_function=get_reward_function("BinaryReward"),
            default_checkpoint=4,
        )


###################################
## PokéAgent Challenge Policies ###
###################################


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
            observation_space=get_observation_space("TeamPreviewObservationSpace"),
            tokenizer=get_tokenizer("DefaultObservationSpace-v1"),
            # temporarily forced to flash attention until we can verify numerical stability
            # of a switch to a standard pytorch sliding window inference alternative
            gin_overrides={
                "amago.nets.traj_encoders.TformerTrajEncoder.attention_type": amago.nets.transformer.FlashAttention,
                "amago.nets.transformer.FlashAttention.window_size": (32, 0),
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
            observation_space=get_observation_space("TeamPreviewObservationSpace"),
            tokenizer=get_tokenizer("DefaultObservationSpace-v1"),
            gin_overrides={
                "amago.nets.traj_encoders.TformerTrajEncoder.attention_type": amago.nets.transformer.FlashAttention,
                "amago.nets.transformer.FlashAttention.window_size": (32, 0),
            },
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
            observation_space=get_observation_space("OpponentMoveObservationSpace"),
            reward_function=get_reward_function("AggressiveShapedReward"),
            tokenizer=get_tokenizer("DefaultObservationSpace-v1"),
        )

    @property
    def base_config(self):
        return {"MetamonPerceiverTstepEncoder.tokenizer": self.tokenizer}
