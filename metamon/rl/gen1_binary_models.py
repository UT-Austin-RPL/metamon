"""
Local Gen 1 specialist model wrappers.

This module keeps the multi-generation public Kakuna wrapper separate from a
Gen 1-only training setup, while also registering a small set of local Gen 1
checkpoints for self-play evaluation and iteration.
"""

from __future__ import annotations

import os

from metamon.interface import get_action_space, get_observation_space, get_reward_function
from metamon.rl.pretrained import LocalPretrainedModel, pretrained_model
from metamon.tokenizer import get_tokenizer


GEN1_CHECKPOINT_ROOT = os.environ.get(
    "METAMON_LOCAL_CHECKPOINT_DIR", os.path.expanduser("~/metamon/models")
)

GEN1_OBSERVATION_SPACE = get_observation_space("DefaultObservationSpace")
GEN1_ACTION_SPACE = get_action_space("MinimalActionSpace")
GEN1_TOKENIZER = get_tokenizer("DefaultObservationSpace-v1")


def _gen1_gin_overrides() -> dict:
    return {
        "MetamonPerceiverTstepEncoder.tokenizer": GEN1_TOKENIZER,
    }


@pretrained_model("KakunaGen1")
class KakunaGen1(LocalPretrainedModel):
    """
    Gen 1-only Kakuna-style model wrapper.

    Use this when training or evaluating a Kakuna-sized policy on Gen 1-only
    mechanics, observation space, and actions.
    """

    def __init__(self):
        super().__init__(
            amago_ckpt_dir=GEN1_CHECKPOINT_ROOT,
            model_name="kakuna_gen1",
            model_gin_config="superkazam.gin",
            train_gin_config="kakuna_gen1.gin",
            default_checkpoint=34,
            observation_space=GEN1_OBSERVATION_SPACE,
            action_space=GEN1_ACTION_SPACE,
            tokenizer=GEN1_TOKENIZER,
            reward_function=get_reward_function("AggressiveShapedReward"),
            battle_backend="metamon",
            gin_overrides=_gen1_gin_overrides(),
        )


class _Gen1BinaryRewardBase(LocalPretrainedModel):
    """
    Shared Gen 1 binary-reward checkpoint wrapper.
    """

    model_name = "gen1_binary_reward_v0"

    def __init__(self, default_checkpoint: int):
        super().__init__(
            amago_ckpt_dir=GEN1_CHECKPOINT_ROOT,
            model_name=self.model_name,
            model_gin_config="superkazam.gin",
            train_gin_config="binary_rl.gin",
            default_checkpoint=default_checkpoint,
            observation_space=GEN1_OBSERVATION_SPACE,
            action_space=GEN1_ACTION_SPACE,
            tokenizer=GEN1_TOKENIZER,
            reward_function=get_reward_function("BinaryReward"),
            battle_backend="metamon",
            gin_overrides=_gen1_gin_overrides(),
        )


@pretrained_model("Gen1BinaryV0_Epoch0")
class Gen1BinaryV0_Epoch0(_Gen1BinaryRewardBase):
    def __init__(self):
        super().__init__(0)


@pretrained_model("Gen1BinaryV0_Epoch2")
class Gen1BinaryV0_Epoch2(_Gen1BinaryRewardBase):
    def __init__(self):
        super().__init__(2)


@pretrained_model("Gen1BinaryV0_Epoch4")
class Gen1BinaryV0_Epoch4(_Gen1BinaryRewardBase):
    def __init__(self):
        super().__init__(4)


@pretrained_model("Gen1BinaryV0_Epoch6")
class Gen1BinaryV0_Epoch6(_Gen1BinaryRewardBase):
    def __init__(self):
        super().__init__(6)


@pretrained_model("Gen1BinaryV0_Epoch8")
class Gen1BinaryV0_Epoch8(_Gen1BinaryRewardBase):
    def __init__(self):
        super().__init__(8)


@pretrained_model("Gen1BinaryV0_Epoch10")
class Gen1BinaryV0_Epoch10(_Gen1BinaryRewardBase):
    def __init__(self):
        super().__init__(10)
