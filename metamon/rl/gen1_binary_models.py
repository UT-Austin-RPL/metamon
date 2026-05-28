"""
Local Gen 1 specialist model wrappers.

This module keeps the multi-generation public Kakuna wrapper separate from a
Gen 1-only training setup, while also registering a small set of local Gen 1
checkpoints for self-play evaluation and iteration.
"""

from __future__ import annotations

import os

from metamon.interface import get_action_space, get_observation_space, get_reward_function
from metamon.rl.pretrained import (
    LocalFinetunedModel,
    LocalPretrainedModel,
    pretrained_model,
)
from metamon.tokenizer import get_tokenizer


GEN1_CHECKPOINT_ROOT = os.environ.get(
    "METAMON_LOCAL_CHECKPOINT_DIR", os.path.expanduser("~/metamon/models")
)
GEN1OU_SPECIALIST_CHECKPOINT_ROOT = os.environ.get(
    "METAMON_GEN1OU_CHECKPOINT_DIR",
    os.path.expanduser("~/metamon/models/gen1ou-specialist"),
)
LAPRAS_SPECIALIST_CHECKPOINT_ROOT = os.environ.get(
    "METAMON_LAPRAS_CHECKPOINT_DIR",
    os.path.expanduser("~/metamon/models/lapras-specialist"),
)

GEN1_OBSERVATION_SPACE = get_observation_space("DefaultObservationSpace")
GEN1OU_SPECIALIST_OBSERVATION_SPACE = get_observation_space(
    "Gen1OpponentMoveObservationSpace"
)
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


@pretrained_model("Bulba")
class Bulba(LocalPretrainedModel):
    """
    Gen1OU-only Kakuna-style model wrapper.

    This points at a separate local checkpoint root and uses the Gen 1
    specialist observation/action spaces, so it does not affect the public
    multi-generation Kakuna wrapper or earlier local Gen 1 experiments.
    """

    def __init__(self):
        super().__init__(
            amago_ckpt_dir=GEN1OU_SPECIALIST_CHECKPOINT_ROOT,
            model_name="bulba",
            model_gin_config="superkazam.gin",
            train_gin_config="kakuna_gen1.gin",
            default_checkpoint=8,
            observation_space=GEN1OU_SPECIALIST_OBSERVATION_SPACE,
            action_space=GEN1_ACTION_SPACE,
            tokenizer=GEN1_TOKENIZER,
            reward_function=get_reward_function("AggressiveShapedReward"),
            battle_backend="metamon",
            gin_overrides=_gen1_gin_overrides(),
        )


@pretrained_model("Articuno")
class Articuno(LocalPretrainedModel):
    """
    Gen1OU local comparison model trained on the Tauros-only self-play mix.
    """

    def __init__(self):
        super().__init__(
            amago_ckpt_dir=GEN1OU_SPECIALIST_CHECKPOINT_ROOT,
            model_name="articuno",
            model_gin_config="superkazam.gin",
            train_gin_config="kakuna_gen1_isfilter.gin",
            default_checkpoint=40,
            observation_space=GEN1OU_SPECIALIST_OBSERVATION_SPACE,
            action_space=GEN1_ACTION_SPACE,
            tokenizer=GEN1_TOKENIZER,
            reward_function=get_reward_function("AggressiveShapedReward"),
            battle_backend="metamon",
            dataset_config="gen1ou_squirtle_tauros_only.yaml",
            gin_overrides=_gen1_gin_overrides(),
        )


@pretrained_model("lapras_bc_last2_v1")
class LaprasBCLast2V1(LocalFinetunedModel):
    """
    Lapras-team Articuno finetune trained on Lapras POV trajectories.

    Registered under the run name so eval commands can select checkpoints with
    ``--checkpoints``. Defaults to epoch 20.
    """

    def __init__(self):
        super().__init__(
            base_model=Articuno,
            amago_ckpt_dir=LAPRAS_SPECIALIST_CHECKPOINT_ROOT,
            model_name="lapras_bc_last2_v1",
            default_checkpoint=20,
            train_gin_config="lapras_bc.gin",
            dataset_config="lapras_only.yaml",
        )


@pretrained_model("lapras_bc_kl_anchor_l9_actor")
class LaprasBCKLAnchorL9Actor(LocalFinetunedModel):
    """
    KL-anchored Lapras-team Articuno finetune with layer 9 + actor trainable.
    """

    def __init__(self):
        super().__init__(
            base_model=Articuno,
            amago_ckpt_dir=LAPRAS_SPECIALIST_CHECKPOINT_ROOT,
            model_name="lapras_bc_kl_anchor_l9_actor",
            default_checkpoint=7,
            train_gin_config="lapras_bc_kl_anchor.gin",
            dataset_config="lapras_only.yaml",
        )


@pretrained_model("lapras_bc_kl_anchor_actor")
class LaprasBCKLAnchorActor(LocalFinetunedModel):
    """
    KL-anchored Lapras-team Articuno finetune with only actor trainable.
    """

    def __init__(self):
        super().__init__(
            base_model=Articuno,
            amago_ckpt_dir=LAPRAS_SPECIALIST_CHECKPOINT_ROOT,
            model_name="lapras_bc_kl_anchor_actor",
            default_checkpoint=7,
            train_gin_config="lapras_bc_kl_anchor.gin",
            dataset_config="lapras_only.yaml",
        )


@pretrained_model("lapras_actorv1")
class LaprasActorV1(LocalFinetunedModel):
    """
    Stable eval alias for epoch 5 of the actor-only KL-anchored Lapras finetune.
    """

    def __init__(self):
        super().__init__(
            base_model=Articuno,
            amago_ckpt_dir=LAPRAS_SPECIALIST_CHECKPOINT_ROOT,
            model_name="lapras_bc_kl_anchor_actor",
            default_checkpoint=5,
            train_gin_config="lapras_bc_kl_anchor.gin",
            dataset_config="lapras_only.yaml",
        )


@pretrained_model("Persian")
class Persian(LocalPretrainedModel):
    """
    Gen1OU local comparison model for the Persian slot-encoder run.

    This is the registered name used by `python -m metamon.rl.evaluate --agent Persian`.
    It points at the local training run under `gen1ou-specialist/persian_v1_slot_compare`.
    """

    def __init__(self):
        super().__init__(
            amago_ckpt_dir=GEN1OU_SPECIALIST_CHECKPOINT_ROOT,
            model_name="persian_v1_slot_compare",
            model_gin_config="gen1_pokemon_slot_persian_compare.gin",
            train_gin_config="grouped_v2_large_isfilter.gin",
            default_checkpoint=4,
            observation_space=GEN1OU_SPECIALIST_OBSERVATION_SPACE,
            action_space=GEN1_ACTION_SPACE,
            tokenizer=GEN1_TOKENIZER,
            reward_function=get_reward_function("AggressiveShapedReward"),
            battle_backend="metamon",
            gin_overrides=_gen1_gin_overrides(),
        )


@pretrained_model("Snorlax")
class Snorlax(LocalPretrainedModel):
    """
    Gen1OU local comparison model for the Snorlax grouped-encoder run.
    """

    def __init__(self):
        super().__init__(
            amago_ckpt_dir=GEN1OU_SPECIALIST_CHECKPOINT_ROOT,
            model_name="snorlax_v0",
            model_gin_config="grouped_v2_50m.gin",
            train_gin_config="grouped_v2_large_isfilter.gin",
            default_checkpoint=10,
            observation_space=get_observation_space("GroupedObservationSpace"),
            action_space=get_action_space("DefaultActionSpace"),
            tokenizer=GEN1_TOKENIZER,
            reward_function=get_reward_function("AggressiveShapedReward"),
            battle_backend="metamon",
            gin_overrides={
                "MetamonGroupedTstepEncoderV2.tokenizer": GEN1_TOKENIZER,
            },
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
