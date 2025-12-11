"""
Custom model wrappers for Gen1 specialist finetuning checkpoints.

These models can be used with the evaluation scripts to test different training epochs
against each other and compute ELO ratings.

Includes:
- Gen1BinaryV0 checkpoints: Failed BinaryReward experiment (for reference)
- P0_SYN_V2_GEN1: Phase 0 baseline for Nash equilibrium training
"""

from metamon.rl.pretrained import LocalFinetunedModel, pretrained_model, SyntheticRLV2
from metamon.interface import get_reward_function

# Checkpoint directory - adjust this path if your checkpoints are elsewhere
CHECKPOINT_DIR = "/home/eddie/metamon/models/metamon_checkpoints"


#####################
## Phase 0: Baseline
#####################


@pretrained_model("P0_SYN_V2_GEN1")
class P0_SYN_V2_GEN1(LocalFinetunedModel):
    """
    Phase 0 Gen1 Baseline - Nash equilibrium training foundation.

    This is a Gen1 OU specialist created by fine-tuning SyntheticRLV2 on Gen1 replays
    using DefaultShapedReward. It serves as the initial strong policy π₁ in the Nash
    population before PSRO training begins.

    Training config:
    - Base: SyntheticRLV2 (200M params, multi-gen general model)
    - Format: Gen1 OU only (175k battles)
    - Reward: DefaultShapedReward (learned from BinaryReward failure)
    - Epochs: 3 (~75k steps)
    - Purpose: Baseline for interaction matrix and PSRO
    """

    def __init__(self):
        super().__init__(
            base_model=SyntheticRLV2,
            amago_ckpt_dir=CHECKPOINT_DIR,
            model_name="P0_SYN_V2_GEN1",
            default_checkpoint=2,  # Epoch 2 is typically best for fine-tuning
            reward_function=get_reward_function("DefaultShapedReward"),
        )


###################################
## Dynamic Damping Self-Play Models
###################################


@pretrained_model("DampedConservative100k_Epoch2")
class DampedConservative100k_Epoch2(LocalFinetunedModel):
    """
    Conservative Damping Self-Play - Epoch 2

    Self-play training with conservative damping parameters on Gen1 OU data.
    Uses adaptive regularization to prevent policy collapse during self-play.

    Training config:
    - Base: SyntheticRLV2
    - Format: Gen1 OU
    - Training: vanilla_selfplay_damped_conservative.gin
    - Reward: DefaultShapedReward
    - Damping: Conservative power-law KL regularization
    - WandB: damped-conservative-100k (run 0jrll78y)
    """

    def __init__(self):
        super().__init__(
            base_model=SyntheticRLV2,
            amago_ckpt_dir="/home/eddie/metamon/models/gen1_selfplay_damped_con_ckpt",
            model_name="damped-conservative-100k",
            default_checkpoint=2,
            train_gin_config="vanilla_selfplay_damped_conservative.gin",
            reward_function=get_reward_function("DefaultShapedReward"),
        )


@pretrained_model("DampedConservative100k_Epoch3")
class DampedConservative100k_Epoch3(LocalFinetunedModel):
    """
    Conservative Damping Self-Play - Epoch 3 (Latest)

    Self-play training with conservative damping parameters on Gen1 OU data.
    Uses adaptive regularization to prevent policy collapse during self-play.

    Training config:
    - Base: SyntheticRLV2
    - Format: Gen1 OU
    - Training: vanilla_selfplay_damped_conservative.gin
    - Reward: DefaultShapedReward
    - Damping: Conservative power-law KL regularization
    - WandB: damped-conservative-100k (run 0jrll78y)
    """

    def __init__(self):
        super().__init__(
            base_model=SyntheticRLV2,
            amago_ckpt_dir="/home/eddie/metamon/models/gen1_selfplay_damped_con_ckpt",
            model_name="damped-conservative-100k",
            default_checkpoint=3,
            train_gin_config="vanilla_selfplay_damped_conservative.gin",
            reward_function=get_reward_function("DefaultShapedReward"),
        )


@pretrained_model("DampedConservativeBinaryV2_Epoch2")
class DampedConservativeBinaryV2_Epoch2(LocalFinetunedModel):
    """
    Conservative Damping Binary Reward V2 - Epoch 2

    Finetuned from DampedConservative100k using BinaryReward (sparse +/-100 win/loss).
    Tests whether the conservative damping policy can adapt to binary rewards while
    maintaining stability.

    Training config:
    - Base: DampedConservative100k (finetuned from SyntheticRLV2)
    - Format: Gen1 OU
    - Training: vanilla_selfplay_damped_conservative.gin
    - Reward: BinaryReward (sparse)
    - Damping: Conservative power-law KL regularization
    """

    def __init__(self):
        super().__init__(
            base_model=SyntheticRLV2,
            amago_ckpt_dir="/home/eddie/metamon/models/gen1_selfplay_damped_con_binary_v2_ckpt",
            model_name="damped-conservative-100k-binary-v2",
            default_checkpoint=2,
            train_gin_config="vanilla_selfplay_damped_conservative.gin",
            reward_function=get_reward_function("BinaryReward"),
        )


@pretrained_model("DampedBinarySuperV1_Epoch4")
class DampedBinarySuperV1_Epoch4(LocalFinetunedModel):
    """
    Damped Binary Super V1 - Epoch 4 (Latest)

    Finetuned from DampedConservativeBinaryV2_Epoch2 on super_dataset.
    Uses conservative damping with BinaryReward on an expanded dataset.

    Training config:
    - Base: DampedConservativeBinaryV2_Epoch2
    - Format: Gen1 OU
    - Training: vanilla_selfplay_damped_conservative.gin
    - Reward: BinaryReward (sparse)
    - Dataset: super_dataset (custom replay directory)
    """

    def __init__(self):
        super().__init__(
            base_model=SyntheticRLV2,
            amago_ckpt_dir="/home/eddie/metamon/models/gen1_binary_loop2_ckpt",
            model_name="damped-binary-super2v1",
            default_checkpoint=4,
            train_gin_config="vanilla_selfplay_damped_conservative.gin",
            reward_function=get_reward_function("BinaryReward"),
        )


###################################
## Aggressive Sleep Strategy Models
###################################


@pretrained_model("AggressiveSleepV5_Epoch0")
class AggressiveSleepV5_Epoch0(LocalFinetunedModel):
    """
    Aggressive Sleep Strategy V5 - Epoch 0

    Ultra-safe filtered BC finetuning from DampedBinarySuperV1_Epoch4 on super_dataset_loop3
    with AggressiveShapedRewardSleep (+200/0 win/loss, +1 sleep bonus).

    This is the only aggressive sleep training that successfully avoided collapse.

    Training config:
    - Base: DampedBinarySuperV1_Epoch4
    - Format: Gen1 OU
    - Training: selfplay_damped_aggressive_v4_safe.gin (PURE filtered BC, NO DPG)
    - Reward: AggressiveShapedRewardSleep (+200/0 win/loss, +1 sleep)
    - Observation: ExpandedObservationSpace
    - Dataset: super_dataset_loop3 (25,072 battles)
    - Data mix: 100% filtered BC, 0% DPG (critic-warmup mode)
    - KL target: 0.008 (very strict)
    - Learning rate: 3e-6 to 6e-6 (tiny)
    """

    def __init__(self):
        super().__init__(
            base_model=SyntheticRLV2,
            amago_ckpt_dir="/home/eddie/metamon/models/gen1_aggressive_sleep_loop3_v5",
            model_name="aggressive-sleep-loop3-v5",
            default_checkpoint=0,
            train_gin_config="selfplay_damped_aggressive_v4_safe.gin",
            reward_function=get_reward_function("AggressiveShapedRewardSleep"),
        )


@pretrained_model("AggressiveSleepV5_Epoch1")
class AggressiveSleepV5_Epoch1(LocalFinetunedModel):
    """Aggressive Sleep Strategy V5 - Epoch 1"""

    def __init__(self):
        super().__init__(
            base_model=SyntheticRLV2,
            amago_ckpt_dir="/home/eddie/metamon/models/gen1_aggressive_sleep_loop3_v5",
            model_name="aggressive-sleep-loop3-v5",
            default_checkpoint=1,
            train_gin_config="selfplay_damped_aggressive_v4_safe.gin",
            reward_function=get_reward_function("AggressiveShapedRewardSleep"),
        )


@pretrained_model("AggressiveSleepV5_Epoch2")
class AggressiveSleepV5_Epoch2(LocalFinetunedModel):
    """Aggressive Sleep Strategy V5 - Epoch 2"""

    def __init__(self):
        super().__init__(
            base_model=SyntheticRLV2,
            amago_ckpt_dir="/home/eddie/metamon/models/gen1_aggressive_sleep_loop3_v5",
            model_name="aggressive-sleep-loop3-v5",
            default_checkpoint=2,
            train_gin_config="selfplay_damped_aggressive_v4_safe.gin",
            reward_function=get_reward_function("AggressiveShapedRewardSleep"),
        )


@pretrained_model("AggressiveSleepV5_Epoch3")
class AggressiveSleepV5_Epoch3(LocalFinetunedModel):
    """Aggressive Sleep Strategy V5 - Epoch 3"""

    def __init__(self):
        super().__init__(
            base_model=SyntheticRLV2,
            amago_ckpt_dir="/home/eddie/metamon/models/gen1_aggressive_sleep_loop3_v5",
            model_name="aggressive-sleep-loop3-v5",
            default_checkpoint=3,
            train_gin_config="selfplay_damped_aggressive_v4_safe.gin",
            reward_function=get_reward_function("AggressiveShapedRewardSleep"),
        )


@pretrained_model("AggressiveSleepV5_Epoch4")
class AggressiveSleepV5_Epoch4(LocalFinetunedModel):
    """Aggressive Sleep Strategy V5 - Epoch 4 (Final)"""

    def __init__(self):
        super().__init__(
            base_model=SyntheticRLV2,
            amago_ckpt_dir="/home/eddie/metamon/models/gen1_aggressive_sleep_loop3_v5",
            model_name="aggressive-sleep-loop3-v5",
            default_checkpoint=4,
            train_gin_config="selfplay_damped_aggressive_v4_safe.gin",
            reward_function=get_reward_function("AggressiveShapedRewardSleep"),
        )


##############################
## Gen1 BinaryReward Experiment
##############################


@pretrained_model("Gen1BinaryV0_Epoch0")
class Gen1BinaryV0_Epoch0(LocalFinetunedModel):
    """Gen1 BinaryReward Specialist - Epoch 0 (75k steps)"""

    def __init__(self):
        super().__init__(
            base_model=SyntheticRLV2,
            amago_ckpt_dir=CHECKPOINT_DIR,
            model_name="Gen1BinaryRewardV0",
            default_checkpoint=0,
            reward_function=get_reward_function("BinaryReward"),
        )


@pretrained_model("Gen1BinaryV0_Epoch2")
class Gen1BinaryV0_Epoch2(LocalFinetunedModel):
    """Gen1 BinaryReward Specialist - Epoch 2 (125k steps)"""

    def __init__(self):
        super().__init__(
            base_model=SyntheticRLV2,
            amago_ckpt_dir=CHECKPOINT_DIR,
            model_name="Gen1BinaryRewardV0",
            default_checkpoint=2,
            reward_function=get_reward_function("BinaryReward"),
        )


@pretrained_model("Gen1BinaryV0_Epoch4")
class Gen1BinaryV0_Epoch4(LocalFinetunedModel):
    """Gen1 BinaryReward Specialist - Epoch 4 (175k steps)"""

    def __init__(self):
        super().__init__(
            base_model=SyntheticRLV2,
            amago_ckpt_dir=CHECKPOINT_DIR,
            model_name="Gen1BinaryRewardV0",
            default_checkpoint=4,
            reward_function=get_reward_function("BinaryReward"),
        )


@pretrained_model("Gen1BinaryV0_Epoch6")
class Gen1BinaryV0_Epoch6(LocalFinetunedModel):
    """Gen1 BinaryReward Specialist - Epoch 6 (225k steps)"""

    def __init__(self):
        super().__init__(
            base_model=SyntheticRLV2,
            amago_ckpt_dir=CHECKPOINT_DIR,
            model_name="Gen1BinaryRewardV0",
            default_checkpoint=6,
            reward_function=get_reward_function("BinaryReward"),
        )


@pretrained_model("Gen1BinaryV0_Epoch8")
class Gen1BinaryV0_Epoch8(LocalFinetunedModel):
    """Gen1 BinaryReward Specialist - Epoch 8 (275k steps)"""

    def __init__(self):
        super().__init__(
            base_model=SyntheticRLV2,
            amago_ckpt_dir=CHECKPOINT_DIR,
            model_name="Gen1BinaryRewardV0",
            default_checkpoint=8,
            reward_function=get_reward_function("BinaryReward"),
        )


@pretrained_model("Gen1BinaryV0_Epoch10")
class Gen1BinaryV0_Epoch10(LocalFinetunedModel):
    """Gen1 BinaryReward Specialist - Epoch 10 (325k steps)"""

    def __init__(self):
        super().__init__(
            base_model=SyntheticRLV2,
            amago_ckpt_dir=CHECKPOINT_DIR,
            model_name="Gen1BinaryRewardV0",
            default_checkpoint=10,
            reward_function=get_reward_function("BinaryReward"),
        )
