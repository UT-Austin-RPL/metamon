#!/usr/bin/env python
"""Standalone param counter: build a metamon AMAGO policy from a model gin
config + the registered base model's obs/action/tokenizer spaces, and print the
trainable param count. No envs, no datasets, no GPU needed.

Usage:
    python scripts/count_params.py <model_gin_config> [<base_model_name>]

Examples:
    python scripts/count_params.py smaller_multitaskagent_grouped_v2_arch.gin V2AGroupedV2DataAblation
    python scripts/count_params.py grouped_v2_50m.gin V2AGroupedV2DataAblation
"""

from __future__ import annotations

import sys
import os

os.environ.setdefault("METAMON_CACHE_DIR", "/home/eddie/metamon_cache")

# Stub out an "rl2_space" — the agent reads its shape; metamon uses a fixed
# small rl2 tstep dim. We mirror what the env produces.
import gymnasium as gym
import numpy as np
import torch

import amago
from amago.cli_utils import use_config
import gin

from metamon.rl.pretrained import get_pretrained_model

# Import the experiment class so gin can resolve `MetamonAMAGOExperiment.*`
# bindings in the model gin files (agent_type / tstep_encoder_type / ...).
import metamon.rl.metamon_to_amago  # noqa: F401  (registers gin configurables)
from metamon.rl.metamon_to_amago import (
    MetamonAMAGOExperiment,
    MetamonGroupedTstepEncoderV2,
)
from amago.agent import MultiTaskAgent
from amago.nets.traj_encoders import TformerTrajEncoder


def count(model_gin_config: str, base_model_name: str) -> None:
    pretrained = get_pretrained_model(base_model_name)
    # Resolve the requested model gin against the metamon model config dir; use
    # the base model's train gin + base_config (tokenizer + gin_overrides)
    # exactly as PretrainedModel.initialize_agent does, so a custom arch gin
    # still gets the right tokenizer / obs-space wiring.
    import metamon.rl as _mrl

    model_gin_path = os.path.join(_mrl.MODEL_CONFIG_DIR, model_gin_config)
    train_gin_path = pretrained.train_gin_config_path

    gin.clear_config()
    use_config(pretrained.base_config, [model_gin_path, train_gin_path], finalize=True)

    # Build dummy obs/rl2/action spaces matching the registered model. The
    # tstep encoder reads obs_space["numbers_active_pokemon"].shape etc.
    obs_space = pretrained.observation_space.gym_space
    # rl2_space: a small Box the agent treats as the recurrent task-embedding
    # input. metamon uses a 1-D box; shape doesn't affect param count materially
    # (only a small input projection). Use a minimal stand-in.
    rl2_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    action_space = pretrained.action_space.gym_space

    agent_type = (
        amago.agent.MultiTaskAgent
    )  # bound by gin (MetamonAMAGOExperiment.agent_type)
    # The gin file binds MetamonAMAGOExperiment.{agent,tstep,traj}_type to
    # @-references. The GroupedV2 arch family is fixed, so resolve to the
    # concrete classes directly (robust across gin versions).
    agent_type = MultiTaskAgent
    tstep_encoder_type = MetamonGroupedTstepEncoderV2
    traj_encoder_type = TformerTrajEncoder
    max_seq_len = gin.query_parameter("MetamonAMAGOExperiment.max_seq_len")

    policy = agent_type(
        obs_space=obs_space,
        rl2_space=rl2_space,
        action_space=action_space,
        max_seq_len=max_seq_len,
        tstep_encoder_type=tstep_encoder_type,
        traj_encoder_type=traj_encoder_type,
    )
    n = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"Model gin: {model_gin_config}")
    print(f"Base model: {base_model_name}")
    print(f"Trainable params: {n:,}  ({n/1e6:.2f}M)")


if __name__ == "__main__":
    model_gin = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "smaller_multitaskagent_grouped_v2_arch.gin"
    )
    base = sys.argv[2] if len(sys.argv) > 2 else "V2AGroupedV2DataAblation"
    count(model_gin, base)
