"""Shared fixtures for the GPU/checkpoint-gated test-time search tests.

The frozen ``MiniOnlinePsroV1_4`` checkpoint (epoch 740) and a CUDA GPU are
required for ``test_policy_state_fork.py`` and ``test_search_equivalence.py``
(Phase 0B / 0F MUST-RUN, skill §8/§21). These tests auto-skip when the
checkpoint or CUDA is unavailable so the rest of the suite stays green in a
CPU/CI environment; in a GPU environment with the checkpoint present they run.

The ``frozen_env_bundle`` fixture mirrors
``eval_search.run_search_eval``'s setup exactly (same model load, same
``BattleAgainstMetamon`` env, same ``AmagoLadderPolicyDriver``) so the gated
tests exercise the real production path. Set ``METAMON_CACHE_DIR`` before
running (the repo requires it).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch

# ---------------------------------------------------------------------------
# checkpoint / GPU availability
# ---------------------------------------------------------------------------

AGENT_NAME = "MiniOnlinePsroV1_4"
CHECKPOINT_EPOCH = 740
CKPT_PATH = Path(
    os.path.expanduser(
        "~/metamon_runs/mini_online_psro_v1.4/mini_online_psro_v1.4/"
        "ckpts/policy_weights/policy_epoch_740.pt"
    )
)
BATTLE_FORMAT = "gen1ou"
TEAM_SET = "competitive"
NUM_PARALLEL = 2  # search eval needs batched obs (eval_search clamps >=2)
SEED = 42


def gpu_checkpoint_available() -> bool:
    """True only when CUDA is available AND the frozen checkpoint file exists."""
    try:
        import torch as _t

        cuda = bool(_t.cuda.is_available())
    except Exception:
        cuda = False
    return cuda and CKPT_PATH.exists()


# Module-level skip marker: applied by the gated test files via ``pytestmark``.
gpu_required = pytest.mark.skipif(
    not gpu_checkpoint_available(),
    reason=(
        "requires CUDA + the frozen MiniOnlinePsroV1_4 checkpoint at "
        f"{CKPT_PATH} (Phase 0B/0F MUST-RUN); set METAMON_CACHE_DIR and run on GPU."
    ),
)


# ---------------------------------------------------------------------------
# fixture bundle
# ---------------------------------------------------------------------------


@dataclass
class FrozenBundle:
    """Everything a gated test needs: the frozen policy, the live env, and a
    runner factory. Modeled on ``eval_search.run_search_eval``."""

    env: Any
    eval_driver: Any
    opponent: Any
    eval_policy: Any
    opponent_policy: Any
    model: Any
    opp_model: Any
    agent: Any
    action_dim: int
    device: torch.device
    reward_multiplier: float
    eval_action_space: Any
    opponent_action_space: Any
    battle_format: str = BATTLE_FORMAT

    def make_runner(self, config) -> Any:
        from metamon.rl.experimental.test_time_search.search_driver import (
            SearchEvalRunner,
        )

        return SearchEvalRunner(
            env=self.env,
            eval_driver=self.eval_driver,
            opponent=self.opponent,
            eval_policy=self.eval_policy,
            opponent_policy=self.opponent_policy,
            eval_action_space=self.eval_action_space,
            opponent_action_space=self.opponent_action_space,
            eval_reward_function=self.model.reward_function,
            opponent_reward_function=self.opp_model.reward_function,
            config=config,
            device=self.device,
            action_dim=self.action_dim,
            battle_format=self.battle_format,
            reward_multiplier=self.reward_multiplier,
        )

    def trunk_obs(self, lane_idx: int = 0):
        """A real eval-side obs + legal list for one trunk lane (mirrors
        ``SearchEvalRunner._build_obs``)."""
        import numpy as np

        lane = self.env.lanes[lane_idx]
        side = self.env.eval_side
        state = lane.universal_state(side)
        obs = self.env.eval_obs_spaces[lane_idx % self.env.batched_envs].state_to_obs(
            state
        )
        legal = lane.legal_action_indices(side, self.eval_action_space, state)
        n = self.eval_action_space.gym_space.n
        mask = np.ones((n,), dtype=bool)
        for idx in legal:
            if 0 <= idx < n:
                mask[idx] = False
        obs["illegal_actions"] = mask
        return obs, legal


@pytest.fixture
def frozen_env_bundle():
    """Load the frozen checkpoint, build the real env + drivers, reset once so
    lanes are live, yield a :class:`FrozenBundle`, and tear everything down.

    Requires CUDA + the checkpoint (skip otherwise -- use ``gpu_required`` on
    the test module so collection skips cheaply without importing heavy deps).
    """
    import metamon.env
    from metamon.env.vectorized.amago_policy import AmagoLadderPolicyDriver
    from metamon.env.vectorized.opponent import (
        AmagoBatchedOpponent,
    )  # noqa: F401  (consistency)
    from metamon.env.vectorized.vector_env import BattleAgainstMetamon
    from metamon.rl.pretrained import get_pretrained_model

    if not gpu_checkpoint_available():
        pytest.skip("CUDA or checkpoint unavailable")

    dev = torch.device("cuda")
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    model = get_pretrained_model(AGENT_NAME)
    agent = model.initialize_agent(
        checkpoint=CHECKPOINT_EPOCH, log=False, action_temperature=1.0
    )
    policy = agent.policy.to(dev)
    policy.eval()
    action_dim = model.action_space.gym_space.n

    opp_model = model  # self-play default
    opp_agent = agent
    opp_policy = opp_agent.policy.to(dev)
    opp_policy.eval()

    team_set = metamon.env.get_metamon_teams(BATTLE_FORMAT, TEAM_SET)
    env = BattleAgainstMetamon(
        battle_format=BATTLE_FORMAT,
        observation_space=model.observation_space,
        action_space=model.action_space,
        reward_function=model.reward_function,
        team_set=team_set,
        opponent_model=opp_model,
        opponent_checkpoint=CHECKPOINT_EPOCH,
        opponent_sample=True,
        batched_envs=NUM_PARALLEL,
        n_workers=1,
        eval_player_side=0,
        seed=SEED,
        device=str(dev),
    )
    eval_driver = AmagoLadderPolicyDriver(
        policy=policy,
        device=dev,
        num_lanes=env.batched_envs,
        action_dim=action_dim,
        sample=True,
    )
    bundle = FrozenBundle(
        env=env,
        eval_driver=eval_driver,
        opponent=env.opponent,
        eval_policy=policy,
        opponent_policy=opp_policy,
        model=model,
        opp_model=opp_model,
        agent=agent,
        action_dim=action_dim,
        device=dev,
        reward_multiplier=float(getattr(agent, "reward_multiplier", 10.0)),
        eval_action_space=model.action_space,
        opponent_action_space=opp_model.action_space,
    )
    try:
        obs, info = env.reset()
        yield bundle
    finally:
        # Tear down aggressively: the gated tests are repeated per test, and any
        # residual GPU/subprocess state raises system load for the later
        # CPU-only sim tests (a pre-existing ``test_sim_fork`` flake is
        # load-sensitive). Close the env (reaps the Node subprocess), drop every
        # heavy ref, force a GC pass, and clear the CUDA cache.
        import gc

        try:
            env.close()
        except Exception:
            pass
        for _ref in (
            policy,
            opp_policy,
            agent,
            opp_agent,
            model,
            opp_model,
            eval_driver,
            env,
        ):
            try:
                del _ref
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
