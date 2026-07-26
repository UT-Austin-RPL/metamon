"""Deterministic keyed seed bank for test-time search branches (skill §7).

Search branches must NOT inherit the trunk's hidden future PRNG stream. The
trunk Showdown battle PRNG is advanced by the live game and is part of the
serialized snapshot. If a fork is restored from that snapshot with no
branch-only reseeding, every candidate root action is evaluated under the
trunk's actual hidden future random stream -- a future-chance oracle.

This module constructs a deterministic, common-random-number (CRN) seed bank.
For a searched root with ``K`` rollouts per action:

  * ``env_seed[root, k]`` is the branch-only Showdown PRNG seed for rollout
    index ``k``. It is **shared across all candidate root actions** ``a`` (CRN
    design -- reduces variance when comparing actions) and **different across**
    ``k != k'``. It never depends on candidate-action identity.
  * ``opp_root_key[root, k]`` seeds the opponent's root-action sampler for
    rollout ``k``; the sampled opponent root action is reused across all
    candidate evaluated-player actions for that ``k`` (skill §7: opponent root
    action coupling).
  * ``policy_rng_key[root, side, k, step]`` seeds a deterministic policy-
    sampling stream for deeper rollout actions, so the same uniform variate is
    used at the same logical (k, step) where practical.

The trunk battle is never reseeded or mutated. Environment RNG and policy-
sampling RNG are separate streams. All seeds are derived from a stable
``blake2b`` digest of the keyed tuple (no dependence on Python's salted
``hash``), so the seed table is reproducible across processes and machines.

Showdown's battle PRNG is constructed from a 4x uint16 seed (list of four ints
in [0, 0xFFFF]); see the ``determinism-and-seeds`` skill. ``branch_env_seed``
returns such a list, suitable for reseeding a forked Battle's PRNG.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

_UINT16 = 0xFFFF


def _digest(
    global_seed: int,
    battle_id: str,
    side: str,
    decision_idx: int,
    k: int,
    stream_kind: str,
) -> bytes:
    """Stable 32-byte digest of the keyed tuple (process/machine independent)."""
    s = f"tts|{global_seed}|{battle_id}|{side}|{decision_idx}|{k}|{stream_kind}"
    return hashlib.blake2b(s.encode("utf-8"), digest_size=32).digest()


def _u64(digest: bytes, word: int = 0) -> int:
    return int.from_bytes(digest[word * 8 : word * 8 + 8], "little")


def branch_env_seed(
    global_seed: int,
    battle_id: str,
    side: str,
    decision_idx: int,
    k: int,
) -> List[int]:
    """Return a 4x uint16 Showdown PRNG seed for rollout index ``k``.

    Shared across candidate actions (determined by (decision_idx, k) only, not
    by action identity). Different across ``k``.
    """
    h = _digest(global_seed, battle_id, side, decision_idx, k, "env")
    return [int.from_bytes(h[j * 2 : j * 2 + 2], "little") & _UINT16 for j in range(4)]


def opp_root_key(global_seed: int, battle_id: str, decision_idx: int, k: int) -> int:
    """Seed for the opponent root-action sampler for rollout ``k``.

    The sampled opponent root action must be reused across all candidate
    evaluated-player actions for the same ``k`` (opponent root coupling).
    """
    return _u64(_digest(global_seed, battle_id, "opp", decision_idx, k, "opp_root"))


def policy_rng_key(
    global_seed: int,
    battle_id: str,
    side: str,
    decision_idx: int,
    k: int,
    step: int,
) -> int:
    """Seed for a deterministic policy-sampling stream at (root, side, k, step).

    Deeper rollout actions may be sampled from different distributions once
    observations diverge, but using a keyed stream means the same uniform
    variate is consumed at the same logical (k, step) where practical. Keep
    environment RNG and policy-sampling RNG as separate streams.
    """
    return _u64(
        _digest(global_seed, battle_id, side, decision_idx, k, f"policy_step{step}")
    )


def make_rng(key: int) -> np.random.Generator:
    """Build a local numpy Generator from a 64-bit key (no global state)."""
    return np.random.default_rng(int(key) & 0xFFFFFFFFFFFFFFFF)


@dataclass
class RootSeedBank:
    """Precomputed CRN seed table for one searched root.

    ``env_seeds[k]`` is the 4x uint16 Showdown seed shared across all candidate
    actions for rollout ``k``. ``opp_root_keys[k]`` seeds the opponent root
    action sampler for rollout ``k``. ``rollout_index`` maps a global branch
    index (in the ``np.repeat(legal, K)`` layout) to its rollout index.
    """

    global_seed: int
    battle_id: str
    side: str
    decision_idx: int
    K: int
    env_seeds: List[List[int]] = field(default_factory=list)
    opp_root_keys: List[int] = field(default_factory=list)
    # stable hash of each env seed, for logging without leaking the raw stream
    env_seed_hashes: List[str] = field(default_factory=list)

    @classmethod
    def build(
        cls,
        global_seed: int,
        battle_id: str,
        side: str,
        decision_idx: int,
        K: int,
    ) -> "RootSeedBank":
        env_seeds = [
            branch_env_seed(global_seed, battle_id, side, decision_idx, k)
            for k in range(K)
        ]
        opp_root_keys = [
            opp_root_key(global_seed, battle_id, decision_idx, k) for k in range(K)
        ]
        env_seed_hashes = [
            hashlib.blake2b(
                ",".join(map(str, s)).encode("utf-8"),
                digest_size=8,
                person=b"tts_envseed",
            ).hexdigest()
            for s in env_seeds
        ]
        return cls(
            global_seed=global_seed,
            battle_id=battle_id,
            side=side,
            decision_idx=decision_idx,
            K=K,
            env_seeds=env_seeds,
            opp_root_keys=opp_root_keys,
            env_seed_hashes=env_seed_hashes,
        )

    def rollout_index_of(self, branch_index: int) -> int:
        """Map a global branch index (``np.repeat(legal, K)`` layout) to ``k``."""
        return int(branch_index) % self.K

    def env_seed_for_branch(self, branch_index: int) -> List[int]:
        return self.env_seeds[self.rollout_index_of(branch_index)]

    def opp_root_key_for_branch(self, branch_index: int) -> int:
        return self.opp_root_keys[self.rollout_index_of(branch_index)]

    def to_log_dict(self) -> Dict:
        return {
            "global_seed": self.global_seed,
            "battle_id": self.battle_id,
            "side": self.side,
            "decision_idx": self.decision_idx,
            "K": self.K,
            "env_seed_hashes": list(self.env_seed_hashes),
            "opp_root_keys": list(self.opp_root_keys),
        }
