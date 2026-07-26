"""Policy recurrent-state forking for test-time search branches.

A search branch needs an independent copy of the frozen policy's recurrent
state so each rollout can advance its own Transformer KV cache / RL² state /
time index without touching the trunk (or other branches).

The frozen checkpoint uses AMAGO's ``TformerTrajEncoder`` whose hidden state is
a ``TformerHiddenState(key_cache, val_cache, seq_lens)``:

  * ``key_cache.data`` / ``val_cache.data``: ``(n_layers, batch, max_seq_len,
    n_heads, head_dim)`` bfloat16 tensors (a sliding-window KV cache).
  * ``seq_lens``: ``(batch,)`` int32.

``AmagoLadderPolicyDriver._snapshot_hidden`` already clones these for *inactive*
lanes. We generalize that into:

  * :func:`fork_hidden` — return an independent ``TformerHiddenState`` holding
    one trunk lane's cached state at a new (batch=1) position.
  * :func:`scatter_hidden` — copy one trunk lane's cached state into many fork
    lanes of a fresh batched hidden state (one KV write per branch, no Python
    loop over layers).

Branches also carry independent ``rl2s`` rows and ``step_counts`` (cheap numpy
copies). Tensors are not copied more than necessary: a fork lane's KV slice is
written once from the trunk; subsequent rollout forwards overwrite it in place.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np
import torch


@dataclass
class PolicyBranchState:
    """Independent recurrent state for one search branch.

    ``hidden`` is a batched ``TformerHiddenState`` with ``batch == n_branches``;
    each branch lane ``i`` holds the trunk lane's KV state. ``rl2s`` and
    ``step_counts`` are per-branch copies.
    """

    hidden: Any  # amago TformerHiddenState (batch = n_branches)
    rl2s: np.ndarray  # (n_branches, action_dim + 1)
    step_counts: np.ndarray  # (n_branches,)
    trunk_lane: int  # the trunk lane this was forked from


def _trunk_lane_slice(hidden: Any, lane: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Return (key_slice, val_slice, seq_len) for one lane of a batched hidden."""
    k = hidden.key_cache.data[
        :, lane : lane + 1
    ]  # (layers, 1, max_seq, heads, head_dim)
    v = hidden.val_cache.data[:, lane : lane + 1]
    sl = int(hidden.seq_lens[lane].item())
    return k, v, sl


def fork_hidden(
    trunk_hidden: Any, trunk_lane: int, n_branches: int, device: torch.device
) -> Any:
    """Create a batched hidden state for ``n_branches`` forks of ``trunk_lane``.

    Each branch lane starts from the trunk lane's KV cache + seq_len. Uses one
    ``expand``+``copy_`` per cache so we don't loop over layers.
    """
    from amago.nets.transformer import Cache, TformerHiddenState

    n_layers = trunk_hidden.n_layers
    max_seq_len = trunk_hidden.key_cache.max_seq_len
    n_heads = trunk_hidden.key_cache.data.shape[3]
    head_dim = trunk_hidden.key_cache.data.shape[4]
    dtype = trunk_hidden.key_cache.data.dtype

    k_src, v_src, sl = _trunk_lane_slice(trunk_hidden, trunk_lane)
    # Broadcast the single-lane slice to all branches.
    k_data = k_src.expand(n_layers, n_branches, max_seq_len, n_heads, head_dim).clone()
    v_data = v_src.expand(n_layers, n_branches, max_seq_len, n_heads, head_dim).clone()

    key_cache = Cache(
        device=device,
        dtype=dtype,
        layers=n_layers,
        batch_size=n_branches,
        max_seq_len=max_seq_len,
        n_heads=n_heads,
        head_dim=head_dim,
    )
    key_cache.data = k_data
    val_cache = Cache(
        device=device,
        dtype=dtype,
        layers=n_layers,
        batch_size=n_branches,
        max_seq_len=max_seq_len,
        n_heads=n_heads,
        head_dim=head_dim,
    )
    val_cache.data = v_data
    seq_lens = torch.full((n_branches,), sl, dtype=torch.int32, device=device)
    return TformerHiddenState(
        key_cache=key_cache, val_cache=val_cache, seq_lens=seq_lens
    )


def make_branch_state(
    trunk_driver, trunk_lane: int, n_branches: int, device: torch.device
) -> PolicyBranchState:
    """Fork a trunk lane's full policy state into ``n_branches`` branches."""
    hidden = fork_hidden(trunk_driver.hidden_state, trunk_lane, n_branches, device)
    rl2 = np.repeat(trunk_driver.rl2s[trunk_lane : trunk_lane + 1], n_branches, axis=0)
    steps = np.repeat(
        trunk_driver.step_counts[trunk_lane : trunk_lane + 1], n_branches, axis=0
    )
    return PolicyBranchState(
        hidden=hidden, rl2s=rl2.copy(), step_counts=steps.copy(), trunk_lane=trunk_lane
    )
