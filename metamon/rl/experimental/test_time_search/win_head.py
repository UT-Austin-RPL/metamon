"""kimi-search M3: trained win-probability head for terminal-aligned search.

The Phase A / M1 gates showed the *shaped* critic (all gamma heads) is only
~0.13-0.17 Spearman-aligned with terminal win at the action level on squirtle
-- the "estimator-positive, game-negative" blocker. Per skill §37's failure
path and Ataraxos (whose value target IS the game outcome), this module trains
a small **win-probability head** on the frozen squirtle trajectory-encoder
embeddings:

    WinHead:  emb (d_model)  ->  MLP  ->  logits (A,)   # per action, like the
                                                            discrete critic
    Q_win(s, a) = sigmoid(logits[a])        # per-action win prob
    V_win(s)    = sum_a pi_base(a|s) * Q_win(s, a)      # policy expectation

Trained with binary cross-entropy against the battle outcome (win=1, loss=0;
draws excluded) on the run's own FIFO buffer (self-play battles, WIN/LOSS in
the filenames), so the states are in-distribution for the final policy. The
backbone is frozen; only the head trains.

The head is used by the search driver as a leaf value
(``search_leaf_value_mode="win_head"``): at a leaf, ``Q_win`` replaces the
shaped critic Q, so the search advantage is in *win-probability* units
(naturally calibrated for the KL ``beta``). See RESEARCH_PLAN_KIMI.md M3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class WinHead(nn.Module):
    """Per-action win-probability head on a frozen trajectory embedding.

    Mirrors the discrete-critic contract: input is the trajectory embedding
    ``(B, d_model)`` (or ``(B, 1, d_model)``), output is per-action logits
    ``(B, A)``. ``q(s, a) = sigmoid(logits[a])`` is the model's estimate of
    P(win | s, take a, then follow the base policy).
    """

    def __init__(
        self,
        d_model: int,
        action_dim: int,
        d_hidden: int = 512,
        n_layers: int = 2,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.action_dim = int(action_dim)
        layers = []
        d_in = self.d_model
        for _ in range(max(int(n_layers) - 1, 0)):
            layers += [nn.Linear(d_in, d_hidden), nn.LeakyReLU(), nn.Dropout(dropout_p)]
            d_in = d_hidden
        layers += [nn.Linear(d_in, self.action_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """emb (B, d_model) or (B, 1, d_model) -> logits (B, A)."""
        if emb.ndim == 3:
            emb = emb.squeeze(1)
        return self.net(emb)

    def q_win(self, emb: torch.Tensor) -> torch.Tensor:
        """Per-action win probability (B, A) in [0, 1]."""
        return torch.sigmoid(self.forward(emb))

    def v_win(self, emb: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        """Policy-expectation win prob: sum_a probs(a) * Q_win(s, a) -> (B,)."""
        q = self.q_win(emb)  # (B, A)
        return (q * probs).sum(-1)


@dataclass
class WinHeadTrainResult:
    n_battles: int
    n_turns: int
    train_loss: float
    val_loss: float
    val_auc: float
    val_brier: float
    ckpt_path: str


def save_win_head(head: WinHead, path: str, meta: Optional[dict] = None) -> str:
    payload = {
        "state_dict": head.state_dict(),
        "d_model": head.d_model,
        "action_dim": head.action_dim,
        "meta": meta or {},
    }
    torch.save(payload, path)
    return path


def load_win_head(path: str, device: str = "cpu") -> WinHead:
    payload = torch.load(path, map_location=device, weights_only=False)
    head = WinHead(d_model=payload["d_model"], action_dim=payload["action_dim"])
    head.load_state_dict(payload["state_dict"])
    head.to(device).eval()
    return head
