from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from metamon.rl.experimental.ensemblev2.action_remap import ActionTranslator


class MemberFeatureExtractor:
    """Runs one member's net and returns canonical-indexed features for a step.

    Q-values are computed by following the amago loss-code progression
    (``Agent.forward`` / ``MultiTaskAgent.forward``), evaluated for every legal
    candidate action in parallel on the batch axis, then decoded to scalar reward
    units format-agnostically (``bin_dist_to_raw_vals`` for distributional
    ``NCriticsTwoHot`` critics, else PopArt-denormalized scalar ``NCritics``).
    The member's full gamma axis is preserved.
    """

    def __init__(
        self,
        policy: torch.nn.Module,
        device: torch.device,
        translator: ActionTranslator,
    ):
        self.policy = policy
        self.device = device
        self.translator = translator
        self.member_dim = translator.member_dim
        self.num_gammas = len(policy.gammas)

    def _actor_probs(
        self, traj_emb: torch.Tensor, member_illegal: torch.Tensor
    ) -> torch.Tensor:
        """Per-gamma legal-normalized action distribution: ``(B, num_gammas, member_dim)``."""
        straight_from_obs: dict[str, torch.Tensor] = {}
        for key in self.policy.pass_obs_keys_to_actor:
            if key == "illegal_actions":
                straight_from_obs[key] = member_illegal
        if "illegal_actions" not in straight_from_obs:
            # Actor still needs the mask to renormalize over legal actions.
            straight_from_obs["illegal_actions"] = member_illegal
        action_dist = self.policy.actor(traj_emb, straight_from_obs=straight_from_obs)
        # probs: (B, L, num_gammas, member_dim) -> take the current (last) timestep
        return action_dist.probs[:, -1, :, :].detach()

    def _critic_q(self, traj_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-gamma Q (mean and critic-ensemble std) for every member action.

        Returns ``(q_mean, q_std)``, each ``(B, member_dim, num_gammas)``.

        Mirrors ``s_a_g = (s_rep, a_buffer.unsqueeze(0)); self.critics(*s_a_g)``
        from the amago training forward, with the candidate actions placed on the
        batch axis so we score them all at once. ``q_std`` is the spread *across
        this agent's own critic ensemble* (the ``C`` axis) in scalar reward units
        -- a per-member epistemic-uncertainty signal -- rather than being averaged
        away.
        """
        policy = self.policy
        N = self.member_dim
        G = self.num_gammas
        traj_last = traj_emb[:, -1:, :]  # (B, 1, D)
        B, _, D = traj_last.shape

        # candidate one-hots, one per member action, broadcast across the batch
        eye = torch.eye(N, device=self.device, dtype=traj_last.dtype)  # (N, N)
        state_rep = traj_last.repeat_interleave(N, dim=0)  # (B*N, 1, D)
        onehot_rep = eye.unsqueeze(0).expand(B, N, N).reshape(B * N, N)  # (B*N, N)
        buffer = onehot_rep.view(1, B * N, 1, 1, N).repeat(1, 1, 1, G, 1)
        critic_actions = policy.actor.policy_dist.action_from_buffer(buffer)

        critic_values = policy.critics(state_rep, critic_actions)
        if hasattr(policy.critics, "bin_dist_to_raw_vals"):
            critic_values = policy.critics.bin_dist_to_raw_vals(critic_values)
        else:
            critic_values = policy.popart(critic_values, normalized=False)
        # (K=1, B*N, L=1, C, G, 1); C (dim=3) is the per-agent critic ensemble.
        q_mean = critic_values.mean(dim=3)[0, :, 0, :, 0]  # (B*N, G)
        q_std = critic_values.std(dim=3, unbiased=False)[0, :, 0, :, 0]  # (B*N, G)
        return (
            q_mean.view(B, N, G).detach(),
            q_std.view(B, N, G).detach(),
        )

    @torch.no_grad()
    def forward(
        self,
        member_obs: dict[str, torch.Tensor],
        rl2s: torch.Tensor,
        time_idxs: torch.Tensor,
        hidden_state: Any,
        canonical_illegal: torch.Tensor,
        gather_q: bool = True,
    ) -> tuple[Any, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Run the member; return ``(next_hidden, probs, q_mean, q_std)`` (canonical).

        ``probs`` is ``(B, num_gammas, 13)`` with 0.0 in illegal / inexpressible
        slots. ``q_mean`` and ``q_std`` are ``(B, num_gammas, 13)`` with NaN in
        slots the member cannot express (or ``None`` if ``gather_q`` is False).
        ``q_std`` is the std across the member's own critic ensemble.
        """
        member_obs = {
            key: value.to(self.device, non_blocking=True)
            for key, value in member_obs.items()
        }
        rl2s = rl2s.to(self.device, non_blocking=True)
        time_idxs = time_idxs.to(self.device, non_blocking=True)
        canonical_illegal = canonical_illegal.to(self.device, non_blocking=True)
        member_illegal = self.translator.canonical_illegal_to_member(canonical_illegal)

        traj_emb, next_hidden = self.policy.get_state_embedding(
            obs=member_obs,
            rl2s=rl2s,
            time_idxs=time_idxs,
            hidden_state=hidden_state,
        )

        probs_member = self._actor_probs(traj_emb, member_illegal)  # (B, G, N)
        probs_canonical = self.translator.member_values_to_canonical(
            probs_member, fill_value=0.0
        )  # (B, G, 13)
        probs_np = probs_canonical.float().cpu().numpy()

        q_np = None
        q_std_np = None
        if gather_q:
            q_member, q_std_member = self._critic_q(traj_emb)  # each (B, N, G)
            q_canonical = self.translator.member_values_to_canonical(
                q_member.transpose(1, 2), fill_value=float("nan")
            )  # (B, G, 13)
            q_std_canonical = self.translator.member_values_to_canonical(
                q_std_member.transpose(1, 2), fill_value=float("nan")
            )  # (B, G, 13)
            q_np = q_canonical.float().cpu().numpy()
            q_std_np = q_std_canonical.float().cpu().numpy()

        return next_hidden, probs_np, q_np, q_std_np
