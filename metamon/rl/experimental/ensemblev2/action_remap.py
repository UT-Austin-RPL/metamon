from __future__ import annotations

import torch

from metamon.interface import ActionSpace, UniversalAction

# Canonical comparison space = DefaultActionSpace layout:
#   0-3 moves, 4-8 switches, 9-12 tera-moves.
CANONICAL_ACTION_DIM = 13


class ActionTranslator:
    """Translate between a member's own action space and canonical universal indices.

    There is no literal ``UniversalActionSpace`` class; the shared comparison
    point is :class:`~metamon.interface.UniversalAction`'s ``action_idx`` (the
    13-wide ``DefaultActionSpace`` layout). Each member's
    :class:`~metamon.interface.ActionSpace` supplies the mapping via
    ``action_to_agent_output`` (universal -> member) and ``agent_output_to_action``
    (member -> universal). For the current action spaces these are pure index maps
    (the ``state`` arg is unused), so translation works from the obs/mask alone.
    """

    def __init__(
        self, action_space: ActionSpace, canonical_dim: int = CANONICAL_ACTION_DIM
    ):
        self.action_space = action_space
        self.canonical_dim = canonical_dim
        self.member_dim = int(action_space.gym_space.n)

        # universal idx -> member idx (a member idx may receive several universal
        # idxs, e.g. MinimalActionSpace collapses tera-moves onto base moves).
        self.u2m: list[int] = []
        for u in range(canonical_dim):
            member_idx = action_space.action_to_agent_output(
                state=None, action=UniversalAction(action_idx=u)
            )
            self.u2m.append(int(member_idx))

        # member idx -> universal idx (injective for the current action spaces).
        self.m2u: list[int] = []
        for j in range(self.member_dim):
            universal = action_space.agent_output_to_action(state=None, agent_output=j)
            self.m2u.append(int(universal.action_idx))

    def canonical_illegal_to_member(
        self, canonical_illegal: torch.Tensor
    ) -> torch.Tensor:
        """Map a canonical illegal mask ``[..., 13]`` to ``[..., member_dim]``.

        A member action is legal if ANY universal action mapping to it is legal.
        """
        out_shape = canonical_illegal.shape[:-1] + (self.member_dim,)
        member_illegal = torch.ones(
            out_shape, dtype=torch.bool, device=canonical_illegal.device
        )
        for u in range(self.canonical_dim):
            m = self.u2m[u]
            member_illegal[..., m] = member_illegal[..., m] & canonical_illegal[..., u]
        return member_illegal

    def member_values_to_canonical(
        self, member_values: torch.Tensor, fill_value: float = 0.0
    ) -> torch.Tensor:
        """Scatter member-indexed values ``[..., member_dim]`` onto ``[..., 13]``.

        Canonical indices a member cannot express (e.g. tera for a member without
        tera actions) are left at ``fill_value``.
        """
        out_shape = member_values.shape[:-1] + (self.canonical_dim,)
        out = torch.full(
            out_shape,
            fill_value,
            dtype=member_values.dtype,
            device=member_values.device,
        )
        for j in range(self.member_dim):
            out[..., self.m2u[j]] = member_values[..., j]
        return out
