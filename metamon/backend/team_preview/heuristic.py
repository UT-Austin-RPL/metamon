"""Usage-stats-driven heuristic team-preview strategy.

``HeuristicTeamPreview`` is a drop-in alternative to
:class:`~metamon.backend.team_preview.preview.TeamPreviewModel`: it exposes the
same ``predict_lead(...)`` signature and a ``trained_formats`` attribute, so any
call site that accepts a team-preview *model* can accept this *strategy* without
further changes.

Lead selection scores each of our Pokemon as a candidate lead against the
revealed opponents using Smogon usage stats:

* ``+w_off * checks(O)[L]`` for each opponent ``O`` -- how strongly our lead
  ``L`` is listed as a check/counter of ``O`` (good: we threaten them).
* ``-w_def * checks(L)[O]`` for each opponent ``O`` -- how strongly ``O`` is
  listed as a check/counter of ``L`` (bad: they threaten us).
* ``+w_lead * leadBonus(L)`` rewards hazard/momentum movesets that make for a
  good lead.

The check-scoring precedent mirrors ``metamon/baselines/base.py``. Only the lead
is decided here; the remaining slots are randomized by the shared order helper to
keep the back-order in-distribution.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from metamon.config import format_for_agent
from metamon.backend.replay_parser.str_parsing import pokemon_name, move_name
from metamon.backend.team_prediction.usage_stats import (
    get_usage_stats,
    DEFAULT_USAGE_RANK,
)


# Move ids that make a Pokemon a good lead (hazards / momentum / hazard control),
# with rough weights. Keyed by normalized move id (see ``move_name``).
_LEAD_MOVE_BONUS: Dict[str, float] = {
    "stealthrock": 0.50,
    "spikes": 0.30,
    "stickyweb": 0.40,
    "toxicspikes": 0.20,
    "uturn": 0.20,
    "voltswitch": 0.20,
    "flipturn": 0.20,
    "rapidspin": 0.15,
    "defog": 0.15,
    "taunt": 0.10,
}

# Above this usage fraction a move counts as "the lead has it" when we only have
# usage stats (not the actual team's moves).
_MOVE_USAGE_THRESHOLD = 0.30


class _AllFormats:
    """``trained_formats`` sentinel: accept any format and fall back gracefully."""

    def __contains__(self, _item) -> bool:
        return True

    def __repr__(self) -> str:  # nicer logging in the player hook
        return "<any format with usage stats>"


class HeuristicTeamPreview:
    """Pick a lead from Smogon usage-stats checks/counters (no neural net)."""

    def __init__(
        self,
        *,
        rank: int = DEFAULT_USAGE_RANK,
        w_off: float = 1.0,
        w_def: float = 1.0,
        w_lead: float = 0.0,
        use_argmax: bool = True,
    ):
        self.rank = int(rank)
        self.w_off = float(w_off)
        self.w_def = float(w_def)
        self.w_lead = float(w_lead)
        self.use_argmax = bool(use_argmax)
        self.trained_formats = _AllFormats()
        self._stats_cache: Dict[str, object] = {}

    # ----- usage stats -----------------------------------------------------

    def _stats(self, battle_format: str):
        fmt = format_for_agent(battle_format)
        if fmt not in self._stats_cache:
            # rank fallback is allowed inside get_usage_stats; if nothing is on
            # disk this raises and the caller (player hook) falls back to random.
            self._stats_cache[fmt] = get_usage_stats(fmt, rank=self.rank)
        return self._stats_cache[fmt]

    def _entry(self, stats, name: str) -> Optional[dict]:
        try:
            return stats[name]
        except Exception:
            return None

    @staticmethod
    def _norm_checks(entry: Optional[dict]) -> Dict[str, float]:
        if not entry:
            return {}
        checks = entry.get("checks", {}) or {}
        out: Dict[str, float] = {}
        for k, v in checks.items():
            try:
                out[pokemon_name(k)] = float(v)
            except (TypeError, ValueError):
                continue
        return out

    def _lead_bonus(self, entry: Optional[dict], lead_moves: List[str]) -> float:
        bonus = 0.0
        if lead_moves:
            have = {move_name(m) for m in lead_moves if m}
            for move_id, weight in _LEAD_MOVE_BONUS.items():
                if move_id in have:
                    bonus += weight
            return bonus
        # No explicit moves: infer from the species' usage-stats moveset.
        if not entry:
            return 0.0
        moves = entry.get("moves", {}) or {}
        usage_by_id: Dict[str, float] = {}
        for k, v in moves.items():
            try:
                usage_by_id[move_name(k)] = float(v)
            except (TypeError, ValueError):
                continue
        for move_id, weight in _LEAD_MOVE_BONUS.items():
            if usage_by_id.get(move_id, 0.0) >= _MOVE_USAGE_THRESHOLD:
                bonus += weight
        return bonus

    def _score_lead(
        self,
        stats,
        lead: str,
        lead_moves: List[str],
        opponents: List[str],
    ) -> float:
        lead_id = pokemon_name(lead)
        lead_entry = self._entry(stats, lead)
        lead_checks = self._norm_checks(lead_entry)

        offense = 0.0
        defense = 0.0
        for opp in opponents:
            opp_entry = self._entry(stats, opp)
            opp_checks = self._norm_checks(opp_entry)
            # our lead is listed as a check/counter of this opponent -> good
            offense += opp_checks.get(lead_id, 0.0)
            # this opponent is listed as a check/counter of our lead -> bad
            defense += lead_checks.get(pokemon_name(opp), 0.0)

        bonus = self._lead_bonus(lead_entry, lead_moves)
        return self.w_off * offense - self.w_def * defense + self.w_lead * bonus

    # ----- public API (matches TeamPreviewModel.predict_lead) --------------

    def predict_lead(
        self,
        our_team: List[str],
        our_team_moves: Optional[List[List[str]]] = None,
        our_team_abilities: Optional[List[str]] = None,
        our_team_items: Optional[List[str]] = None,
        opponent_team: Optional[List[str]] = None,
        battle_format: Optional[str] = None,
        device: Optional[str] = None,
    ) -> Tuple[str, np.ndarray, List[str]]:
        """Choose a lead from usage-stats checks/counters.

        Returns ``(lead_name, scores, team_names)`` where ``scores`` aligns with
        ``team_names`` (the input ``our_team`` order). Deterministic tie-break to
        the lowest slot. Robust to unknown species / missing stats.
        """
        if not our_team:
            raise ValueError("our_team is empty")
        our_team_moves = our_team_moves or [[] for _ in our_team]
        opponent_team = opponent_team or []

        if battle_format is None:
            raise ValueError("HeuristicTeamPreview.predict_lead requires battle_format")
        stats = self._stats(battle_format)

        scores = np.array(
            [
                self._score_lead(
                    stats,
                    lead,
                    our_team_moves[i] if i < len(our_team_moves) else [],
                    opponent_team,
                )
                for i, lead in enumerate(our_team)
            ],
            dtype=np.float64,
        )

        if self.use_argmax:
            lead_idx = int(np.argmax(scores))  # ties -> lowest slot
        else:
            shifted = scores - scores.max()
            weights = np.exp(shifted)
            probs = weights / weights.sum()
            lead_idx = int(np.random.choice(len(our_team), p=probs))

        return our_team[lead_idx], scores, list(our_team)
