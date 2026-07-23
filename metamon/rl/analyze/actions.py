"""Human-readable labels for universal action indices given a UniversalState."""

from __future__ import annotations

from typing import Optional

from metamon.interface import (
    UniversalAction,
    UniversalState,
    consistent_move_order,
    consistent_pokemon_order,
)


def action_label(state: UniversalState, action_idx: int) -> str:
    """Map a universal action index to a short display label."""
    if action_idx < 0:
        return "unrevealed"
    if action_idx >= 9:
        moves = consistent_move_order(state.player_active_pokemon.moves)
        mi = action_idx - 9
        if 0 <= mi < len(moves):
            return f"Tera {moves[mi].name}"
        return f"Tera move {mi}"
    if action_idx <= 3:
        moves = consistent_move_order(state.player_active_pokemon.moves)
        if action_idx < len(moves):
            return moves[action_idx].name
        return f"Move {action_idx}"
    # switches 4–8
    switches = consistent_pokemon_order(state.available_switches)
    si = action_idx - 4
    if 0 <= si < len(switches):
        return f"Switch {switches[si].name}"
    return f"Switch {si}"


def legal_action_labels(state: UniversalState) -> dict[int, str]:
    """Map each maybe-legal action index to its label."""
    out: dict[int, str] = {}
    for ua in UniversalAction.maybe_valid_actions(state):
        out[ua.action_idx] = action_label(state, ua.action_idx)
    return out


def sprite_id(species: Optional[str]) -> str:
    """Alphanumeric lowercase id for Showdown sprite URLs."""
    if not species:
        return "unknown"
    return "".join(ch for ch in species.lower() if ch.isalnum())
