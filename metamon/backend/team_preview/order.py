"""The Showdown team-preview contract, in one place.

A team-preview choice is ALWAYS a *full order string*: a sequence of 1-indexed
party slots where the first entry is the lead and the length equals
``maxTeamSize`` (the number of Pokemon you bring). In bring-6 singles only the
lead is competitively meaningful; the remaining slots are arbitrary and poke-env
shuffles them (``random_teampreview``). We mirror that: the lead is chosen by a
strategy and the rest are genuinely random-shuffled to stay in-distribution.

The only thing that differs between transports is the wire prefix:

* Websocket / PS server (poke-env): a chat command ``"/team 312456"``.
* In-process sim ``BattleStream`` (vectorized): the raw choice ``"team 312456"``
  (no slash), same convention as moves.

Both call sites build the order through :func:`order_from_lead` /
:func:`build_team_order` so the back-order policy and the two wire formats can
never drift apart.
"""

from __future__ import annotations

import random as _random
from typing import List, Optional, Sequence

from metamon.backend.replay_parser.str_parsing import pokemon_name


def order_from_lead(
    lead_slot: int,
    n_slots: int,
    max_team_size: Optional[int] = None,
    rng=None,
) -> List[int]:
    """Build a full team order: ``lead_slot`` first, remaining slots shuffled.

    Args:
        lead_slot: 1-indexed party slot to lead with.
        n_slots: number of Pokemon on the team (party size).
        max_team_size: how many slots to submit (``maxTeamSize``). Defaults to
            ``n_slots`` (bring-everything formats).
        rng: object with a ``shuffle`` method (e.g. ``random.Random`` or the
            ``random`` module). Defaults to the global ``random`` module.

    Returns:
        A list of distinct 1-indexed slots, lead first, length
        ``min(max_team_size, n_slots)``.
    """
    rng = rng or _random
    if not (1 <= lead_slot <= n_slots):
        raise ValueError(f"lead_slot={lead_slot} out of range for n_slots={n_slots}")
    if max_team_size is None:
        max_team_size = n_slots
    rest = [s for s in range(1, n_slots + 1) if s != lead_slot]
    rng.shuffle(rest)
    order = [lead_slot] + rest
    return order[: max(1, min(int(max_team_size), n_slots))]


def build_team_order(positions: Sequence[int], *, slash: bool) -> str:
    """Render a validated order to the wire format for the chosen transport.

    Args:
        positions: 1-indexed, lead-first, de-duplicated slot order.
        slash: ``True`` -> ``"/team ..."`` (websocket); ``False`` -> ``"team ..."``
            (in-process sim stream).
    """
    if not positions:
        raise ValueError("team order must contain at least the lead slot")
    if len(set(positions)) != len(positions):
        raise ValueError(f"team order has duplicate slots: {positions}")
    if any(p < 1 for p in positions):
        raise ValueError(f"team order must be 1-indexed: {positions}")
    body = "".join(str(p) for p in positions)
    return ("/team " if slash else "team ") + body


def resolve_lead_slot(team_species: Sequence[str], lead_name: str) -> Optional[int]:
    """Map a predicted lead species back to its 1-indexed slot.

    Compares on normalized species ids so callers can pass raw Showdown species
    (``"Great Tusk"``) or already-normalized names. Returns ``None`` if no slot
    matches (caller should fall back).
    """
    target = pokemon_name(lead_name)
    for i, species in enumerate(team_species):
        if pokemon_name(species) == target:
            return i + 1
    return None


def species_from_details(details: str) -> str:
    """Extract the species from a Showdown request ``details`` field.

    ``details`` looks like ``"Great Tusk, L50, M"`` or ``"Pikachu-Alola"``; the
    species is everything before the first comma.
    """
    return (details or "").split(",")[0].strip()
