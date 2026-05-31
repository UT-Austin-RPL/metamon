"""Map metamon agent actions to Showdown ``BattleStream`` choice strings.

Because each lane holds a real poke-env ``Battle`` (via ``MetamonBackendBattle``),
we can reuse metamon's canonical action decoding
(:meth:`UniversalAction.action_idx_to_BattleOrder`) and simply convert the
resulting ``BattleOrder`` message into the choice string ``BattleStream`` expects
(i.e. drop the ``/choose `` prefix). This mirrors what poke-env sends online; the
sim's ``Side.choose`` is the same code path.
"""

from __future__ import annotations

from typing import Optional

from poke_env.environment import Battle
from poke_env.player.battle_order import BattleOrder

from metamon.interface import UniversalAction


CHOOSE_PREFIX = "/choose "
DEFAULT_CHOICE = "default"


def _to_id(text: str) -> str:
    return "".join(ch for ch in (text or "").lower() if ch.isalnum())


def battle_order_to_choice(message: str) -> str:
    """Strip the ``/choose `` prefix to get a raw ``BattleStream`` choice."""
    if message.startswith(CHOOSE_PREFIX):
        return message[len(CHOOSE_PREFIX) :]
    if message.startswith("/"):
        # e.g. "/forfeit" — pass through without the leading slash.
        return message[1:]
    return message


def _switch_to_slot(order: BattleOrder, choice: str, request: Optional[dict]) -> str:
    """Rewrite ``switch <species>`` to ``switch <1-based slot>`` via the request.

    poke-env builds switch orders as ``/choose switch {pokemon.species}``. For a
    Pokemon that forme-changes mid-battle and reverts on switch-out (Morpeko,
    Minior, Aegislash, Palafin, ...), ``.species`` is the *battle* forme
    (``morpekohangry``) while Showdown reverts the benched copy and matches on its
    base species id (``morpeko``), so the species-name choice is rejected. The
    ``|request|`` ``side.pokemon`` list is ground truth and in party-slot order,
    and Showdown accepts ``switch <slot>`` unambiguously, so resolve to that.
    """
    if request is None or not choice.startswith("switch "):
        return choice
    target = choice[len("switch ") :].strip()
    if target.isdigit():
        return choice
    poke = getattr(order, "order", None)
    pokemon = request.get("side", {}).get("pokemon", []) or []
    if poke is None or not pokemon:
        return choice
    # The order's Pokemon may be in a battle forme; match either its current
    # species or its base species against each benched mon's (reverted) species.
    target_ids = {_to_id(getattr(poke, "species", "") or "")}
    base = getattr(poke, "base_species", None)
    if base:
        target_ids.add(_to_id(base))
    target_ids.discard("")
    for i, mon in enumerate(pokemon):
        details = mon.get("details", "") or ""
        species = details.split(",")[0].strip()
        if _to_id(species) in target_ids:
            return f"switch {i + 1}"
    return choice


def action_idx_to_choice(
    action_idx: int, battle: Battle, request: Optional[dict] = None
) -> Optional[str]:
    """Convert an agent action index to a choice string, or ``None`` if invalid.

    ``None`` signals an invalid/illegal action; the env decides the fallback
    (typically :data:`DEFAULT_CHOICE`, which lets Showdown auto-pick). When the
    side's ``|request|`` is supplied, switches are encoded by party slot (see
    :func:`_switch_to_slot`) so forme-changed Pokemon switch reliably.
    """
    order = UniversalAction.action_idx_to_BattleOrder(
        battle, action_idx=int(action_idx)
    )
    if order is None:
        return None
    choice = battle_order_to_choice(order.message)
    return _switch_to_slot(order, choice, request)
