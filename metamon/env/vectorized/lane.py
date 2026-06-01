"""Per-lane battle state for the vectorized Showdown env.

A :class:`StreamBattleLane` holds two points of view of a single Showdown
battle, one per player, each backed by a :class:`MetamonBackendBattle`:

  * ``p1`` / ``p2`` -> one :class:`MetamonBackendBattle` POV per Showdown side
  * The env maps ``eval_side`` / ``opp_side`` onto these physical sides via
    ``eval_player_side`` (see :class:`~metamon.env.vectorized.vector_env.VectorizedShowdownEnv`).

Showdown's per-player streams are channel-filtered (each side only sees its own
hidden info and private ``|request|``), so feeding each side's stream into its
own battle reproduces exactly what the websocket path gives each player. We
mirror :meth:`MetamonPlayer._handle_battle_message`: every ``|`` protocol line
is handed to ``battle.parse_message`` and every ``|request|`` additionally goes
to ``battle.parse_request``.

The lane is transport-agnostic: :meth:`handle_chunk` is the only ingress and is
called by :class:`~metamon.env.vectorized.sim_process.ShowdownSimProcess`.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Tuple

import metamon.backend
from metamon.env.metamon_battle import MetamonBackendBattle
from metamon.interface import (
    ActionSpace,
    UniversalAction,
    UniversalState,
)

_LANE_LOGGER = logging.getLogger("metamon.env.vectorized.lane")
_LANE_LOGGER.addHandler(logging.NullHandler())

SIDES = ("p1", "p2")

# Request kinds, in the vocabulary the env cares about.
KIND_DONE = "done"
KIND_WAIT = "wait"
KIND_TEAMPREVIEW = "teampreview"
KIND_FORCESWITCH = "forceswitch"
KIND_MOVE = "move"
# Kinds that require a *learned* decision (agent action or opponent NN forward).
AGENT_KINDS = (KIND_FORCESWITCH, KIND_MOVE)


class StreamBattleLane:
    """Two-POV view of one Showdown battle driven by raw protocol text."""

    def __init__(self, lane_id: int, battle_format: str):
        self.lane_id = int(lane_id)
        self.battle_format = battle_format
        self.gen = metamon.backend.format_to_gen(battle_format)
        self._battles: Dict[str, MetamonBackendBattle] = {}
        # Latest request JSON per side. The raw sim (BattleStream) does not stamp
        # requests with an rqid (that is added by the server room layer), so we
        # synchronize decision cycles on per-side request counts instead. Every
        # `Battle.makeRequest` emits exactly one request to *each* side (idle
        # sides receive `{wait:true}`), so the two counts advance in lockstep.
        self.last_request: Dict[str, Optional[dict]] = {s: None for s in SIDES}
        self.request_serial: Dict[str, int] = {s: 0 for s in SIDES}
        # Per-side request count the env last consumed/acted on.
        self.settled_serial: Dict[str, int] = {s: 0 for s in SIDES}
        self.ended = False
        self.winner: Optional[str] = None
        self.error: Dict[str, Optional[str]] = {s: None for s in SIDES}
        # Set on ``|error|``, cleared once the env re-answers that side's
        # re-prompt. Unlike ``error``, survives the follow-up ``|request|``.
        self.reprompt_pending: Dict[str, bool] = {s: False for s in SIDES}
        self._mutation_serial: Dict[str, int] = {s: 0 for s in SIDES}
        self._state_cache: Dict[str, Tuple[int, UniversalState]] = {}
        self.reset_state()

    # ----- lifecycle -------------------------------------------------------

    def reset_state(self) -> None:
        """Create fresh per-POV battles for a new game in this lane."""
        self._battles = {
            side: MetamonBackendBattle(
                battle_tag=f"battle-{self.battle_format}-{self.lane_id}{side}",
                username=f"{side}-{self.lane_id}",
                logger=_LANE_LOGGER,
                save_replays=False,
                gen=self.gen,
            )
            for side in SIDES
        }
        for side in SIDES:
            self.last_request[side] = None
            self.request_serial[side] = 0
            self.settled_serial[side] = 0
            self.error[side] = None
            self.reprompt_pending[side] = False
            self._mutation_serial[side] = 0
            self._state_cache.pop(side, None)
        self.ended = False
        self.winner = None

    def _touch_side(self, side: str) -> None:
        """Record that ``side``'s parsed battle state changed."""
        self._mutation_serial[side] += 1
        self._state_cache.pop(side, None)

    def battle(self, side: str) -> MetamonBackendBattle:
        return self._battles[side]

    # ----- ingress ---------------------------------------------------------

    def handle_chunk(self, stream: str, data: str) -> None:
        if getattr(self, "_trace", None) is not None:
            self._trace.append((stream, data.split("\n", 1)[0][:50]))
        if stream == "omniscient":
            # Legacy host versions only; win/tie is handled on player streams.
            for line in data.split("\n"):
                self._scan_global(line)
            return
        if stream not in self._battles:
            return
        battle = self._battles[stream]
        for line in data.split("\n"):
            if not line.startswith("|"):
                continue
            parts = line.split("|")
            if len(parts) <= 1:
                continue
            cmd = parts[1]
            if cmd in ("t:", "expire", "uhtmlchange"):
                continue
            if cmd == "request":
                # Extract JSON without splitting on '|' (move/item names are safe
                # but JSON strings could in principle contain '|').
                body = line[len("|request|") :]
                if body:
                    try:
                        req = json.loads(body)
                    except json.JSONDecodeError:
                        continue
                    battle.parse_request(req)
                    self._on_request(stream, req)
                    self._touch_side(stream)
                continue
            # Hand every other protocol line to the metamon parser.
            try:
                battle.parse_message(parts)
                self._touch_side(stream)
            except Exception as exc:  # noqa: BLE001
                _LANE_LOGGER.debug(
                    "lane %s side %s parse_message failed on %r: %s",
                    self.lane_id,
                    stream,
                    line[:120],
                    exc,
                )
            self._scan_global(line)
            if cmd == "turn":
                # Mirror MetamonPlayer: keep the turnlist short to bound memory.
                battle._mm_battle.turnlist = battle._mm_battle.turnlist[-2:]
            elif cmd == "error":
                self.error[stream] = parts[2] if len(parts) > 2 else "error"
                self.reprompt_pending[stream] = True

    def _scan_global(self, line: str) -> None:
        if line.startswith("|win|"):
            if not self.ended:
                for side in SIDES:
                    self._touch_side(side)
            self.ended = True
            self.winner = line[len("|win|") :].strip() or None
        elif line == "|tie" or line.startswith("|tie|"):
            if not self.ended:
                for side in SIDES:
                    self._touch_side(side)
            self.ended = True

    def _on_request(self, side: str, req: dict) -> None:
        self.last_request[side] = req
        self.request_serial[side] += 1
        # A fresh actionable request clears any stale error for that side.
        self.error[side] = None

    # ----- decision-point introspection -----------------------------------

    def request_kind(self, side: str) -> str:
        if self.ended:
            return KIND_DONE
        req = self.last_request[side]
        if req is None:
            return KIND_DONE
        if req.get("wait"):
            return KIND_WAIT
        if req.get("teamPreview"):
            return KIND_TEAMPREVIEW
        force = req.get("forceSwitch")
        if force and (force[0] if isinstance(force, list) else force):
            return KIND_FORCESWITCH
        if "active" in req or req.get("active"):
            return KIND_MOVE
        # Fallback: a request with a usable side but no active block (rare).
        return KIND_MOVE

    def needs_agent_decision(self, side: str) -> bool:
        return self.request_kind(side) in AGENT_KINDS

    def _side_ready(self, side: str) -> bool:
        """Whether ``side`` has a fresh, *fully-materialized* decision available.

        A new request must have arrived (its count advanced past what the env last
        consumed). For move/forceswitch decisions we additionally require the
        battle's active Pokemon to be populated: Showdown flushes the ``|request|``
        in a separate (often earlier) chunk than the ``|switch|`` public log that
        sets ``active_pokemon`` (most visibly at battle start), so the request
        alone does not guarantee the parser has applied the switch-in yet. Gating
        on the active Pokemon avoids reading a half-applied state (which would make
        ``UniversalState.from_Battle`` dereference ``None``). ``wait``/``teampreview``
        decisions have no active Pokemon and need no such gate.
        """
        if self.request_serial[side] <= self.settled_serial[side]:
            return False
        kind = self.request_kind(side)
        # Force-switch requests follow a faint/removal; the outgoing active may
        # already be cleared before we answer. Only gate move decisions on the
        # active Pokemon being materialized (battle start |request| before |switch|).
        if kind == KIND_MOVE:
            return self._battles[side].active_pokemon is not None
        return True

    def decision_ready(self) -> bool:
        """True once a new, fully-synchronized decision cycle is available.

        Each ``makeRequest`` emits one request to both sides simultaneously, so a
        cycle is ready once both sides' requests have advanced past the one the env
        last consumed and materialized (or the battle ended).
        """
        if self.ended:
            return True
        return all(self._side_ready(s) for s in SIDES)

    def mark_settled(self) -> None:
        """Record that the env has consumed/acted on the current cycle."""
        for s in SIDES:
            self.settled_serial[s] = self.request_serial[s]

    # ----- observations / legality ----------------------------------------

    def universal_state(self, side: str) -> UniversalState:
        serial = self._mutation_serial[side]
        cached = self._state_cache.get(side)
        if cached is not None and cached[0] == serial:
            return cached[1]
        state = UniversalState.from_Battle(self._battles[side])
        self._state_cache[side] = (serial, state)
        return state

    def legal_action_indices(
        self,
        side: str,
        action_space: ActionSpace,
        state: Optional[UniversalState] = None,
    ) -> List[int]:
        """Legal agent-action indices, mirroring ``PokeEnvWrapper._update_legal_actions``."""
        battle = self._battles[side]
        if state is None:
            state = self.universal_state(side)
        legal_actions = UniversalAction.definitely_valid_actions(
            state=state, battle=battle
        )
        return [
            action_space.action_to_agent_output(state=state, action=action)
            for action in legal_actions
        ]
