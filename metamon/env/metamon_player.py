import random
from typing import List, Optional

import numpy as np
import orjson

from poke_env.player import Player
from poke_env.environment import AbstractBattle
from poke_env.exceptions import ShowdownException

from metamon.config import format_for_agent
from metamon.env.metamon_battle import MetamonBackendBattle, PokeAgentBackendBattle
from metamon.backend.showdown_dex import Dex
from metamon.backend.replay_parser.str_parsing import pokemon_name, move_name
from metamon.backend.team_preview.order import order_from_lead, build_team_order
from metamon.interface import UniversalPokemon


class MetamonPlayer(Player):
    """Extended Player with an optional team-preview strategy.

    A *strategy* is anything exposing ``predict_lead(...)`` and ``trained_formats``
    (e.g. :class:`~metamon.backend.team_preview.preview.TeamPreviewModel` or
    :class:`~metamon.backend.team_preview.heuristic.HeuristicTeamPreview`). When no
    strategy is set we fall back to poke-env's random team preview.
    """

    def __init__(
        self, *args, team_preview_strategy=None, team_preview_model=None, **kwargs
    ):
        """
        Initialize MetamonPlayer.

        Args:
            team_preview_strategy: Optional object with ``predict_lead`` /
                ``trained_formats`` used to choose the lead. If None, falls back to
                random team preview selection.
            team_preview_model: Backwards-compatible alias for
                ``team_preview_strategy``.
            *args, **kwargs: Arguments passed to Player.__init__
        """
        super().__init__(*args, **kwargs)
        self.team_preview_strategy = (
            team_preview_strategy
            if team_preview_strategy is not None
            else team_preview_model
        )

    @property
    def team_preview_model(self):
        """Backwards-compatible alias for :attr:`team_preview_strategy`."""
        return self.team_preview_strategy

    @team_preview_model.setter
    def team_preview_model(self, value):
        self.team_preview_strategy = value

    def create_metamon_battle(self, battle_tag: str) -> MetamonBackendBattle:
        return MetamonBackendBattle(
            battle_tag=battle_tag,
            username=self.username,
            logger=self.logger,
            save_replays=self._save_replays,
            gen=Dex.from_format(self.format).gen,
        )

    async def _create_battle(self, split_message: List[str]) -> AbstractBattle:
        """
        Override the default battle creation logic to use our own MetamonBackendBattle.
        """
        # We check that the battle has the correct format
        if split_message[1] == self._format and len(split_message) >= 2:
            # Battle initialisation
            battle_tag = "-".join(split_message)[1:]

            if battle_tag in self._battles:
                return self._battles[battle_tag]
            else:
                if self.format_is_doubles:
                    raise NotImplementedError("Metamon does not support doubles")
                else:
                    battle = self.create_metamon_battle(battle_tag)
                await self._battle_count_queue.put(None)
                if battle_tag in self._battles:
                    await self._battle_count_queue.get()
                    return self._battles[battle_tag]
                async with self._battle_start_condition:
                    self._battle_semaphore.release()
                    self._battle_start_condition.notify_all()
                    self._battles[battle_tag] = battle
                if self._start_timer_on_battle_start:
                    await self.ps_client.send_message("/timer on", battle.battle_tag)
                return battle
        else:
            self.logger.critical(
                "Unmanaged battle initialisation message received: %s", split_message
            )
            raise ShowdownException()

    async def _handle_battle_message(self, split_messages: List[List[str]]):
        """
        Override the default battle message handling logic to use our own MetamonBackendBattle.
        """
        if (
            len(split_messages) > 1
            and len(split_messages[1]) > 1
            and split_messages[1][1] == "init"
        ):
            battle_info = split_messages[0][0].split("-")
            battle = await self._create_battle(battle_info)
        else:
            battle = await self._get_battle(split_messages[0][0])

        for split_message in split_messages[1:]:
            # let the metamon replay parser see every message
            if len(split_message) <= 1:
                continue
            elif split_message[0] == "":
                battle.parse_message(split_message)

            # handle Player-level behavior for some message types
            if split_message[1] in self.MESSAGES_TO_IGNORE:
                pass
            elif split_message[1] == "request":
                if split_message[2]:
                    request = orjson.loads(split_message[2])
                    battle.parse_request(request)
                    if battle._wait:
                        self._waiting.set()
                    else:
                        await self._handle_battle_request(battle)
            elif split_message[1] == "win" or split_message[1] == "tie":
                await self._battle_count_queue.get()
                self._battle_count_queue.task_done()
                self._battle_finished_callback(battle)
                async with self._battle_end_condition:
                    self._battle_end_condition.notify_all()
                if hasattr(self.ps_client, "websocket"):
                    await self.ps_client.send_message(f"/leave {battle.battle_tag}")
            elif split_message[1] == "error":
                self.logger.log(
                    25, "Error message received: %s", "|".join(split_message)
                )
                if split_message[2].startswith(
                    "[Invalid choice] Sorry, too late to make a different move"
                ):
                    if battle.trapped:
                        self._trying_again.set()
                elif split_message[2].startswith(
                    "[Unavailable choice] Can't switch: The active Pokémon is "
                    "trapped"
                ) or split_message[2].startswith(
                    "[Invalid choice] Can't switch: The active Pokémon is trapped"
                ):
                    self._trying_again.set()
                elif split_message[2].startswith(
                    "[Invalid choice] Can't switch: You can't switch to an active "
                    "Pokémon"
                ):
                    await self._handle_battle_request(battle, maybe_default_order=True)
                elif split_message[2].startswith(
                    "[Invalid choice] Can't switch: You can't switch to a fainted "
                    "Pokémon"
                ):
                    await self._handle_battle_request(battle, maybe_default_order=True)
                elif split_message[2].startswith(
                    "[Invalid choice] Can't switch: You do not have a Pokémon named"
                ):
                    await self._handle_battle_request(battle, maybe_default_order=True)
                elif split_message[2].startswith(
                    "[Invalid choice] Can't switch: You have to pass to a fainted"
                ):
                    await self._handle_battle_request(battle, maybe_default_order=True)
                elif split_message[2].startswith(
                    "[Invalid choice] Can't move: Invalid target for"
                ):
                    await self._handle_battle_request(battle, maybe_default_order=True)
                elif split_message[2].startswith(
                    "[Invalid choice] Can't move: You can't choose a target for"
                ):
                    await self._handle_battle_request(battle, maybe_default_order=True)
                elif split_message[2].startswith(
                    "[Invalid choice] Can't move: "
                ) and split_message[2].endswith("needs a target"):
                    await self._handle_battle_request(battle, maybe_default_order=True)
                elif (
                    split_message[2].startswith("[Invalid choice] Can't move: Your")
                    and " doesn't have a move matching " in split_message[2]
                ):
                    await self._handle_battle_request(battle, maybe_default_order=True)
                elif split_message[2].startswith(
                    "[Invalid choice] Incomplete choice: "
                ):
                    await self._handle_battle_request(battle, maybe_default_order=True)
                elif split_message[2].startswith(
                    "[Unavailable choice]"
                ) and split_message[2].endswith("is disabled"):
                    battle.move_on_next_request = True
                elif split_message[2].startswith("[Invalid choice]") and split_message[
                    2
                ].endswith("is disabled"):
                    battle.move_on_next_request = True
                elif split_message[2].startswith(
                    "[Invalid choice] Can't move: You sent more choices than unfainted"
                    " Pokémon."
                ):
                    await self._handle_battle_request(battle, maybe_default_order=True)
                elif split_message[2].startswith(
                    "[Invalid choice] Can't move: You can only Terastallize once per battle."
                ):
                    await self._handle_battle_request(battle, maybe_default_order=True)
                elif split_message[2].startswith(
                    "[Invalid choice] Can't move: You need a switch response"
                ):
                    await self._handle_battle_request(battle, maybe_default_order=True)
                else:
                    self.logger.warning(
                        "Unexpected error message (falling back to default order): %s",
                        split_message,
                    )
                    await self._handle_battle_request(battle, maybe_default_order=True)
            elif split_message[1] == "turn":
                # cut the turnlist to save memory
                battle._mm_battle.turnlist = battle._mm_battle.turnlist[-2:]
            elif split_message[1] == "bigerror":
                self.logger.warning("Received 'bigerror' message: %s", split_message)
            elif split_message[1] == "uhtml" and split_message[2] == "otsrequest":
                await self._handle_ots_request(battle.battle_tag)

    def teampreview(self, battle: AbstractBattle) -> str:
        """
        Returns a teampreview order for the given battle.

        If a team-preview strategy is provided, uses it to predict the best lead;
        the remaining slots are genuinely random-shuffled (matches poke-env's
        ``random_teampreview``, keeps the back-order in-distribution). Otherwise,
        falls back entirely to random selection.

        Args:
            battle: The battle in team preview

        Returns:
            Team order string in format "/team 3461..." where first pokemon is the lead
        """
        if self.team_preview_strategy is None:
            # fallback to random if no strategy provided
            return self.random_teampreview(battle)

        # Map Showdown variants (e.g. gen9oulongtimer) to the format the strategy knows.
        agent_format = format_for_agent(self._format.replace("-", "").lower())
        if agent_format not in self.team_preview_strategy.trained_formats:
            self.logger.warning(
                f"Battle format {self._format} (agent: {agent_format}) not in trained formats "
                f"{self.team_preview_strategy.trained_formats}. "
                f"Falling back to random."
            )
            return self.random_teampreview(battle)

        team_list = list(battle.team.values())
        opponent_list = list(battle.opponent_team.values())
        if len(team_list) != 6 or len(opponent_list) != 6:
            self.logger.warning(
                f"Invalid team sizes: our={len(team_list)}, opponent={len(opponent_list)}. "
                f"Falling back to random."
            )
            return self.random_teampreview(battle)

        # build team preview input
        our_team_names = [pokemon_name(p.species) for p in team_list]
        our_team_moves = [
            [move_name(m.id) for m in p.moves.values()] if p.moves else []
            for p in team_list
        ]
        our_team_abilities = [
            UniversalPokemon.universal_abilities(p.ability) for p in team_list
        ]
        our_team_items = [UniversalPokemon.universal_items(p.item) for p in team_list]
        opponent_team_names = [pokemon_name(p.species) for p in opponent_list]

        # team preview inference
        predicted_lead_name, probs, sorted_team = (
            self.team_preview_strategy.predict_lead(
                our_team=our_team_names,
                our_team_moves=our_team_moves,
                our_team_abilities=our_team_abilities,
                our_team_items=our_team_items,
                opponent_team=opponent_team_names,
                battle_format=agent_format,
            )
        )

        # format team preview prediction output to showdown command
        lead_position = None
        for i, pokemon in enumerate(team_list):
            if pokemon_name(pokemon.species) == predicted_lead_name:
                lead_position = i + 1  # 1-indexed
                break
        if lead_position is None:
            self.logger.warning(
                f"Could not find predicted lead {predicted_lead_name} in team, falling back to random"
            )
            return self.random_teampreview(battle)

        # lead first, remaining slots genuinely shuffled (in-distribution back-order)
        max_team_size = getattr(battle, "max_team_size", None) or len(team_list)
        positions = order_from_lead(
            lead_position, len(team_list), max_team_size, rng=random
        )
        team_order = build_team_order(positions, slash=True)

        # Log team preview with clear sorted order -> scores -> selection mapping
        probs_np = (
            probs.detach().cpu().numpy()
            if hasattr(probs, "detach")
            else np.asarray(probs)
        )
        candidates = ", ".join(
            f"{name}={score:.2f}" for name, score in zip(sorted_team, probs_np)
        )
        self.logger.warning(
            f"Team preview: [{candidates}] -> selected {predicted_lead_name}"
        )

        return team_order

    @staticmethod
    def choose_random_move(battle: MetamonBackendBattle):
        # default version demands built-in Battle/DoubleBattle types
        return Player.choose_random_singles_move(battle)


class PokeAgentPlayer(MetamonPlayer):

    def create_metamon_battle(self, battle_tag: str) -> PokeAgentBackendBattle:
        return PokeAgentBackendBattle(
            battle_tag=battle_tag,
            username=self.username,
            logger=self.logger,
            save_replays=self._save_replays,
            gen=Dex.from_format(self.format).gen,
        )
