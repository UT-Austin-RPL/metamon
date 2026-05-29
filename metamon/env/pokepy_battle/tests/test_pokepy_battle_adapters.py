"""Tests for pokepy_battle adapters."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

POKEPY_ENGINE = Path(__file__).resolve().parents[1] / "pokepy-engine"
if POKEPY_ENGINE.exists() and str(POKEPY_ENGINE) not in os.environ.get(
    "PYTHONPATH", ""
):
    os.environ["POKEPY_DATA_PATH"] = str(
        POKEPY_ENGINE / "pokepy" / "data" / "extracted"
    )


def _require_pokepy_data():
    from pokepy.data.loader import get_data_path

    data_path = get_data_path()
    if not (data_path / "type_chart.npy").exists():
        pytest.skip(f"pokepy extracted data missing at {data_path}")


@pytest.fixture(scope="module")
def pokepy_fixtures():
    _require_pokepy_data()
    from dataclasses import dataclass
    from pokepy.core.constants import TYPE_NORMAL
    from pokepy.data.loader import (
        load_game_data,
        load_id_mappings,
        load_move_effect_data,
    )
    from pokepy.env.battle_env import init_battle_state
    from pokepy.utils.gen5_prng import Gen5PRNG

    @dataclass
    class MonSpec:
        species: str
        moves: list
        item: str = ""
        ability: str = ""
        tera_type: int = TYPE_NORMAL
        level: int = 100

    gd = load_game_data()
    mappings = load_id_mappings()

    def _make_team(specs):
        full = list(specs)
        while len(full) < 6:
            full.append(specs[-1])
        species, moves, items, abilities, teras, levels = [], [], [], [], [], []
        for s in full[:6]:
            sid = mappings.species_to_idx[s.species]
            mids = [mappings.move_to_idx[m] for m in s.moves[:4]]
            while len(mids) < 4:
                mids.append(-1)
            species.append(sid)
            moves.append(mids[:4])
            items.append(mappings.item_to_idx.get(s.item, 0) if s.item else 0)
            abilities.append(
                mappings.ability_to_idx.get(s.ability, 0) if s.ability else 0
            )
            teras.append(int(s.tera_type))
            levels.append(int(s.level))
        return dict(
            species=species,
            moves=moves,
            items=items,
            abilities=abilities,
            tera_types=teras,
            levels=levels,
        )

    def fresh(team0_specs, team1_specs, seed=12345):
        state = init_battle_state(
            _make_team(team0_specs), _make_team(team1_specs), gd, seed=seed
        )
        prng = Gen5PRNG((seed & 0xFFFF, (seed >> 16) & 0xFFFF, 0, 0))
        return state, prng

    return gd, mappings, fresh, MonSpec


def test_universal_state_has_computed_stats(pokepy_fixtures):
    from metamon.env.pokepy_battle.state_adapter import pokepy_state_to_universal

    gd, mappings, fresh, MonSpec = pokepy_fixtures
    state, _ = fresh(
        [MonSpec("garchomp", ["earthquake", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle"] * 4)],
    )
    us = pokepy_state_to_universal(
        state, gd, mappings, format_str="gen9ou", player_side=0
    )
    assert us.player_active_pokemon.atk_stat > 0
    assert us.player_active_pokemon.hp_stat > 0


def test_action_mask_matches_pokepy_moves(pokepy_fixtures):
    from metamon.interface import DefaultActionSpace
    from metamon.env.pokepy_battle.action_adapter import build_illegal_actions_mask
    from metamon.env.pokepy_battle.state_adapter import pokepy_state_to_universal
    from pokepy.engine.action_mask import get_action_mask

    gd, mappings, fresh, MonSpec = pokepy_fixtures
    state, _ = fresh(
        [MonSpec("garchomp", ["earthquake", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle"] * 4)],
    )
    us = pokepy_state_to_universal(
        state, gd, mappings, format_str="gen9ou", player_side=0
    )
    action_space = DefaultActionSpace()
    illegal = build_illegal_actions_mask(
        action_space, us, state, gd, mappings, player_side=0
    )
    pokepy_mask = get_action_mask(state, 0, gd)
    for move_slot in range(4):
        if pokepy_mask[move_slot]:
            assert not illegal[move_slot]


def test_legal_actions_follow_maybe_valid_actions(pokepy_fixtures):
    from metamon.interface import DefaultActionSpace, UniversalAction
    from metamon.env.pokepy_battle.action_adapter import legal_action_indices
    from metamon.env.pokepy_battle.state_adapter import pokepy_state_to_universal

    gd, mappings, fresh, MonSpec = pokepy_fixtures
    state, _ = fresh(
        [MonSpec("garchomp", ["earthquake", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle"] * 4)],
    )
    us = pokepy_state_to_universal(
        state, gd, mappings, format_str="gen9ou", player_side=0
    )
    action_space = DefaultActionSpace()
    legal = set(
        legal_action_indices(action_space, us, state, gd, mappings, player_side=0)
    )
    maybe = {
        a.action_idx
        for a in UniversalAction.maybe_valid_actions(us)
        if 0 <= a.action_idx < action_space.gym_space.n
    }
    assert legal <= maybe
    assert legal, "expected at least one legal move at battle start"


def test_obs_interchangeability_with_pokepy_kakuna(pokepy_fixtures):
    from metamon.interface import DefaultObservationSpace
    from metamon.env.pokepy_battle.state_adapter import pokepy_state_to_universal
    from pokepy.obs.kakuna_obs import build_kakuna_obs
    from pokepy.obs.state_to_universal import state_to_universal_state as pokepy_us

    gd, mappings, fresh, MonSpec = pokepy_fixtures
    state, _ = fresh(
        [MonSpec("kakuna", ["poisonsting", "tackle", "tackle", "tackle"])],
        [MonSpec("kakuna", ["poisonsting", "tackle", "tackle", "tackle"])],
    )
    metamon_us = pokepy_state_to_universal(
        state, gd, mappings, format_str="gen9ou", player_side=0
    )
    pokepy_universal = pokepy_us(
        state, gd, mappings, format_str="gen9ou", player_side=0
    )
    obs_space = DefaultObservationSpace()
    metamon_obs = obs_space.state_to_obs(metamon_us)
    pokepy_obs = build_kakuna_obs(pokepy_universal)
    for key in ("numbers", "text"):
        np.testing.assert_array_equal(metamon_obs[key], pokepy_obs[key])


def test_forced_switch_flag_is_side_aware(pokepy_fixtures):
    from metamon.env.pokepy_battle.state_adapter import pokepy_state_to_universal
    from pokepy.core.constants import PHASE_FORCED_SWITCH

    gd, mappings, fresh, MonSpec = pokepy_fixtures
    state, _ = fresh(
        [MonSpec("garchomp", ["earthquake", "tackle", "tackle", "tackle"])],
        [MonSpec("snorlax", ["tackle"] * 4)],
    )
    state.phase = PHASE_FORCED_SWITCH
    state.forced_switch_side = 1

    u0 = pokepy_state_to_universal(
        state, gd, mappings, format_str="gen9ou", player_side=0
    )
    u1 = pokepy_state_to_universal(
        state, gd, mappings, format_str="gen9ou", player_side=1
    )
    assert not u0.forced_switch
    assert u1.forced_switch

    state.forced_switch_side = 2
    u0 = pokepy_state_to_universal(
        state, gd, mappings, format_str="gen9ou", player_side=0
    )
    u1 = pokepy_state_to_universal(
        state, gd, mappings, format_str="gen9ou", player_side=1
    )
    assert u0.forced_switch
    assert u1.forced_switch
