"""Playable smoke: arbitrary battles via metamon adapters for all target gens."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

POKEPY_ENGINE = Path(__file__).resolve().parents[1] / "pokepy-engine"
if POKEPY_ENGINE.exists():
    os.environ.setdefault(
        "POKEPY_DATA_PATH", str(POKEPY_ENGINE / "pokepy" / "data" / "extracted")
    )

from metamon.env.pokepy_battle.action_adapter import build_illegal_actions_mask
from metamon.env.pokepy_battle.state_adapter import pokepy_state_to_universal
from metamon.interface import DefaultActionSpace
from pokepy.core.gen_profile import registered_gens
from pokepy.data.loader import load_game_data, load_id_mappings, load_move_effect_data
from pokepy.data.type_charts import load_type_chart_for_gen
from pokepy.engine import step_battle
from pokepy.engine.action_mask import get_action_mask
from pokepy.env.battle_env import DEFAULT_TEAM, init_battle_state
from pokepy.utils.gen5_prng import Gen5PRNG


@pytest.mark.parametrize("gen", sorted(registered_gens()))
def test_battle_plays_with_universal_state_and_action_mask(gen):
    gd = load_game_data(gen=gen)
    mappings = load_id_mappings(gen=gen)
    me = load_move_effect_data(gen=gen)
    chart = load_type_chart_for_gen(gen)
    state = init_battle_state(DEFAULT_TEAM, DEFAULT_TEAM, gd, seed=42, gen=gen)
    prng = Gen5PRNG((42 & 0xFFFF, (42 >> 16) & 0xFFFF, 0, 0))
    fmt = f"gen{gen}ou" if gen != 9 else "gen9ou"
    action_space = DefaultActionSpace()
    done = False
    steps = 0
    while not done and steps < 200:
        us = pokepy_state_to_universal(
            state, gd, mappings, format_str=fmt, player_side=0
        )
        assert us.player_active_pokemon is not None
        assert us.player_active_pokemon.hp_pct >= 0.0
        mask = get_action_mask(state, 0, gd)
        illegal = build_illegal_actions_mask(
            action_space, us, state, gd, mappings, player_side=0
        )
        legal = [i for i in range(len(illegal)) if not illegal[i]]
        assert legal, "no legal actions"
        from metamon.env.pokepy_battle.action_adapter import universal_action_to_pokepy

        ua = next(
            a
            for a in __import__(
                "metamon.interface", fromlist=["UniversalAction"]
            ).UniversalAction.maybe_valid_actions(us)
            if int(a.action_idx) in legal
        )
        a0, _wtera = universal_action_to_pokepy(ua, us, state, mappings, player_side=0)
        m1 = get_action_mask(state, 1, gd)
        a1 = next(i for i, v in enumerate(m1) if v)
        r0, r1, done = step_battle(gen, state, a0, a1, gd, me, chart, prng)
        assert np.isfinite(r0) and np.isfinite(r1)
        steps += 1
    assert steps > 0
