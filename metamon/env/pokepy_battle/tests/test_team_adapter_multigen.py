"""Profile-aware team conversion for gen1/2 (no abilities/items)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

POKEPY_ENGINE = Path(__file__).resolve().parents[1] / "pokepy-engine"
if POKEPY_ENGINE.exists():
    os.environ.setdefault(
        "POKEPY_DATA_PATH", str(POKEPY_ENGINE / "pokepy" / "data" / "extracted")
    )


def _require_pokepy_data():
    from pokepy.data.loader import get_data_path

    if not (get_data_path() / "type_chart.npy").exists():
        pytest.skip("pokepy extracted data missing")


def _packed_mon(
    *,
    species: str = "kakuna",
    item: str = "",
    ability: str = "noability",
    moves: str = "poisonsting,tackle,tackle,tackle",
) -> str:
    return (
        f"Kakuna|{species}|{item}|{ability}|{moves}|Bold|"
        "84,84,84,84,84,84|31,31,31,31,31,31|||100]"
    )


def test_gen2_noability_does_not_lookup_ability():
    _require_pokepy_data()
    from pokepy.core.gen_profile import GEN2_PROFILE
    from pokepy.data.loader import load_game_data, load_id_mappings
    from pokepy.env.battle_env import init_battle_state
    from metamon.env.pokepy_battle.team_adapter import showdown_team_to_pokepy_dict

    mappings = load_id_mappings(gen=2)
    packed = _packed_mon(item="miracleseed")
    team = showdown_team_to_pokepy_dict(
        packed, mappings=mappings, profile=GEN2_PROFILE
    )
    assert all(a == 0 for a in team["abilities"])
    assert any(i > 0 for i in team["items"])
    assert all(t == -1 for t in team["tera_types"])

    gd = load_game_data(gen=2)
    state = init_battle_state(team, team, gd, seed=1, gen=2)
    assert int(state.battle_state[5]) == 0


def test_gen1_noitem_noability():
    _require_pokepy_data()
    from pokepy.core.gen_profile import GEN1_PROFILE
    from pokepy.data.loader import load_game_data, load_id_mappings
    from pokepy.env.battle_env import init_battle_state
    from metamon.env.pokepy_battle.team_adapter import showdown_team_to_pokepy_dict

    mappings = load_id_mappings(gen=1)
    packed = _packed_mon(item="noitem", ability="noability")
    team = showdown_team_to_pokepy_dict(
        packed, mappings=mappings, profile=GEN1_PROFILE
    )
    assert all(a == 0 for a in team["abilities"])
    assert all(i == 0 for i in team["items"])

    gd = load_game_data(gen=1)
    state = init_battle_state(team, team, gd, seed=1, gen=1)
    assert int(state.battle_state[5]) == 0
    assert int(state.battle_state[6]) == 0


def test_gen2_berserkgene_maps():
    _require_pokepy_data()
    from pokepy.core.gen_profile import GEN2_PROFILE
    from pokepy.data.loader import load_id_mappings
    from metamon.env.pokepy_battle.team_adapter import showdown_team_to_pokepy_dict

    mappings = load_id_mappings(gen=2)
    assert "berserkgene" in mappings.item_to_idx
    packed = _packed_mon(item="berserkgene")
    team = showdown_team_to_pokepy_dict(
        packed, mappings=mappings, profile=GEN2_PROFILE
    )
    assert team["items"][0] == mappings.item_to_idx["berserkgene"]


def test_gen9_still_maps_real_ability():
    _require_pokepy_data()
    from pokepy.core.gen_profile import GEN9_PROFILE
    from pokepy.data.loader import load_id_mappings
    from metamon.env.pokepy_battle.team_adapter import showdown_team_to_pokepy_dict

    mappings = load_id_mappings(gen=9)
    packed = _packed_mon(item="", ability="swarm")
    team = showdown_team_to_pokepy_dict(
        packed, mappings=mappings, profile=GEN9_PROFILE
    )
    assert team["abilities"][0] > 0
