import copy
import os

import gymnasium as gym
import numpy as np
import pytest
import torch

from metamon.interface import (
    DefaultObservationSpace,
    Gen1PokemonSlotObservationSpace,
    TokenizedObservationSpace,
    UniversalMove,
    UniversalPokemon,
    UniversalState,
)
from metamon.tokenizer import get_tokenizer


def make_move(name="tackle", base_power=40, current_pp=35, max_pp=35):
    return UniversalMove(
        name=name,
        move_type="normal",
        category="physical",
        base_power=base_power,
        accuracy=1.0,
        priority=0,
        current_pp=current_pp,
        max_pp=max_pp,
    )


def make_pokemon(name, status="nostatus", moves=None):
    return UniversalPokemon(
        name=name,
        base_species=name,
        hp_pct=1.0,
        types="normal notype",
        item="noitem",
        ability="noability",
        lvl=100,
        status=status,
        effect="noeffect",
        moves=moves if moves is not None else [make_move()],
        atk_boost=0,
        spa_boost=0,
        def_boost=0,
        spd_boost=0,
        spe_boost=0,
        accuracy_boost=0,
        evasion_boost=0,
        base_atk=100,
        base_spa=100,
        base_def=100,
        base_spd=100,
        base_spe=100,
        base_hp=100,
        tera_type="notype",
    )


def make_state(opponent_name="mewtwo", opponent_status="nostatus", switches=None):
    return UniversalState(
        format="gen1ou",
        player_active_pokemon=make_pokemon(
            "alakazam",
            moves=[
                make_move("psychic", 90),
                make_move("recover", 0),
                make_move("seismictoss", 0),
                make_move("thunderwave", 0),
            ],
        ),
        opponent_active_pokemon=make_pokemon(
            opponent_name,
            status=opponent_status,
            moves=[make_move("bodyslam", 85)],
        ),
        available_switches=switches
        if switches is not None
        else [make_pokemon("chansey"), make_pokemon("tauros")],
        player_prev_move=make_move("psychic", 90),
        opponent_prev_move=make_move("bodyslam", 85),
        opponents_remaining=6,
        player_conditions="noconditions",
        opponent_conditions="noconditions",
        weather="noweather",
        battle_field="nofield",
        forced_switch=False,
        battle_won=False,
        battle_lost=False,
        can_tera=False,
        opponent_teampreview=[],
    )


def test_gen1_pokemon_slot_observation_shapes_and_padding():
    obs_space = Gen1PokemonSlotObservationSpace()
    obs = obs_space.state_to_obs(make_state())

    assert set(obs) == {
        "pokemon_text",
        "pokemon_numbers",
        "global_text",
        "global_numbers",
    }
    assert obs["pokemon_numbers"].shape == (13 * 31,)
    assert obs["global_numbers"].shape == (3,)
    assert len(obs["pokemon_text"].item().split(" ")) == 13 * 9
    assert len(obs["global_text"].item().split(" ")) == 6

    pokemon_words = obs["pokemon_text"].item().split(" ")
    padded_switch_slot = pokemon_words[3 * 9 : 4 * 9]
    assert padded_switch_slot[:5] == ["<blank>"] * 5
    assert padded_switch_slot[5:] == ["nomove"] * 4


def test_gen1_pokemon_slot_revealed_memory_and_reset():
    obs_space = Gen1PokemonSlotObservationSpace()
    first = obs_space.state_to_obs(make_state(opponent_name="mewtwo"))
    second = obs_space.state_to_obs(make_state(opponent_name="snorlax"))

    first_words = first["pokemon_text"].item().split(" ")
    second_words = second["pokemon_text"].item().split(" ")
    first_revealed_slot = first_words[7 * 9 : 8 * 9]
    second_revealed_slots = second_words[7 * 9 : 9 * 9]
    assert first_revealed_slot[0] == "mewtwo"
    assert second_revealed_slots[0] == "mewtwo"
    assert second_revealed_slots[9] == "snorlax"

    obs_space.reset()
    reset_obs = obs_space.state_to_obs(make_state(opponent_name="snorlax"))
    reset_words = reset_obs["pokemon_text"].item().split(" ")
    assert reset_words[7 * 9] == "snorlax"
    assert reset_words[8 * 9] == "<blank>"


def test_gen1_pokemon_slot_tokenization_and_default_shape_unchanged():
    tokenizer = get_tokenizer("DefaultObservationSpace-v1")
    base = Gen1PokemonSlotObservationSpace()
    tokenized = TokenizedObservationSpace(base, tokenizer)
    obs = tokenized.state_to_obs(make_state())

    assert obs["pokemon_text_tokens"].shape == (13 * 9,)
    assert obs["global_text_tokens"].shape == (6,)
    assert obs["pokemon_numbers"].shape == (13 * 31,)
    assert obs["global_numbers"].shape == (3,)
    assert DefaultObservationSpace().tokenizable == {"text": 87}


def test_gen1_pokemon_slot_encoder_forward():
    pytest.importorskip("amago")
    os.environ.setdefault("METAMON_CACHE_DIR", "/tmp/metamon-cache")
    from metamon.rl.metamon_to_amago import MetamonPokemonSlotTstepEncoder

    tokenizer = get_tokenizer("DefaultObservationSpace-v1")
    tokenized = TokenizedObservationSpace(Gen1PokemonSlotObservationSpace(), tokenizer)
    encoder = MetamonPokemonSlotTstepEncoder(
        obs_space=tokenized.gym_space,
        rl2_space=gym.spaces.Box(low=-10.0, high=10.0, shape=(4,), dtype=np.float32),
        tokenizer=tokenizer,
        extra_emb_dim=3,
        d_model=16,
        n_heads=4,
        slot_layers=1,
        team_layers=1,
        slot_latent_tokens=2,
        team_latent_tokens=2,
        pokemon_numerical_tokens=2,
        global_numerical_tokens=1,
        token_mask_aug=False,
        dropout=0.0,
        max_pokemon_tokens=16,
        max_team_tokens=24,
    )
    encoder.eval()

    single_obs = tokenized.state_to_obs(make_state())
    obs = {
        key: torch.as_tensor(value).view(1, 1, *value.shape)
        for key, value in single_obs.items()
    }
    rl2s = torch.zeros(1, 1, 4)
    out = encoder.inner_forward(copy.deepcopy(obs), rl2s)

    assert out.shape == (1, 1, encoder.emb_dim)
    assert torch.isfinite(out).all()
