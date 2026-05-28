import time
from argparse import ArgumentParser

from metamon.baselines.heuristic.basic import GymLeader, RandomBaseline
from metamon.baselines.model_based.bcrnn_baselines import BaseRNN
from metamon.interface import (
    get_observation_space,
    TokenizedObservationSpace,
    DefaultActionSpace,
    DefaultShapedReward,
    UniversalPokemon,
)
from metamon.tokenizer import get_tokenizer
from metamon.env.wrappers import get_metamon_teams, BattleAgainstBaseline

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--battle_format", type=str, default="gen1ou")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--team_set", type=str, default="competitive")
    parser.add_argument(
        "--observation_space", type=str, default="DefaultObservationSpace"
    )
    parser.add_argument(
        "--battle_backend",
        type=str,
        default="poke-env",
        choices=["poke-env", "metamon", "pokeagent"],
    )
    parser.add_argument(
        "--print_stats",
        action="store_true",
        help="After each episode reset, print computed stat fields from UniversalState "
        "(player active, player switches, opponent active).",
    )
    args = parser.parse_args()

    def _format_stats(mon: UniversalPokemon) -> str:
        return (
            f"hp={mon.hp_stat} atk={mon.atk_stat} def={mon.def_stat} "
            f"spa={mon.spa_stat} spd={mon.spd_stat} spe={mon.spe_stat}"
        )

    env = BattleAgainstBaseline(
        battle_format=args.battle_format,
        team_set=get_metamon_teams(args.battle_format, args.team_set),
        opponent_type=GymLeader,
        observation_space=get_observation_space(args.observation_space),
        action_space=DefaultActionSpace(),
        reward_function=DefaultShapedReward(),
        battle_backend=args.battle_backend,
    )

    start = time.time()
    counter = 0
    for ep in range(args.episodes):
        print(f"Episode {ep}")
        inner_start = time.time()
        state, info = env.reset()
        if args.print_stats:
            team_file = env.metamon_team_set.most_recent_team_file
            print(f"  team file: {team_file}")
            us = env._most_recent_state
            print(
                f"  player active ({us.player_active_pokemon.name}): "
                f"{_format_stats(us.player_active_pokemon)}"
            )
            for sw in us.available_switches:
                print(f"  player switch ({sw.name}): {_format_stats(sw)}")
            print(
                f"  opponent active ({us.opponent_active_pokemon.name}): "
                f"{_format_stats(us.opponent_active_pokemon)}"
            )
        done = False
        return_ = 0.0
        timesteps = 0
        while not done:
            env.render()
            obs, reward, terminated, truncated, info = env.step(
                env.action_space.sample()
            )
            return_ += reward
            done = terminated or truncated
            timesteps += 1
            counter += 1
        print(
            f"Episode {ep}:: Timesteps: {timesteps}, Total Return: {return_ : .2f}, FPS: {timesteps / (time.time() - inner_start) : .2f}, Invalid Action: {info['invalid_action_count']}, Valid Actions: {info['valid_action_count']}"
        )

    end = time.time()
    print(f"{counter / (end - start) : .2f} Steps Per Second")
