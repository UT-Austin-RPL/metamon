"""
Worker script: runs one side of a head-to-head matchup.

Loads a pretrained model, creates a ChallengeByUsername environment, and
runs battles. This is launched as a subprocess by the h2h/sweep launchers.

Fully self-contained — all configuration is passed via CLI args.
"""

import json
import warnings
from argparse import ArgumentParser
from functools import partial
from typing import Optional

warnings.filterwarnings("ignore")


def main():
    parser = ArgumentParser(description="Run one side of a head-to-head matchup.")
    parser.add_argument("--model_name", required=True, help="Pretrained model name.")
    parser.add_argument("--username", required=True, help="This player's username.")
    parser.add_argument(
        "--opponent_username", required=True, help="Opponent's username."
    )
    parser.add_argument(
        "--role",
        required=True,
        choices=["challenger", "acceptor"],
        help="challenger sends challenges, acceptor waits for them.",
    )
    parser.add_argument("--format", required=True, help="Battle format (e.g. gen1ou).")
    parser.add_argument(
        "--n_battles", type=int, required=True, help="Number of battles."
    )
    parser.add_argument("--team_set", default="competitive", help="Team set name.")
    parser.add_argument("--battle_backend", default="metamon", help="Battle backend.")
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature."
    )
    parser.add_argument(
        "--checkpoint", type=int, default=None, help="Checkpoint epoch."
    )
    parser.add_argument(
        "--save_results_to", default=None, help="Directory for per-battle CSV logs."
    )
    parser.add_argument(
        "--save_trajectories_to", default=None, help="Directory for trajectory files."
    )
    args = parser.parse_args()

    import amago
    from metamon.env import get_metamon_teams, ChallengeByUsername
    from metamon.rl.pretrained import get_pretrained_model
    from metamon.rl.metamon_to_amago import make_challenge_env

    # Load model
    pretrained = get_pretrained_model(args.model_name)
    agent = pretrained.initialize_agent(
        checkpoint=args.checkpoint,
        log=False,
        action_temperature=args.temperature,
    )
    agent.env_mode = "sync"
    agent.parallel_actors = 1
    agent.verbose = False

    # Load teams
    player_team_set = get_metamon_teams(args.format, args.team_set)

    # Create env factory
    make_env = partial(
        make_challenge_env,
        battle_format=args.format,
        num_battles=args.n_battles,
        observation_space=pretrained.observation_space,
        action_space=pretrained.action_space,
        reward_function=pretrained.reward_function,
        player_team_set=player_team_set,
        player_username=args.username,
        opponent_username=args.opponent_username,
        role=args.role,
        battle_backend=args.battle_backend,
        save_results_to=args.save_results_to,
        save_trajectories_to=args.save_trajectories_to,
        print_battle_bar=False,
    )

    # Run battles
    results = agent.evaluate_test(
        [make_env],
        timesteps=args.n_battles * 1000,
        episodes=args.n_battles,
    )
    print(json.dumps(results, indent=4, sort_keys=True))


if __name__ == "__main__":
    main()
