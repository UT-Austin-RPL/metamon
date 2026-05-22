import json
import os
import warnings
from functools import partial
from typing import Optional

import amago

from metamon.env import get_metamon_teams, QueueOnLocalLadder, TeamSet
from metamon.interface import ObservationSpace, RewardFunction, ActionSpace
from metamon.rl.pretrained import get_pretrained_model
from metamon.rl.metamon_to_amago import PSLadderAMAGOWrapper
from metamon.rl.evaluate.common import load_config, merge_defaults, random_choice

warnings.filterwarnings("ignore")


def make_ladder_env(
    battle_format: str,
    player_team_set: TeamSet,
    observation_space: ObservationSpace,
    action_space: ActionSpace,
    reward_function: RewardFunction,
    num_battles: int,
    username: str,
    save_trajectories_to: Optional[str] = None,
    battle_backend: str = "metamon",
):
    """
    Battle on the local Showdown ladder
    """
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=amago.utils.AmagoWarning)
    menv = QueueOnLocalLadder(
        battle_format=battle_format,
        num_battles=num_battles,
        observation_space=observation_space,
        action_space=action_space,
        reward_function=reward_function,
        player_team_set=player_team_set,
        player_username=username,
        save_trajectories_to=save_trajectories_to,
        battle_backend=battle_backend,
        print_battle_bar=False,
    )
    return PSLadderAMAGOWrapper(menv)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--username",
        required=True,
        help="Choose a username from the config to evaluate.",
    )
    parser.add_argument(
        "--format",
        default="gen1ou",
        choices=["gen1ou", "gen2ou", "gen3ou", "gen4ou", "gen9ou"],
        help="Specify the battle format/tier.",
    )
    parser.add_argument(
        "--save_trajectories_to",
        type=str,
        default=None,
        help="Path to save trajectories to.",
    )
    parser.add_argument(
        "--n_challenges",
        type=int,
        default=10,
        help=(
            "Number of battles to run before returning eval stats. "
            "Note this is the total sample size across all parallel actors."
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML config file.",
    )
    args = parser.parse_args()

    # load config using shared utility
    raw_config = load_config(args.config)

    # validate structure
    if "agents" not in raw_config:
        raise ValueError("Config must have 'agents' section")

    defaults = raw_config.get("defaults", {})
    agents = raw_config.get("agents", {})

    # validate defaults
    required_defaults = [
        "team_set",
        "battle_backend",
        "checkpoints",
        "temperatures",
        "num_agents",
    ]
    missing_defaults = [field for field in required_defaults if field not in defaults]
    if missing_defaults:
        raise ValueError(
            f"defaults section missing required fields: {', '.join(missing_defaults)}"
        )

    # find base username (strip numeric suffix if from num_agents expansion)
    username = args.username
    if username in agents:
        base_username = username
    else:
        if "-" in username and username.split("-")[-1].isdigit():
            potential_base = "-".join(username.split("-")[:-1])
            if potential_base in agents:
                base_username = potential_base
            else:
                raise ValueError(
                    f"Username {username} not found in config and could not find base"
                )
        else:
            raise ValueError(f"Username {username} not found in config")

    # merge config using shared utility
    agent_config = agents[base_username]
    account_config = merge_defaults(defaults, agent_config or {})
    account_config["battle_format"] = args.format

    # validate required fields
    if "model_name" not in account_config:
        raise ValueError(f"Agent {base_username} missing required field: model_name")

    # load model and team set
    model_name = account_config["model_name"]
    agent_maker = get_pretrained_model(model_name)

    # team_set / checkpoint / temperature all support:
    #   plain scalar, plain list, "range(...)", "linspace(...)", {weighted: {...}}
    team_set_choice = random_choice(account_config["team_set"])
    print(f"Using team_set {team_set_choice}")
    player_team_set = get_metamon_teams(args.format, team_set_choice)

    checkpoint = random_choice(account_config["checkpoints"])
    print(f"Using checkpoint {checkpoint}")

    temperature = float(random_choice(account_config.get("temperatures", [1.0])))
    print(f"Using temperature {temperature}")
    battle_backend = account_config["battle_backend"]
    print(f"Using battle backend {battle_backend}")

    save_trajectories_to = os.path.join(
        args.save_trajectories_to, model_name, battle_backend
    )
    os.makedirs(save_trajectories_to, exist_ok=True)

    # initialize agent
    agent = agent_maker.initialize_agent(
        checkpoint=checkpoint, log=False, action_temperature=temperature
    )
    agent.env_mode = "sync"
    # create envs
    env_kwargs = dict(
        battle_format=args.format,
        player_team_set=player_team_set,
        observation_space=agent_maker.observation_space,
        action_space=agent_maker.action_space,
        reward_function=agent_maker.reward_function,
        save_trajectories_to=save_trajectories_to,
        battle_backend=battle_backend,
    )
    make_envs = [
        partial(
            make_ladder_env,
            **env_kwargs,
            num_battles=args.n_challenges,
            username=username,
        )
    ]
    agent.verbose = False
    agent.parallel_actors = len(make_envs)

    # evaluate
    results = agent.evaluate_test(
        make_envs,
        # sets upper bound on total timesteps
        timesteps=args.n_challenges * 350,
        # terminates after n_challenges
        episodes=args.n_challenges,
    )
    print(json.dumps(results, indent=4, sort_keys=True))
