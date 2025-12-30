import json
import collections
import functools
from typing import Optional, Dict, Any, Callable, List

import metamon
from metamon.rl.pretrained import (
    get_pretrained_model,
    get_pretrained_model_names,
    PretrainedModel,
)
from metamon.baselines import get_baseline
from metamon.rl.metamon_to_amago import (
    make_baseline_env,
    make_local_ladder_env,
    make_pokeagent_ladder_env,
)


HEURISTIC_COMPOSITE_BASELINES = [
    "PokeEnvHeuristic",
    "Gen1BossAI",
    "Grunt",
    "GymLeader",
    "EmeraldKaizo",
    "RandomBaseline",
]


def pretrained_vs_baselines(
    pretrained_model: PretrainedModel,
    battle_format: str,
    team_set: metamon.env.TeamSet,
    checkpoint: Optional[int] = None,
    total_battles: int = 250,
    parallel_actors_per_baseline: int = 5,
    async_mp_context: str = "forkserver",
    battle_backend: str = "poke-env",
    log_to_wandb: bool = False,
    save_trajectories_to: Optional[str] = None,
    save_team_results_to: Optional[str] = None,
    baselines: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Evaluate a pretrained model against built-in baseline opponents.

    Defaults to the 6 baselines that the paper calls the "Heuristic Composite Score",
    but you can specify a list of any of the available baselines (see metamon.baselines.get_all_baseline_names()).
    """
    agent = pretrained_model.initialize_agent(checkpoint=checkpoint, log=log_to_wandb)
    baselines = baselines or HEURISTIC_COMPOSITE_BASELINES
    agent.async_env_mp_context = async_mp_context
    # create envs that match the agent's observation/actions/rewards
    make_envs = [
        functools.partial(
            make_baseline_env,
            battle_format=battle_format,
            observation_space=pretrained_model.observation_space,
            action_space=pretrained_model.action_space,
            reward_function=pretrained_model.reward_function,
            save_trajectories_to=save_trajectories_to,
            save_team_results_to=save_team_results_to,
            battle_backend=battle_backend,
            team_set=team_set,
            opponent_type=get_baseline(opponent),
        )
        for opponent in baselines
    ]
    # amago will play `parallel_actors_per_baseline` copies of each baseline
    # in parallel and aggregate the results by baseline name.
    make_envs *= parallel_actors_per_baseline
    # evaluate
    agent.parallel_actors = len(make_envs)
    results = agent.evaluate_test(
        make_envs,
        timesteps=total_battles * 250 // len(make_envs),
        episodes=total_battles,
    )
    return results


def _pretrained_on_ladder(
    pretrained_model: PretrainedModel,
    make_ladder: Callable,
    total_battles: int,
    checkpoint: Optional[int],
    log_to_wandb: bool,
    **ladder_kwargs,
) -> Dict[str, Any]:
    """Helper function for ladder-based evaluation."""
    agent = pretrained_model.initialize_agent(checkpoint=checkpoint, log=log_to_wandb)
    agent.env_mode = "sync"
    agent.parallel_actors = 1
    agent.verbose = False  # turn off tqdm progress bar and print poke-env battle status

    make_env = functools.partial(
        make_ladder,
        observation_space=pretrained_model.observation_space,
        action_space=pretrained_model.action_space,
        reward_function=pretrained_model.reward_function,
        num_battles=total_battles,
        **ladder_kwargs,
    )

    results = agent.evaluate_test(
        [make_env],
        timesteps=total_battles * 1000,
        episodes=total_battles,
    )
    return results


def pretrained_vs_local_ladder(
    pretrained_model: PretrainedModel,
    username: str,
    battle_format: str,
    team_set: metamon.env.TeamSet,
    total_battles: int,
    avatar: Optional[str] = None,
    checkpoint: Optional[int] = None,
    battle_backend: str = "poke-env",
    save_trajectories_to: Optional[str] = None,
    save_team_results_to: Optional[str] = None,
    log_to_wandb: bool = False,
) -> Dict[str, Any]:
    """Evaluate a pretrained model on the ladder of your Local Showdown server.

    Make sure you've started your local server in the background with
    `node pokemon-showdown start --no-security`. Usernames must be unique,
    but do not need to be registered in advance, and do not require a password.

    Will automatically queue the agent for battles against any other agents (or humans)
    that are also online. This is the simplest way to evaluate pretrained models head-to-head
    and generate self-play data. It is also how the paper handled evals against third-party
    baselines like PokéLLMon.
    """

    return _pretrained_on_ladder(
        pretrained_model=pretrained_model,
        make_ladder=make_local_ladder_env,
        total_battles=total_battles,
        checkpoint=checkpoint,
        log_to_wandb=log_to_wandb,
        player_username=username,
        player_avatar=avatar,
        player_team_set=team_set,
        battle_backend=battle_backend,
        battle_format=battle_format,
        save_trajectories_to=save_trajectories_to,
        save_team_results_to=save_team_results_to,
    )


def pretrained_vs_pokeagent_ladder(
    pretrained_model: PretrainedModel,
    username: str,
    password: str,
    battle_format: str,
    team_set: metamon.env.TeamSet,
    total_battles: int,
    avatar: Optional[str] = None,
    checkpoint: Optional[int] = None,
    battle_backend: str = "poke-env",
    save_trajectories_to: Optional[str] = None,
    save_team_results_to: Optional[str] = None,
    log_to_wandb: bool = False,
) -> Dict[str, Any]:
    """Evaluate a pretrained model on the PokéAgent Challenge ladder.

    Must provide a registered username and password. See instructions in the README!

    Will automatically queue the agent for ranked battles against any other agents (or humans)
    that are logged into the PokéAgent Challenge ladder.

    Once eval begins, you can watch battles in real time by visiting
    http://pokeagentshowdown.com.insecure.psim.us and clicking "Watch a Battle".
    Visit http://pokeagentshowdown.com.insecure.psim.us/ladder to see the live
    leaderboard.
    """
    return _pretrained_on_ladder(
        pretrained_model=pretrained_model,
        make_ladder=make_pokeagent_ladder_env,
        total_battles=total_battles,
        checkpoint=checkpoint,
        log_to_wandb=log_to_wandb,
        player_username=username,
        player_password=password,
        player_avatar=avatar,
        player_team_set=team_set,
        battle_backend=battle_backend,
        battle_format=battle_format,
        save_trajectories_to=save_trajectories_to,
        save_team_results_to=save_team_results_to,
    )


def _get_default_eval(args, base_eval_kwargs):
    """Get the appropriate evaluation helper and update required args based on eval_type."""
    if args.eval_type == "heuristic":
        base_eval_kwargs.update(
            {
                "baselines": HEURISTIC_COMPOSITE_BASELINES,
                "async_mp_context": args.async_mp_context,
            }
        )
        return pretrained_vs_baselines
    elif args.eval_type == "il":
        base_eval_kwargs.update(
            {
                "baselines": ["BaseRNN"],
                "async_mp_context": args.async_mp_context,
                # sets this low to avoid overloading CPU with RNN baseline inference
                "parallel_actors_per_baseline": 1,
            }
        )
        return pretrained_vs_baselines
    elif args.eval_type == "ladder":
        base_eval_kwargs.update(
            {
                "username": args.username,
                "avatar": args.avatar,
            }
        )
        return pretrained_vs_local_ladder
    elif args.eval_type == "pokeagent":
        base_eval_kwargs.update(
            {
                "username": args.username,
                "password": args.password,
                "avatar": args.avatar,
            }
        )
        return pretrained_vs_pokeagent_ladder
    else:
        raise ValueError(f"Invalid evaluation type: {args.eval_type}")


def _run_default_evaluation(args) -> Dict[str, List[Dict[str, Any]]]:
    # Validate arguments
    if args.custom_checkpoint_path is None and args.agent is None:
        raise ValueError("Must provide either --agent or --custom_checkpoint_path")
    if args.custom_checkpoint_path is not None and args.agent is None:
        raise ValueError("When using --custom_checkpoint_path, must also specify --agent to indicate which base model config to use")

    pretrained_model = get_pretrained_model(args.agent)
    all_results = collections.defaultdict(list)
    for gen in args.gens:
        for format_name in args.formats:
            battle_format = f"gen{gen}{format_name.lower()}"
            team_set_type = (
                metamon.env.PokeAgentTeamSet
                if args.eval_type == "pokeagent"
                else metamon.env.TeamSet
            )
            player_team_set = metamon.env.get_metamon_teams(
                battle_format, args.team_set, set_type=team_set_type
            )
            for checkpoint in args.checkpoints:
                # If custom checkpoint path is provided, pass it as a path instead of checkpoint number
                if args.custom_checkpoint_path:
                    checkpoint_to_use = args.custom_checkpoint_path
                else:
                    checkpoint_to_use = checkpoint

                eval_kwargs = {
                    "pretrained_model": pretrained_model,
                    "battle_format": battle_format,
                    "team_set": player_team_set,
                    "total_battles": args.total_battles,
                    "checkpoint": checkpoint_to_use,
                    "battle_backend": args.battle_backend,
                    "save_trajectories_to": args.save_trajectories_to,
                    "save_team_results_to": args.save_team_results_to,
                    "log_to_wandb": args.log_to_wandb,
                }
                eval_function = _get_default_eval(args, eval_kwargs)
                results = eval_function(**eval_kwargs)
                print(json.dumps(results, indent=4, sort_keys=True))
                all_results[battle_format].append(results)
    return all_results


def add_cli(parser):
    parser.add_argument(
        "--agent",
        required=False,
        choices=get_pretrained_model_names(),
        help="Choose a pretrained model to evaluate. Required unless using --custom_checkpoint_path.",
    )
    parser.add_argument(
        "--custom_checkpoint_path",
        type=str,
        default=None,
        help="Path to a custom checkpoint file (.pt). When provided, --agent specifies which base model config to use.",
    )
    parser.add_argument(
        "--eval_type",
        required=True,
        choices=["heuristic", "il", "ladder", "pokeagent"],
        help=(
            "Type of evaluation to perform. 'heuristic' will run against 6 "
            "heuristic baselines, 'il' will run against a BCRNN baseline, "
            "'ladder' will queue the agent for battles on your self-hosted Showdown ladder, "
            "'pokeagent' will submit the agent to the NeurIPS 2025 PokéAgent Challenge ladder!"
        ),
    )
    parser.add_argument(
        "--gens",
        type=int,
        nargs="+",
        default=[1],
        help="Specify the Pokémon generations to evaluate.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["ou"],
        choices=["ubers", "ou", "uu", "nu"],
        help="Specify the battle tier.",
    )
    parser.add_argument(
        "--total_battles",
        type=int,
        default=10,
        help=(
            "Number of battles to run before returning eval stats. "
            "Note this is the total sample size across all parallel actors (if applicable)."
        ),
    )
    parser.add_argument(
        "--checkpoints",
        type=int,
        nargs="+",
        default=[None],
        help="Checkpoints to evaluate.",
    )
    parser.add_argument(
        "--username",
        default="Metamon",
        help="Username for the Showdown server.",
    )
    parser.add_argument(
        "--password",
        default=None,
        help="Password for the Showdown server.",
    )
    parser.add_argument(
        "--avatar",
        default="red-gen1main",
        help="Avatar to use for the battles.",
    )
    parser.add_argument(
        "--team_set",
        default="competitive",
        help="Team Set. Built-in options are: "
        + ", ".join(metamon.env.METAMON_TEAM_SETS),
    )
    parser.add_argument(
        "--battle_backend",
        type=str,
        default="poke-env",
        choices=["poke-env", "metamon"],
        help=(
            "Method for interpreting Showdown's requests and simulator messages. "
            "poke-env is the default. metamon is an experimental option that aims to "
            "remove sim2sim gap by reusing the code that generates our huggingface "
            "replay dataset."
        ),
    )
    parser.add_argument(
        "--async_mp_context",
        type=str,
        default="forkserver",
        help="Async environment setup method. Does not apply to `--eval_type ladder` or `--eval_type pokeagent`. Options: 'forkserver' (recommended, fast), 'fork' (fastest but unsafe with threads), 'spawn' (slowest but safest). Use 'spawn' only if others hang.",
    )
    parser.add_argument(
        "--save_trajectories_to",
        default=None,
        help="Save replays (in the parsed replay format) to a directory.",
    )
    parser.add_argument(
        "--save_team_results_to",
        default=None,
        help="Save records of team selection, opponent, and outcome.",
    )
    parser.add_argument(
        "--log_to_wandb",
        action="store_true",
        help="Log results to Weights & Biases.",
    )

    return parser


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Evaluate a pretrained Metamon model by playing battles against opponents. "
        "This script allows you to evaluate a pretrained model's performance against a set of "
        "heuristic baselines, local ladder, or the PokéAgent Challenge ladder. It can also save replays in the same format "
        "as the human replay dataset for further training."
    )
    add_cli(parser)
    args = parser.parse_args()
    _run_default_evaluation(args)
