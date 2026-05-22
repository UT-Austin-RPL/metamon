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
from metamon.backend.team_preview.preview import TeamPreviewModel
from metamon.rl.metamon_to_amago import (
    make_baseline_env,
    make_local_ladder_env,
    make_pokeagent_ladder_env,
    make_challenge_env,
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
    action_temperature: float = 1.0,
    async_mp_context: str = "forkserver",
    battle_backend: str = "metamon",
    log_to_wandb: bool = False,
    save_trajectories_to: Optional[str] = None,
    save_results_to: Optional[str] = None,
    baselines: Optional[List[str]] = None,
    team_preview_model: Optional[TeamPreviewModel] = None,
) -> Dict[str, Any]:
    """Evaluate a pretrained model against built-in baseline opponents.

    Defaults to the 6 baselines that the paper calls the "Heuristic Composite Score",
    but you can specify a list of any of the available baselines (see metamon.baselines.get_all_baseline_names()).
    """
    agent = pretrained_model.initialize_agent(
        checkpoint=checkpoint, log=log_to_wandb, action_temperature=action_temperature
    )
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
            save_results_to=save_results_to,
            battle_backend=battle_backend,
            team_set=team_set,
            opponent_type=get_baseline(opponent),
            team_preview_model=team_preview_model,
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
    action_temperature: float = 1.0,
    team_preview_model: Optional[TeamPreviewModel] = None,
    **ladder_kwargs,
) -> Dict[str, Any]:
    """Helper function for ladder-based evaluation."""
    agent = pretrained_model.initialize_agent(
        checkpoint=checkpoint, log=log_to_wandb, action_temperature=action_temperature
    )
    agent.env_mode = "sync"
    agent.parallel_actors = 1
    agent.verbose = False  # turn off tqdm progress bar and print poke-env battle status

    make_env = functools.partial(
        make_ladder,
        observation_space=pretrained_model.observation_space,
        action_space=pretrained_model.action_space,
        reward_function=pretrained_model.reward_function,
        num_battles=total_battles,
        team_preview_model=team_preview_model,
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
    battle_backend: str = "metamon",
    action_temperature: float = 1.0,
    save_trajectories_to: Optional[str] = None,
    save_results_to: Optional[str] = None,
    log_to_wandb: bool = False,
    team_preview_model: Optional[TeamPreviewModel] = None,
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
        action_temperature=action_temperature,
        team_preview_model=team_preview_model,
        player_username=username,
        player_avatar=avatar,
        player_team_set=team_set,
        battle_backend=battle_backend,
        battle_format=battle_format,
        save_trajectories_to=save_trajectories_to,
        save_results_to=save_results_to,
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
    battle_backend: str = "metamon",
    action_temperature: float = 1.0,
    save_trajectories_to: Optional[str] = None,
    save_results_to: Optional[str] = None,
    log_to_wandb: bool = False,
    team_preview_model: Optional[TeamPreviewModel] = None,
) -> Dict[str, Any]:
    """Evaluate a pretrained model on the PokéAgent Challenge ladder.

    Must provide a registered username and password. See instructions in the README!

    Will automatically queue the agent for ranked battles against any other agents (or humans)
    that are logged into the PokéAgent Challenge ladder.

    Once eval begins, you can watch battles in real time by visiting
    https://battling.pokeagentchallenge.com and clicking "Watch a Battle".
    Visit https://battling.pokeagentchallenge.com/ladder to see the live
    leaderboard.
    """
    return _pretrained_on_ladder(
        pretrained_model=pretrained_model,
        make_ladder=make_pokeagent_ladder_env,
        total_battles=total_battles,
        checkpoint=checkpoint,
        log_to_wandb=log_to_wandb,
        action_temperature=action_temperature,
        team_preview_model=team_preview_model,
        player_username=username,
        player_password=password,
        player_avatar=avatar,
        player_team_set=team_set,
        battle_backend=battle_backend,
        battle_format=battle_format,
        save_trajectories_to=save_trajectories_to,
        save_results_to=save_results_to,
    )


def pretrained_vs_challenge(
    pretrained_model: PretrainedModel,
    username: str,
    opponent_username: str,
    role: str,
    battle_format: str,
    team_set: metamon.env.TeamSet,
    total_battles: int,
    avatar: Optional[str] = None,
    checkpoint: Optional[int] = None,
    battle_backend: str = "metamon",
    action_temperature: float = 1.0,
    save_trajectories_to: Optional[str] = None,
    save_results_to: Optional[str] = None,
    log_to_wandb: bool = False,
    team_preview_model: Optional[TeamPreviewModel] = None,
) -> Dict[str, Any]:
    """Evaluate a pretrained model by challenging a specific opponent by username.

    This creates a deterministic head-to-head matchup between two agents.
    One side must be the "challenger" (sends challenges) and the other the
    "acceptor" (accepts challenges). Launch two instances of this command
    with opposite roles and matching usernames.

    Example (two terminals):
        Terminal 1 (challenger):
            python -m metamon.rl.evaluate --eval_type challenge --agent ModelA \\
                --username PlayerA --opponent_username PlayerB --role challenger

        Terminal 2 (acceptor):
            python -m metamon.rl.evaluate --eval_type challenge --agent ModelB \\
                --username PlayerB --opponent_username PlayerA --role acceptor
    """
    return _pretrained_on_ladder(
        pretrained_model=pretrained_model,
        make_ladder=make_challenge_env,
        total_battles=total_battles,
        checkpoint=checkpoint,
        log_to_wandb=log_to_wandb,
        action_temperature=action_temperature,
        team_preview_model=team_preview_model,
        player_username=username,
        opponent_username=opponent_username,
        role=role,
        player_avatar=avatar,
        player_team_set=team_set,
        battle_backend=battle_backend,
        battle_format=battle_format,
        save_trajectories_to=save_trajectories_to,
        save_results_to=save_results_to,
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
    elif args.eval_type == "challenge":
        base_eval_kwargs.update(
            {
                "username": args.username,
                "opponent_username": args.opponent_username,
                "role": args.role,
                "avatar": args.avatar,
            }
        )
        return pretrained_vs_challenge
    else:
        raise ValueError(f"Invalid evaluation type: {args.eval_type}")


def _run_default_evaluation(args) -> Dict[str, List[Dict[str, Any]]]:
    pretrained_model = get_pretrained_model(args.agent)
    all_results = collections.defaultdict(list)
    backend = args.battle_backend or pretrained_model.battle_backend

    # Load team preview model if checkpoint provided
    team_preview_model = None
    if args.team_preview_checkpoint is not None:
        team_preview_model = TeamPreviewModel.load_from_checkpoint(
            checkpoint_path=args.team_preview_checkpoint,
            device="cuda" if backend == "metamon" else "cpu",
            use_argmax=args.team_preview_use_argmax,
        )
        print(f"Team preview model loaded from: {args.team_preview_checkpoint}")

        if backend != "metamon":
            print(
                "WARNING: team_preview_model only works with --battle_backend metamon. It will be ignored."
            )
            team_preview_model = None

    # Print banner and evaluation info
    metamon.print_banner()
    print(f"  Agent: {pretrained_model.model_name}  |  Backend: {backend}", end="")
    if team_preview_model is not None:
        print(f"  |  Team Preview: ✓")
    else:
        print()
    print()

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
                eval_kwargs = {
                    "pretrained_model": pretrained_model,
                    "battle_format": battle_format,
                    "team_set": player_team_set,
                    "total_battles": args.total_battles,
                    "checkpoint": checkpoint,
                    "battle_backend": backend,
                    "save_trajectories_to": args.save_trajectories_to,
                    "action_temperature": args.temperature,
                    "save_results_to": args.save_results_to,
                    "log_to_wandb": args.log_to_wandb,
                    "team_preview_model": team_preview_model,
                }
                eval_function = _get_default_eval(args, eval_kwargs)
                results = eval_function(**eval_kwargs)
                print(json.dumps(results, indent=4, sort_keys=True))
                all_results[battle_format].append(results)
    return all_results


def add_cli(parser):
    parser.add_argument(
        "--agent",
        required=True,
        choices=get_pretrained_model_names(),
        help="Choose a pretrained model to evaluate.",
    )
    parser.add_argument(
        "--eval_type",
        required=True,
        choices=["heuristic", "il", "ladder", "pokeagent", "challenge"],
        help=(
            "Type of evaluation to perform. 'heuristic' will run against 6 "
            "heuristic baselines, 'il' will run against a BCRNN baseline, "
            "'ladder' will queue the agent for battles on your self-hosted Showdown ladder, "
            "'pokeagent' will submit the agent to the NeurIPS 2025 PokéAgent Challenge ladder, "
            "'challenge' will send/accept challenges to a specific opponent by username "
            "(launch two instances with opposite --role for head-to-head)."
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
        "--opponent_username",
        default=None,
        help=(
            "Username of the opponent to challenge (only for --eval_type challenge). "
            "Launch two instances with opposite --role and matching usernames."
        ),
    )
    parser.add_argument(
        "--role",
        default="challenger",
        choices=["challenger", "acceptor"],
        help=(
            "Role in a challenge matchup (only for --eval_type challenge). "
            "'challenger' sends challenges to --opponent_username, "
            "'acceptor' waits for and accepts challenges from --opponent_username."
        ),
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
        default=None,
        choices=["poke-env", "metamon", "pokeagent"],
        help=(
            "Method for interpreting Showdown's requests and simulator messages. "
            "Handles backwards-compatibility for models trained on old versions of metamon. "
            "`None` will default to the version requested by the pretrained model you are evaluating."
            "'metamon' is the lateset version"
            "'pokeagent' maintains policies trained and used as the organizer baselines during the PokéAgent Challenge"
            "'poke-env' is deprecated; maintains the original paper's models. "
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
        "--save_results_to",
        default=None,
        help="Directory to save per-battle result logs.",
    )
    parser.add_argument(
        "--log_to_wandb",
        action="store_true",
        help="Log results to Weights & Biases.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for temperature-based sampling. Higher temperature means more exploration.",
    )
    parser.add_argument(
        "--team_preview_checkpoint",
        type=str,
        default=None,
        help=(
            "Path to a team preview model checkpoint (e.g., './checkpoints/best_model.pt'). "
            "If provided, the model will predict which pokemon to lead with during team preview. "
            "Only works with --battle_backend metamon."
        ),
    )
    parser.add_argument(
        "--team_preview_use_argmax",
        action="store_true",
        help=(
            "If set, use argmax for team preview lead selection instead of sampling from the distribution. "
            "Only applies when --team_preview_checkpoint is provided."
        ),
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
