#!/usr/bin/env python3
"""
Battle on the main Pokémon Showdown server.

WARNING: Bots are NOT allowed on the official server without explicit permission!
Only use this if you have received approval from Pokémon Showdown staff.
"""

from poke_env.ps_client.server_configuration import ServerConfiguration
from metamon.env.wrappers import QueueOnLocalLadder
from metamon.rl.pretrained import get_pretrained_model
from metamon.rl.metamon_to_amago import PSLadderAMAGOWrapper
import metamon


# Main Pokémon Showdown server configuration
MainServerConfiguration = ServerConfiguration(
    "wss://sim3.psim.us/showdown/websocket",
    "https://play.pokemonshowdown.com/action.php?",
)


class MainServerLadder(QueueOnLocalLadder):
    """
    Battle on the main Pokémon Showdown server (play.pokemonshowdown.com).

    REQUIRES:
    - Registered account on play.pokemonshowdown.com
    - Explicit permission from Pokémon Showdown staff to run bots
    """

    @property
    def server_configuration(self):
        return MainServerConfiguration


class MainServerAcceptChallenges(metamon.env.PokeEnvWrapper):
    """
    Sit online on main server and accept challenges (doesn't actively ladder).

    This will connect to the server and wait for incoming challenges without
    actively queueing for battles.
    """

    @property
    def server_configuration(self):
        return MainServerConfiguration


def battle_on_main_server(
    agent_name: str,
    username: str,
    password: str,
    battle_format: str = "gen1ou",
    team_set_name: str = "modern_replays_v2",
    num_battles: int = 10,
    checkpoint: int = None,
    accept_only: bool = False,
    save_trajectories_to: str = None,
    save_results_to: str = None,
):
    """
    Battle on the main Pokémon Showdown server.

    Args:
        agent_name: Name of the pretrained model (e.g., "DampedBinarySuperV1_Epoch4")
        username: Your REGISTERED Pokémon Showdown username
        password: Your Pokémon Showdown password
        battle_format: Battle format (e.g., "gen1ou")
        team_set_name: Team set to use
        num_battles: Number of battles to play
        checkpoint: Model checkpoint to load (None = default)
        accept_only: If True, sit online and accept challenges instead of actively laddering
        save_trajectories_to: Directory to save battle trajectories (in parsed replay format)
        save_results_to: Directory to save team selection and battle outcome stats
    """
    # Get the pretrained model
    pretrained_model = get_pretrained_model(agent_name)

    # Initialize the agent
    agent = pretrained_model.initialize_agent(checkpoint=checkpoint, log=False)
    agent.env_mode = "sync"
    agent.parallel_actors = 1
    agent.verbose = True

    # Get team set
    team_set = metamon.env.get_metamon_teams(battle_format, team_set_name)

    # Create the main server environment
    def make_env():
        if accept_only:
            # Sit online and accept challenges
            env = MainServerAcceptChallenges(
                battle_format=battle_format,
                observation_space=pretrained_model.observation_space,
                action_space=pretrained_model.action_space,
                reward_function=pretrained_model.reward_function,
                player_team_set=team_set,
                player_username=username,
                player_password=password,
                start_challenging=False,  # Don't actively challenge
                battle_backend="poke-env",
                save_trajectories_to=save_trajectories_to,
                save_results_to=save_results_to,
            )
            from metamon.rl.metamon_to_amago import MetamonAMAGOWrapper
            return MetamonAMAGOWrapper(env)
        else:
            # Actively queue for ladder battles
            env = MainServerLadder(
                battle_format=battle_format,
                num_battles=num_battles,
                observation_space=pretrained_model.observation_space,
                action_space=pretrained_model.action_space,
                reward_function=pretrained_model.reward_function,
                player_team_set=team_set,
                player_username=username,
                player_password=password,
                battle_backend="poke-env",
                save_trajectories_to=save_trajectories_to,
                save_results_to=save_results_to,
            )
            return PSLadderAMAGOWrapper(env)

    # Run battles
    print(f"Starting battles on main server as '{username}'...")
    print(f"Format: {battle_format}")
    print(f"Agent: {agent_name}")
    if accept_only:
        print("Mode: ACCEPT ONLY - Waiting for incoming challenges...")
        print("The bot will sit online and accept any Gen1 OU challenges it receives.")
    else:
        print("Mode: ACTIVE LADDERING - Queuing for battles...")
    if save_trajectories_to:
        print(f"Saving trajectories to: {save_trajectories_to}/{battle_format}/")
    if save_results_to:
        print(f"Saving battle results to: {save_results_to}/")
    print("WARNING: Make sure you have permission to run bots on the main server!")

    results = agent.evaluate_test(
        [make_env],
        timesteps=num_battles * 1000,
        episodes=num_battles,
    )

    print("\nResults:")
    print(results)
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Battle on main Pokémon Showdown server (REQUIRES PERMISSION!)"
    )
    parser.add_argument("--agent", required=True, help="Agent name")
    parser.add_argument("--username", required=True, help="Your registered PS username")
    parser.add_argument("--password", required=True, help="Your PS password")
    parser.add_argument("--battle_format", default="gen1ou", help="Battle format")
    parser.add_argument("--team_set", default="modern_replays_v2", help="Team set")
    parser.add_argument("--num_battles", type=int, default=10, help="Number of battles")
    parser.add_argument("--checkpoint", type=int, default=None, help="Model checkpoint")
    parser.add_argument(
        "--accept_only",
        action="store_true",
        help="Sit online and accept challenges instead of actively laddering",
    )
    parser.add_argument(
        "--save_trajectories_to",
        default=None,
        help="Directory to save battle trajectories (in parsed replay format)",
    )
    parser.add_argument(
        "--save_results_to",
        "--save_team_results_to",
        default=None,
        help=(
            "Directory to save team selection and battle outcome stats. "
            "--save_team_results_to is kept as a deprecated alias."
        ),
    )

    args = parser.parse_args()

    battle_on_main_server(
        agent_name=args.agent,
        username=args.username,
        password=args.password,
        battle_format=args.battle_format,
        team_set_name=args.team_set,
        num_battles=args.num_battles,
        checkpoint=args.checkpoint,
        accept_only=args.accept_only,
        save_trajectories_to=args.save_trajectories_to,
        save_results_to=args.save_results_to,
    )
