"""
Run the self-play launcher as a module.

Usage:
    python -m metamon.rl.self_play --format gen9ou --gpus 0 1 --config metamon/rl/self_play/gen9ou_config.yaml --save_trajectories_to ./trajectories
"""

from metamon.rl.self_play.launch_models import main

if __name__ == "__main__":
    main()
