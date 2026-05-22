"""
python -m metamon.rl.evaluate.ladder_self_play --format gen9ou --gpus 0 1 --config config.yaml --save_trajectories_to ./trajectories
"""

from metamon.rl.evaluate.ladder_self_play.launch_models import main

if __name__ == "__main__":
    main()
