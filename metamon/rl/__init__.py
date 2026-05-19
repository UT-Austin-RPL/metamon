from metamon.rl.pretrained import (
    LocalPretrainedModel,
    PretrainedModel,
    LocalFinetunedModel,
)
from metamon.rl.evaluate import (
    pretrained_vs_pokeagent_ladder,
    pretrained_vs_local_ladder,
    pretrained_vs_baselines,
)

# Import gen1 specialist models to register them
from metamon.rl import gen1_binary_models  # noqa: F401

import os

MODEL_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs", "models")
TRAINING_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs", "training")
