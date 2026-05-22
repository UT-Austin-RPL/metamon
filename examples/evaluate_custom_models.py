"""Evaluate custom checkpoints from ``rl.train`` or ``rl.finetune``.

See ``metamon/rl/pretrained.py`` (``LocalPretrainedModel``, ``LocalFinetunedModel``)
and ``metamon/rl/evaluate/README.md`` for other eval modes.
"""

import metamon
from metamon.rl import (
    pretrained_vs_pokeagent_ladder,
    LocalPretrainedModel,
    LocalFinetunedModel,
)
from metamon.rl.pretrained import SmallRL

# --- Model trained from scratch (``rl.train``) ---
#
# python -m metamon.rl.train \\
#     --run_name gen9v3 \\
#     --model_gin_config medium_multitaskagent.gin \\
#     --train_gin_config binary_rl.gin \\
#     --dataset_config self_play_dset.yaml \\
#     --save_dir ~/metamon_ckpts/ \\
#     --obs_space TeamPreviewObservationSpace \\
#     --tokenizer DefaultObservationSpace-v1 \\
#     --log
MyCustomModel = LocalPretrainedModel(
    amago_ckpt_dir="~/metamon_ckpts/",
    model_name="gen9v3",
    model_gin_config="medium_multitaskagent.gin",
    train_gin_config="binary_rl.gin",
    default_checkpoint=40,
    action_space=metamon.interface.DefaultActionSpace(),
    observation_space=metamon.interface.TeamPreviewObservationSpace(),
    tokenizer=metamon.tokenizer.get_tokenizer("DefaultObservationSpace-v1"),
)

# --- Model finetuned from a public checkpoint (``rl.finetune``) ---
#
# python -m metamon.rl.finetune \\
#     --run_name smallrl_finetune \\
#     --save_dir ~/metamon_ckpts/ \\
#     --base_model SmallRL \\
#     --dataset_config self_play_dset.yaml \\
#     --epochs 10 \\
#     --log \\
#     --eval_gens 9
MyFinetunedModel = LocalFinetunedModel(
    base_model=SmallRL,
    amago_ckpt_dir="~/metamon_ckpts/",
    model_name="smallrl_finetune",
    default_checkpoint=10,
)

teams = metamon.env.get_metamon_teams("gen9ou", "competitive")
results = pretrained_vs_pokeagent_ladder(
    pretrained_model=MyFinetunedModel,
    username="PAC-MyTeamName",
    password="my_password",
    battle_format="gen9ou",
    team_set=teams,
    total_battles=10,
)
print(results)
