# Metamon Ladder Eval

Use this note when evaluating a local Gen1 specialist run on a Showdown ladder.

## Path Convention

Bulba checkpoints live under:

```text
/home/eddie/metamon/models/gen1ou-specialist/bulba/ckpts/policy_weights/policy_epoch_<N>.pt
```

Load the highest-numbered `policy_epoch_*.pt` file in that directory.

## Ladder Eval Command

Use `--agent Bulba` so the evaluator loads the local Gen1 specialist config, then pass the checkpoint by path.

```bash
latest_ckpt="$(find /home/eddie/metamon/models/gen1ou-specialist/bulba/ckpts/policy_weights -maxdepth 1 -name 'policy_epoch_*.pt' | sort -V | tail -n 1)"

METAMON_CACHE_DIR=/home/eddie/metamon_cache uv run python -m metamon.rl.evaluate \
  --agent Bulba \
  --eval_type ladder \
  --gens 1 \
  --formats ou \
  --team_set competitive \
  --total_battles 1 \
  --username Metamon \
  --custom_checkpoint_path "$latest_ckpt" \
  --step
```

If you already know the epoch number, `--checkpoints <N>` is also valid.
