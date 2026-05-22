# Evaluation

`metamon.rl.evaluate` runs pretrained models against various opponents, and provides launchers to automatically manage large-scale evaluations across many models and GPUs.

## Evaluate (`python -m metamon.rl.evaluate`)

The main evaluation script. Runs one pretrained model against a chosen opponent type.

### Eval Types

#### `heuristic` — Built-in Baselines

Play against the 6 heuristic baselines from the paper (RandomBaseline, Gen1BossAI, Grunt, GymLeader, PokeEnvHeuristic, EmeraldKaizo):

```bash
python -m metamon.rl.evaluate \
    --eval_type heuristic \
    --agent Kakuna \
    --gens 1 \
    --formats ou \
    --total_battles 100
```

#### `il` — IL Baseline

Play against the BaseRNN imitation learning policy:

```bash
python -m metamon.rl.evaluate --eval_type il --agent Kakuna --gens 1 --formats ou --total_battles 50
```

#### `ladder` — Local Showdown Ladder

Queue for battles on your local Showdown server against any other online agents or humans:

```bash
python -m metamon.rl.evaluate \
    --eval_type ladder \
    --agent Kakuna \
    --gens 1 \
    --formats ou \
    --total_battles 50 \
    --username MyUsername \
    --team_set competitive
```

#### `pokeagent` — PokéAgent Challenge Ladder

Submit to the PokéAgent Challenge practice ladder (requires a registered username and password):

```bash
python -m metamon.rl.evaluate \
    --eval_type pokeagent \
    --agent Kakuna \
    --gens 9 \
    --formats ou \
    --total_battles 50 \
    --username RegisteredName \
    --password MyPassword
```

#### `challenge` — Head-to-Head by Username

Send or accept challenges to a specific opponent. Launch two instances with opposite `--role` and matching usernames:

```bash
# Terminal 1 (acceptor — start first):
python -m metamon.rl.evaluate --eval_type challenge --agent Kakuna \
    --username PlayerA --opponent_username PlayerB --role acceptor \
    --gens 1 --formats ou --total_battles 50

# Terminal 2 (challenger — start second):
python -m metamon.rl.evaluate --eval_type challenge --agent SyntheticRLV2 \
    --username PlayerB --opponent_username PlayerA --role challenger \
    --gens 1 --formats ou --total_battles 50
```

The acceptor must be online before the challenger starts sending challenges.

### Common Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--agent` | (required) | Pretrained model name (see `metamon/rl/pretrained.py`) |
| `--gens` | `1` | Pokémon generation(s) |
| `--formats` | `ou` | Battle tier(s) (ou, uu, nu, ubers) |
| `--total_battles` | `10` | Number of battles |
| `--checkpoints` | model default | Checkpoint epoch(s) to evaluate |
| `--temperature` | `1.0` | Action sampling temperature |
| `--team_set` | `competitive` | Team set name |
| `--battle_backend` | model default | `metamon`, `pokeagent`, or `poke-env` |
| `--save_trajectories_to` | off | Save replays in parsed format |
| `--save_results_to` | off | Save per-battle result logs |
| `--team_preview_checkpoint` | off | Team preview model for Gen 9 |

### Custom Models

To eval a custom agent trained from scratch (`rl.train`), create a `LocalPretrainedModel`. `LocalFinetunedModel` provides quick setup for models finetuned with `rl.finetune`. See [`examples/evaluate_custom_models.py`](../../../examples/evaluate_custom_models.py) for examples.

---

## Auto-Launchers

Utilities to automatically launch and manage evaluations across multiple models and GPUs. Define policies and matchups in a YAML config, then let the launcher handle subprocess orchestration, GPU assignment, and crash recovery.

All modes support `--dry_run` to preview what will be launched without actually running anything.

### Head-to-Head (`h2h`)

Play every pair of policies against each other. Produces a win matrix.

```bash
python -m metamon.rl.evaluate.h2h \
    --config metamon/rl/evaluate/h2h/example_config.yaml \
    --gpus 0 1 2 3 \
    --output_dir ./h2h_results \
    --dry_run
```

#### Config

See `metamon/rl/evaluate/h2h/example_config.yaml` (paper + PokéAgent policies). Minimal shape:

```yaml
battle_format: gen1ou
battles_per_matchup: 50

defaults:
  team_set: competitive
  battle_backend: metamon
  checkpoint: null
  temperature: 1.0

policies:
  SmallRL: {}
  SyntheticRLV2:
    checkpoint: 40
  Kadabra: {}
  Alakazam:
    variants:
      - { checkpoint: null, team_set: competitive }
      - { checkpoint: 8, team_set: modern_replays_v2 }
```

Generates all unordered pairs (N choose 2). Results are saved to `matchup_results.jsonl` (crash recovery) and `win_matrix.csv`.

#### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--max_concurrent` | #GPUs | Max matchups running in parallel |
| `--timeout` | 3600 | Seconds before killing a matchup |
| `--acceptor_startup_delay` | 10 | Seconds to wait for acceptor before starting challenger |
| `--save_trajectories` | off | Save trajectory files per matchup |
| `--verbose` | off | Stream subprocess output in real-time |

---

### Sweep

Evaluate one policy across a parameter grid against a fixed opponent. Useful for checkpoint sweeps, temperature tuning, etc.

```bash
python -m metamon.rl.evaluate.sweep \
    --config metamon/rl/evaluate/sweep/example_config.yaml \
    --gpus 0 1 \
    --output_dir ./sweep_results \
    --dry_run
```

Configs may use `${var}` / `${var:default}` placeholders; undeclared variables become extra CLI flags (see `sweep/temperature_sweep_self_play.yaml`).

#### Config

See `metamon/rl/evaluate/sweep/example_config.yaml`. Minimal shape:

```yaml
battle_format: gen1ou
battles_per_matchup: 50

defaults:
  team_set: competitive
  battle_backend: metamon

opponent:
  model_name: Kadabra
  checkpoint: null
  temperature: 1.0

sweep:
  model_name: SyntheticRLV2
  checkpoints: "range(40, 50, 2)"
  temperatures: [1.0, 1.5, 2.0]
```

`checkpoints` and `temperatures` accept an explicit list or a shorthand string:
- **`"range(start, stop, step)"`** — Python-style, stop exclusive. Best for integer checkpoints.
- **`"linspace(start, stop, n)"`** — N evenly-spaced points, both endpoints inclusive. Best for float temperatures.

The launcher generates a cartesian product of `checkpoints × temperatures`, each played against the fixed opponent. Same flags as h2h.

---

### Ladder Self-Play

Put multiple agents on the local Showdown ladder. They battle whoever they match with via random matchmaking. Runs continuously (restart on crash) until interrupted. This is what we used to run the PokéAgent Challenge and generate self-play data in batches.

```bash
python -m metamon.rl.evaluate.ladder_self_play \
    --config metamon/rl/evaluate/ladder_self_play/example_config.yaml \
    --format gen1ou \
    --gpus 0 1 2 3 \
    --save_trajectories_to ./trajectories \
    --dry_run
```

#### Config

See `metamon/rl/evaluate/ladder_self_play/example_config.yaml`. Ladder configs also support `"range(...)"`, `"linspace(...)"`, and `{weighted: {...}}` for checkpoints, temperatures, and team sets (via `evaluate/common.py`).

```yaml
defaults:
  team_set: competitive
  battle_backend: metamon
  checkpoints: [null]
  temperatures: [1.0, 1.25, 1.5, 2.0]
  num_agents: 1

agents:
  SynRLV2:
    model_name: SyntheticRLV2
    num_agents: 2

  Kadabra:
    model_name: Kadabra
    num_agents: 2
```

Each agent instance randomly samples a checkpoint and temperature from the list on each launch, which is a way to add variety without increasing the number of parallel agents.

#### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--n_challenges` | 50 | Battles per agent before restart |
| `--restart_delay` | 80 | Seconds between restarts |
| `--timeout` | 2700 | Seconds before killing a run |
| `--verbose` | off | Stream subprocess output |

---

## Config Syntax Reference

### Policy specification (h2h / sweep)

The outer key is used as `model_name` by default:

```yaml
policies:
  Kadabra: {}                    # model_name = "Kadabra", all defaults
  SyntheticRLV2:                 # model_name = "SyntheticRLV2"
    checkpoint: 40               # override checkpoint
  MyAlias:                       # display name in win matrix
    model_name: Alakazam2        # actual model to load
    checkpoint: 48
```

### Variants

Expand one model into multiple configs:

```yaml
policies:
  Alakazam2:
    variants:
      - { team_set: modern_replays_v2 }
      - { team_set: competitive }
      - { team_set: competitive, checkpoint: 20 }
```

Generates `Alakazam2-1`, `Alakazam2-2`, `Alakazam2-3` — all loading `Alakazam2` with different settings.

### Defaults

Any field in `defaults:` is inherited by all policies unless overridden:

```yaml
defaults:
  team_set: modern_replays_v2
  battle_backend: metamon
  checkpoint: null
  temperature: 1.0
```

This is all kinda confusing and a work in progress, so make sure to check `--dry_run` on your command to see if the results will match what you expect.