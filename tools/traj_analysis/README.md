# Squirtle ladder battle analysis

Tools for analyzing the **squirtle** agent's human-ladder battles
(`~/metamon/trajectories/squirtle/gen1ou/`, 310 battles vs 114 opponents on the
Pokémon Showdown ladder, played with the 5 `smog_ladder` teams).

## What's here

| file | purpose |
|------|---------|
| `build_cache.py` | Loads the squirtle model (latest `mini_online_smogon_v0` checkpoint) and computes, for every turn of every battle, the model's **value estimate V(s)** (NCriticsTwoHot critic, 7 gammas), Q(s,a) and advantage, plus battle metadata (team, opponent, result, pokemon remaining per side, HP, active mons). Writes `battles.parquet`, `turns.parquet`, `values.npz` to the eval cache. |
| `app.py` | Gradio demo over the cache (no model needed at runtime). |
| `compute_evals.py` | Early prototype of the value computation (first 2 battles). |

## Eval cache

Default output: `~/metamon/trajectories/squirtle/eval_cache/`

- `battles.parquet` — one row per battle: file, opponent, result, team, roster,
  n_turns, V(s₀), V(s_final), main gamma.
- `turns.parquet` — one row per turn: V(γ=0.999), V(mean γ), Q(s,a), advantage,
  player/opponent pokemon remaining, active species, HP%, action taken.
- `values.npz` — raw per-turn matrices (n_battles × max_turns × 7 gammas) + lengths + gammas.

## Build the cache (requires GPU + model checkpoint)

```bash
cd ~/repos/metamon
.venv/bin/python tools/traj_analysis/build_cache.py --out ~/metamon/trajectories/squirtle/eval_cache
```

Loads `squirtle` (i.e. `mini_online_smogon_v0`) latest checkpoint, so the model
files under `~/metamon_runs/mini_online_smogon_v0/` must be present.

## Run the demo

```bash
.venv/bin/python tools/traj_analysis/app.py --port 7860
# open http://127.0.0.1:7860
```

No model/GPU needed at demo time — everything is served from the eval cache.

### Tabs

1. **Teams vs opponent strength** — win-rate heatmap (team × weak/mid/strong
   opponent tiers derived from the squirtle agent's Laplace-smoothed win rate
   per opponent), per-team win-rate bars, and a breakdown table.
2. **Single battle evaluation** — pick any of the 310 battles: the model's
   V(s) curve per turn (γ=0.999 or mean-over-γ), per-turn ΔV bars, optional
   Q(s,a)/advantage, and a pokemon-remaining step chart for both sides.
3. **Aggregate evaluation** — mean V(s) by turn (with ±1σ band, split
   win/loss), mean per-turn ΔV by turn, and a heatmap of mean eval / win rate /
   turn counts by (squirtle remaining × opponent remaining).
