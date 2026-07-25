---
name: determinism-and-seeds
description: "How to make metamon battles deterministic / reproducible by seeding the Showdown simulator RNG, plus the repo's other random sources (per-lane seed draw, dataset split_seed, torch.manual_seed, opponent rng, default_rng). Use when asked about seeding the pokemon showdown simulator, reproducing a battle result, setting a fixed seed, RNG determinism, the 4-uint16 Showdown PRNG seed, np.random.seed vs random.Random, or why the same action from the same gamestate gives a different outcome."
---

# Determinism & Seeds in metamon

Pokemon has many RNG-dependent events (damage rolls, crits, secondary-effect
procs, accuracy checks, confusion/duration rolls). Metamon wraps the official
Pokemon Showdown sim, whose battle RNG is **fully deterministic given a seed**.
This skill explains where seeds enter the pipeline, what else is random, and how
to make a run byte-reproducible.

## TL;DR — how to get a reproducible battle

1. Pass `--seed N` to the CLI (online RL, evaluate, team-construction sim) **or**
   construct `VectorizedShowdownEnv(..., seed=N)`.
2. Call `env.reset(seed=N)` immediately before the battle you care about, with
   the same `batched_envs` and the same number of prior resets.
3. Same seed + same action stream + same teams ⇒ identical damage rolls, crits,
   misses, secondary procs. Every time.

The one subtlety: the seed does **not** go straight to Showdown. It seeds a
Python `random.Random` that **draws a fresh 4×uint16 Showdown PRNG seed per
lane per start** (see "The seed draw" below). So reproducibility is keyed to the
**order and count of lane starts**, not just the seed number.

## Where the seed enters the simulator

### The chain: env → sim_process → battle_host.js → Showdown

1. **`VectorizedShowdownEnv.__init__(..., seed=None)`**
   (`metamon/env/vectorized/vector_env.py:106`) creates
   `self._rng = random.Random(seed)`.
2. **On each lane start** (`_start_lane`, `vector_env.py:339-340`):
   ```python
   seed = self._random_seed()
   self.proc.start_battle(i, self.battle_format, p1=p1_spec, p2=p2_spec, seed=seed)
   ```
3. **`_random_seed()`** (`vector_env.py:347-349`) draws the Showdown-format seed:
   ```python
   def _random_seed(self):
       # Showdown PRNG seed: four 16-bit ints (sodium seed also accepted as str).
       return [self._rng.randint(0, 0xFFFF) for _ in range(4)]
   ```
   This is the format Showdown's `PRNG` constructor expects — a list of four
   16-bit ints. (Showdown also accepts a sodium string seed, but metamon always
   emits the int-list form.)
4. **`ShowdownSimProcess.start_battle(..., seed=None)`**
   (`metamon/env/vectorized/sim_process.py:240-253`) puts `"seed"` into the JSON
   `start` message only when it is not `None`.
5. **The Node host** (`metamon/env/vectorized/battle_host.js:207`):
   ```js
   if (msg.seed !== undefined && msg.seed !== null) spec.seed = msg.seed;
   ```
   Showdown then builds its battle `PRNG` from `spec.seed`, and every
   `random()` call in the battle engine (damage rolls, accuracy, crits, etc.)
   draws from that PRNG.

### `reset()` reseeds everything

`VectorizedShowdownEnv.reset(*, seed=None, options=None)`
(`vector_env.py:653-656`):
```python
if seed is not None:
    self._rng.seed(seed)
    np.random.seed(seed)
```
So `env.reset(seed=N)` reseeds both the per-lane-seed `random.Random` **and**
the global numpy RNG. Call it right before the battle you want reproduced.

## The other random sources (don't forget these)

Determinism of the *simulator* is necessary but not sufficient for full-run
reproducibility. These also consume randomness and are seeded by the same
top-level `--seed`:

| Source | Where | Seeded by | Notes |
|---|---|---|---|
| Per-lane Showdown PRNG seed draw | `vector_env.py` `_random_seed` | `self._rng = random.Random(seed)` | The main one — see above. |
| Opponent sampling / `RandomPolicy` | `metamon/env/vectorized/opponent.py:66` `self.rng = rng or np.random.default_rng()` | **Not** seeded by `--seed` unless you pass `rng=` in | If you need reproducible opponent picks, construct the opponent with an explicit `np.random.default_rng(N)`. The `OpponentPool` (`metamon/rl/evaluate/opponent_pool.py:61-70`) takes `rng: Optional[random.Random]` — pass `random.Random(N)`. |
| `np.random` global | `vector_env.py:656` `np.random.seed(seed)` on reset | `--seed` via `reset()` | Used by various numpy sampling. |
| Team-construction coordinate ascent / simulation | `metamon/backend/team_construction/cli.py` `--seed` (default 0) | `--seed` | Defaults to 0, so sims are reproducible by default. |
| Team-prediction training | `metamon/backend/team_prediction/train_prediction_model.py:631-632` `random.seed(config.seed); torch.manual_seed(config.seed)` | `config.seed` (default 42) | Sets both `random` and torch. |
| IL training train/val split | `metamon/il/train.py:143` `torch.Generator().manual_seed(231)` | Hard-coded `231` | Always deterministic regardless of `--seed`. |
| Parsed-replay dataset train/test split | `metamon/data/parsed_replay_dset.py:116,354` `split_seed=42` | `split_seed` (default 42, configurable via `DatasetConfig`) | `random.Random(self.split_seed).shuffle(fnames)`. |
| Heuristic baselines (BasicHeuristic, Kaizo, etc.) | `metamon/baselines/heuristic/*.py` use `random.random()` | **Global `random`** — not explicitly seeded by the env | These use the module-level `random.random()`. To make a heuristic opponent reproducible, call `random.seed(N)` in your script before constructing it, or wrap it. |
| `StatsDropoutObservationSpace` | `metamon/rl/online_rl.py:174` `random.random() < self.dropout_prob` | Global `random` | Reproducible only if `random` is seeded (e.g. via `env.reset(seed=N)` → `np.random.seed`, **but that does NOT reseed `random`**). See gotcha below. |

## CLI flags that take a seed

| CLI | Flag | Default | Where it lands |
|---|---|---|---|
| `metamon.rl.online_rl` (collect/learner/validator) | `--seed` | `None` | Passed to both `make_collect_train_env` and the val env factory (`online_envs.py:332,410`). |
| `metamon.rl.evaluate` (`--eval_type ...`) | `--seed` (kwarg) | `None` | `make_metamon_env(..., seed=seed)` (`evaluate/__main__.py:107,169`). |
| `metamon.env.vectorized._profile_step` | `--seed` | `0` | Both the env and a local `np.random.default_rng(args.seed)`. |
| `metamon.env.vectorized._smoke_env` | (hard-coded) | `0` | `seed=0`. |
| `metamon.backend.team_construction` CLI (simulate, coordinate_ascent, etc.) | `--seed` / `--seed-team` | `0` | Internal samplers. `--seed-team` takes a packed-team string, not an int. |
| Team-prediction training | `config.seed` | `42` | `random.seed` + `torch.manual_seed`. |

**Default is `None` for the RL pipelines** — meaning an unseeded run is
non-deterministic. Always pass `--seed` if you want reproducibility.

## Gotchas

### 1. `env.reset(seed=N)` reseeds `random.Random` and numpy, but NOT the global `random` module

`reset()` does `self._rng.seed(seed); np.random.seed(seed)`. It does **not** call
`random.seed(seed)`. Anything that uses the module-level `random.random()`
(heuristic baselines, `StatsDropoutObservationSpace`, `random.randint` for
battle IDs) is **not** reseeded by `reset()`. For full reproducibility call
`random.seed(N)` yourself in your script, or at env-construction time before
any of those components are built.

The team-prediction trainer (`train_prediction_model.py:631`) does both
`random.seed` and `torch.manual_seed` — that's the cleanest pattern to copy.

### 2. Reproducibility is keyed to lane-start order, not just the seed number

`_random_seed()` draws sequentially from `self._rng`. Lane 0's first battle gets
the first 4 ints, lane 1's first battle gets the next 4, etc. After a lane is
reset/restarted, it draws again. So to reproduce a *specific* battle:

- same `seed` / `reset(seed=...)`,
- same `batched_envs`,
- same number of prior `reset()`/restart cycles,
- same teams on the same lanes.

If you want one isolated battle to be byte-identical, re-seed via
`env.reset(seed=N)` immediately before starting it and don't interleave other
lane activity.

### 3. Opponent policy RNGs are often unseeded

`RandomPolicy` and `OpponentPool` default to unseeded `np.random.default_rng()`
/ `random.Random()`. `--seed` does **not** flow into them automatically (the
env passes its `seed` to the sim, not to the opponent policy's rng). Pass an
explicit `rng=` when constructing these if you need their action picks to be
reproducible.

### 4. Showdown-side nondeterminism beyond the PRNG

The Showdown battle PRNG covers damage rolls, accuracy, crits, secondary
effects, status duration, etc. — that's the bulk of "RNG-dependent actions."
But a few things are **not** PRNG-driven and can still differ between runs even
with the same seed:
- **Team ordering within a side** — Showdown shuffles the lead unless the team
  is passed in a fixed order. Metamon passes packed teams; if your team source
  shuffles, the lead differs.
- **Map iteration order in JS** — Showdown occasionally iterates object keys,
  which is insertion-order-stable in V8 but not guaranteed across Node versions.
  In practice this hasn't caused issues, but if you see subtle drift across Node
  versions, this is why.
- **Floating-point / batched inference nondeterminism on the GPU** — for the
  *learner*, not the sim. The sim is CPU/Node only and bit-stable.

### 5. `np.random.seed` is global state

`reset()` calls `np.random.seed(seed)`, which mutates the global numpy RNG. If
other code in the same process relies on numpy randomness, a reset will perturb
it. Prefer `np.random.default_rng(seed)` (a local `Generator`) for new code —
that's what `_smoke_env.py` and `_profile_step.py` do.

## Reproducing a specific online-RL battle

Online collection writes trajectories to `buffer_dir/<format>/` as
`.json.lz4` files whose names encode the opponents, teamset, and win/loss (see
the online-collector skill for the `_ts-` tokens). The trajectory itself does
**not** store the Showdown PRNG seed — it's drawn fresh each lane start from
`self._rng`. So to replay a specific collected battle bit-identically you'd
need to reproduce the exact lane-start sequence, which is impractical across a
long run.

For **targeted reproduction** (e.g. debugging a single matchup), don't try to
replay from the buffer — instead reconstruct the two teams and run a fresh
isolated sim with a fixed `--seed`:

```python
from metamon.env.vectorized.vector_env import VectorizedShowdownEnv
env = VectorizedShowdownEnv(
    player_team_set=<your team set>,
    opponent_team_set=<opponent team set>,
    battle_format="gen1ou",
    batched_envs=1, n_workers=1,
    seed=1234,
    ...
)
env.reset(seed=1234)  # reseed immediately before the battle
# ... send the same action stream ...
```

This is deterministic by construction.

## Smoke / profiling references

- `metamon/env/vectorized/_smoke_env.py` — `seed=0`, a quick deterministic env
  smoke test.
- `metamon/env/vectorized/_profile_step.py` — `--seed` (default 0) drives both
  the env and a local `np.random.default_rng(args.seed)` for action sampling.

## Best practices

1. **Always pass `--seed` to RL runs you care about reproducing.** The RL
   pipelines default to `None` (non-deterministic).
2. **For one-off reproducible sims, call `env.reset(seed=N)` right before the
   battle**, with `batched_envs` matching the run you're reproducing.
3. **Seed `random` yourself if you use heuristic opponents or
   `StatsDropoutObservationSpace`** — `reset()` does not reseed the global
   `random` module. The cleanest pattern is `random.seed(N);
   torch.manual_seed(N); np.random.seed(N)` (or a local `default_rng`) at the
   top of your script.
4. **Pass explicit `rng=` to `RandomPolicy` / `OpponentPool`** if you need their
   picks reproducible; `--seed` doesn't flow into them.
5. **Use `np.random.default_rng(seed)` (local `Generator`) for new code**, not
   the global `np.random.seed`, to avoid cross-talk with `reset()`.
6. **Don't expect to bit-replay a specific collected online-RL battle** from the
   buffer — the PRNG seed isn't stored. Reconstruct the teams and run a fresh
   seeded sim instead.
7. **Team-construction sims default to `--seed 0`** and are reproducible out of
   the box — no action needed unless you want a different seed.

## Reference

- `metamon/env/vectorized/vector_env.py:106` — `VectorizedShowdownEnv.__init__`
  (`seed` param → `self._rng`)
- `metamon/env/vectorized/vector_env.py:339-340` — `_start_lane` draws the seed
  and calls `start_battle(..., seed=seed)`
- `metamon/env/vectorized/vector_env.py:347-349` — `_random_seed` (the 4×uint16
  Showdown PRNG seed)
- `metamon/env/vectorized/vector_env.py:653-656` — `reset(seed=...)` reseeds
  `self._rng` and `np.random`
- `metamon/env/vectorized/sim_process.py:240-253` —
  `ShowdownSimProcess.start_battle` puts `seed` into the JSON `start` msg
- `metamon/env/vectorized/battle_host.js:207` — host copies `msg.seed` into
  `spec.seed` for Showdown
- `metamon/env/vectorized/opponent.py:62-66` — `RandomPolicy(rng=...)` (default
  unseeded `default_rng()`)
- `metamon/rl/evaluate/opponent_pool.py:61-70` — `OpponentPool(rng=...)`
  (default unseeded `random.Random()`)
- `metamon/rl/online_rl.py:392,460,528,563,838` — `--seed` wired through the
  online RL CLI to collect + val env factories
- `metamon/rl/online_envs.py:291,332,391,410` — `make_collect_train_env` /
  val env factory pass `seed` to `make_metamon_env`
- `metamon/rl/evaluate/__main__.py:107,169` — evaluate CLI `seed` kwarg
- `metamon/rl/dataset_config.py:370-392` — `split_seed=42` for offline dataset
  train/test split
- `metamon/data/parsed_replay_dset.py:116,142,354` —
  `random.Random(self.split_seed).shuffle(fnames)`
- `metamon/il/train.py:143` — IL train/val split, hard-coded `manual_seed(231)`
- `metamon/backend/team_prediction/train_prediction_model.py:631-632` —
  `random.seed` + `torch.manual_seed` (cleanest pattern)
- `metamon/backend/team_construction/cli.py` — `--seed` (default 0) across the
  simulate / coordinate-ascent subcommands; `--seed-team` (packed team string)
- `metamon/env/vectorized/_smoke_env.py:34` and
  `metamon/env/vectorized/_profile_step.py:32,69,79` — seeded smoke/profiling
  entry points
- See also: the `online-training` / `online-collector` / `online-learner` /
  `online-validator` skills for how `--seed` flows through those launch scripts.
