# PSRO-Lite: Prioritized Opponent Sampling for Online RL

## TL;DR / Recommendation

Implement **"PSRO-Lite"**: reweight the opponent pool *at collection time* using
a meta-distribution derived from the learner's empirical win rate per opponent,
computed from the **win/loss tags already embedded in the FIFO buffer's
trajectory filenames**. No new population growth, no extra simulation budget
beyond what collection already spends.

This is the minimal-complexity slice of the PSRO idea and is directly supported
by the literature (Prioritized Fictitious Self-Play / AlphaStar's PFSP, and the
"best response to a meta-distribution" framing of PSRO,
Lanctot et al. 2017 arXiv:1711.00832). It reuses infrastructure that already
exists in this repo (`build_payoff_matrix`, `solve_zero_sum_equilibrium`,
`OpponentPoolConfig` row expansion, per-spec `short_label` / `unique_key`) and
requires **no changes to the learner's loss, the dataset mixture ratios, or the
AMAGO training loop**.

**Deployment target**: apply to the **current live run, switched on at epoch
1000** (no A/B — wall-clock cost is prohibitive). All flags are default-off and
gated by `--psro_start_epoch`, so the switchover is a resume + flag flip, not a
new run.

---

## Why this and not "full PSRO"

| Option | Complexity | Lit support | Verdict |
|---|---|---|---|
| **A. Prioritized collection (PSRO-Lite)** — per-agent win-rate weighting | Low | PFSP, PSRO BR-to-meta-dist | **v1 — do this** |
| B. Full Nash over pool-vs-pool matrix | Med-High | canonical PSRO | Defer to v3; needs extra pool-vs-pool sim games |
| C. Per-trajectory FIFO reweighting by opponent tag | Low-Med | off-regime (Offline FSP, arXiv:2403.00841 flags the mismatch) | **Ship in v1** — no A/B means we can't tolerate buffer lag silently hurting the run |
| D. Growing population + BR oracle + pruning | High | canonical PSRO | Reject for now; checkpoint/pruning infra not worth it |

The dominant cost in real PSRO is the payoff-matrix simulation
(arXiv:2509.23462 GEMS, arXiv:2601.05279 Simulation-Free PSRO). Option A pays
**zero extra simulation**: the win-rate signal comes from games the collector is
already playing. Option B would require a separate pool-vs-pool eval sweep each
update — the exact cost those papers were written to remove.

Because we apply this to the **current live run starting at epoch 1000** with no
A/B, the buffer-lag issue is acute rather than theoretical: at epoch 1000 the
FIFO holds up to `dset_max_size` (~300k) files **all sampled uniformly**. If we
only reweight *collection*, the learner keeps training on uniform-skewed data for
a long time after the flip, which without an A/B safety net could silently hurt
the run. We therefore **ship the per-trajectory FIFO reweighting (option C)
alongside the collection reweighting in v1** — it's cheap (the filenames already
tag every trajectory with opponent + result, so no new logging) and it's the
direct fix for the lag. See "Buffer lag" below.

---

## What already exists (no changes needed)

- `metamon/env/vectorized/vector_env.py:_save_lane_outcome` writes each finished
  lane to `{buffer_dir}/{format}/metamon-{fmt}-{id}_Unrated_{player}_vs_{opponent}_{ts}_{WIN|LOSS}.json.lz4`.
  **The opponent identity and outcome are already in the filename** — this is
  the entire signal source for v1.
- `metamon/backend/team_construction/restricted_game.py` has
  `build_payoff_matrix` and `solve_zero_sum_equilibrium` (nashpy-backed). We
  reuse `solve_zero_sum_equilibrium` for the v3 Nash option and the
  payoff→mixture plumbing; v1 only needs a simple win-rate→weight map.
- `metamon/rl/evaluate/opponent_pool.py:OpponentPoolConfig` already expands
  agents into weighted rows via `num_agents` and exposes `sample_opponent()`.
- `metamon/rl/evaluate/common.py:PolicySpec.short_label` / `unique_key` give us
  the exact per-spec key used in the filenames.
- `metamon/rl/metamon_to_amago.py:MetamonOnlineExperiment.collect_new_training_data`
  is the experiment-owned hook that runs **once per epoch, in the collector
  process only**, right where we want to refresh weights.
- `metamon/env/vectorized/opponent.py:ConfigBatchedOpponent.configure` resamples
  the opponent every env reset (and `force_reset_train_envs_every=1` ⇒ once per
  epoch). This is the consumption point for new weights.
- `MetamonFIFODataset.on_end_of_collection` already calls
  `parsed_replay_dset.refresh_files()` every epoch — the natural hook point for
  re-building a weighted file index.

---

## Design

### Signal

For each opponent agent `o` seen in the buffer, compute over a rolling window of
recent finished games:

```
n_o        = #games vs o
wins_o     = #wins  vs o
p_o        = (wins_o + α) / (n_o + 2α)        # Laplace-smoothed win rate
```

`α = 1` (or a CLI knob). Only count games from the last `--psro_window` files
(per-opponent staleness → rolling window, not full-buffer, so the distribution
tracks the *current* learner rather than its whole history).

### Meta-distribution (weight per opponent)

v1 ships a **prioritized** transform (PFSP-style), not Nash:

```
exploitability_o = 0.5 - p_o                  # high when learner loses
score_o   = max(0, exploitability_o)          # drop opponents we dominate
w_o       = score_o ** (1/τ)                  # temperature τ; τ→∞ ⇒ uniform
w_o       = max(floor, w_o)                   # keep diversity floor
W         = w / sum(w)
```

- `τ` (`--psro_temp`, default 1.0): interpolates between sharp prioritization
  (τ small) and uniform (τ large).
- `floor` (`--psro_floor`, default 0.05): every opponent keeps non-zero mass so
  the FIFO doesn't collapse to one opponent and so we keep exploring cycles
  (non-transitivity — the Conflux-PSRO / Self-Play PSRO motivation).
- **EMA smoothing across updates**: `W_t = β·W_t + (1-β)·W_{t-1}` (β ~0.7) to
  stop thrashing on noisy per-epoch win rates. Initialized to uniform at the
  first update (the switchover epoch).
- **Minimum-games gate**: if `n_o < min_games` (`--psro_min_games`, default 20),
  fall back to uniform for that opponent (Wilson-interval-style caution without
  needing scipy).

A `--psro_solver {prioritized,nash}` flag reserves the v3 path: `nash` builds a
pool-vs-pool matrix (requires extra eval games — out of scope for v1) and calls
`solve_zero_sum_equilibrium`. Ship `prioritized` only; the flag is a placeholder.

### Transport (how weights reach the env)

No object plumbing across the env boundary. A **sidecar JSON file**:

```
{buffer_dir}/{format}/meta_weights.json   # {agent_name: weight}
```

- **Writer**: the collector process, in
  `MetamonOnlineExperiment.collect_new_training_data` (overridden), after
  `super().collect_new_training_data()` and only when `--psro_weighting` is on
  **and `self.epoch >= psro_start_epoch`**. Writes atomically (`.tmp` →
  `rename`). Before the start epoch the sidecar is not written, so the reader
  sees no file and stays uniform — clean gated switchover.
- **Reader**: `ConfigBatchedOpponent.configure` (collection) and
  `MetamonFIFODataset` (learner's replay sampling) both read the sidecar; each
  falls back to uniform if missing/stale/unparseable. Cache the parsed weights;
  re-read only if mtime changed.

This keeps the change localized and survives the existing async
collect/learn/validate split cleanly (the validator uses its own pool and is
untouched).

### Update cadence

Every `--psro_update_interval` epochs (default 5), and only once
`self.epoch >= --psro_start_epoch`. Updating every epoch is fine mechanically
but wasteful (scan the buffer dir each time) and noisier. The EMA already
smooths within-update noise. **At the start epoch itself, force one immediate
update** so the sidecar exists from the first prioritized collection epoch
rather than appearing up to `update_interval` epochs late.

---

## Concrete changes

### 1. `metamon/rl/evaluate/opponent_pool.py` — weighted sampling

Add an optional weights vector to `OpponentPoolConfig`:

```python
class OpponentPoolConfig:
    def __init__(self, agents, battle_format, rng=None, weights=None):
        ...
        self._weights = None  # None ⇒ uniform (current behavior)
        self.set_weights(weights)  # validates length, normalizes

    def set_weights(self, weights):
        # weights aligned with self.agents; None ⇒ uniform
        # store as list[float] aligned to agent rows
        ...

    def sample_opponent(self) -> PolicySpec:
        if self._weights is None:
            name, merged = self.rng.choice(self.agents)
        else:
            name, merged = self.rng.choices(self.agents, weights=self._weights, k=1)[0]
        return sample_policy_from_merged(name, merged)
```

`parse_opponent_pool_dict` / `from_dict` unchanged (uniform by default), so all
existing call sites (validation, ladder) are untouched.

### 2. `metamon/env/vectorized/opponent.py` — read sidecar in `configure`

In `ConfigBatchedOpponent`:

- `__init__`: accept `weights_path: Optional[str] = None`.
- `configure`: if `weights_path` set and its mtime is newer than last read,
  `json.load` it, map `agent_name → weight`, align to `self.config.agents` by
  the agent base name (`agents[i][0]`). **Decision: key weights by the agent
  base name** (the `name` field in each row), which is what the pool samples
  over and is stable across checkpoint/temp draws. The filename's `short_label`
  is per-spec; we aggregate win rates over all specs of an agent for v1
  (simpler, and within-agent variance is lower than across-agent). v2 can go
  per-spec.
- Call `self.config.set_weights(aligned_weights)` (falling back to uniform on
  any error / missing agent).

Plumb `weights_path` through `make_metamon_env` / `ShowdownEnv` as one new
optional kwarg defaulting to `None` (no behavior change for val/ladder).

### 3. `metamon/rl/online_rl.py` — CLI + wiring

New CLI args (all default-off ⇒ current behavior identical unless explicitly
enabled, so a fresh run with no `--psro_*` flags is byte-for-byte today):

```
--psro_weighting            (flag, off by default)
--psro_start_epoch INT      (default 0; the live run sets 1000)
--psro_temp FLOAT           (default 1.0)
--psro_floor FLOAT          (default 0.05)
--psro_min_games INT        (default 20)
--psro_window INT           (default 50000, # most-recent files to score)
--psro_update_interval INT  (default 5 epochs)
--psro_ema FLOAT            (default 0.7)
--psro_fifo_reweight        (flag, off by default — see "Buffer lag"; enable
                             for the live switchover)
--psro_buffer_trim INT      (default None; if set, evict the FIFO down to this
                             many files once at psro_start_epoch to accelerate
                             turnover of the uniform-sampled backlog)
```

In `_make_collect_train_env`, pass `weights_path =
os.path.join(buffer_dir, battle_format, "meta_weights.json")` when
`--psro_weighting` is on (else `None`). The learner and validator processes are
launched with the same args for config symmetry, but only the collector writes
the sidecar; the learner reads it via its `MetamonFIFODataset`
(`--psro_fifo_reweight`); the validator uses its own pool and is unaffected.

### 4. `metamon/rl/metamon_to_amago.py` `MetamonOnlineExperiment` — the updater

New small module `metamon/rl/psro_lite.py` (keeps the experiment file clean):

```python
def compute_prioritized_weights(
    *, buffer_dir, battle_format, agent_names, window, min_games,
    temp, floor, ema, prev_weights,
) -> tuple[dict[str, float], dict[str, dict]]:
    # 1. list .json.lz4 in {buffer_dir}/{format}/, take last `window` by mtime
    # 2. regex opponent + WIN/LOSS out of each filename
    #    (reuse the _save_lane_outcome filename format as the parse spec)
    # 3. aggregate per agent-name: n, wins → p_o (Laplace)
    # 4. score_o = max(0, 0.5 - p_o); w_o = score_o**(1/temp); floor; normalize
    # 5. EMA with prev_weights (uniform if None)
    # 6. min_games gate → uniform for cold opponents
    # return ({agent_name: weight}, diagnostics {agent: {n, win_rate, weight}})
```

In `MetamonOnlineExperiment`:
- `__init__`: accept a `psro_config` dataclass (or None) holding the CLI knobs +
  `buffer_dir`, `battle_format`, `agent_names`, `start_epoch`.
- Override `collect_new_training_data`:
  ```python
  def collect_new_training_data(self):
      super().collect_new_training_data()
      psro = self._psro
      if psro is None or self.epoch < psro.start_epoch:
          return
      # force one update on the very first prioritized epoch so the sidecar
      # exists immediately, then respect update_interval thereafter.
      first = self.epoch == psro.start_epoch
      if first or self.epoch % psro.update_interval == 0:
          psro.step(epoch=self.epoch)  # scan buffer, compute weights, write sidecar
  ```
- `step()` keeps `prev_weights` in memory for the EMA (initialized to uniform at
  the first call) and writes `meta_weights.json` atomically. Also logs
  per-opponent `n / win_rate / weight` + `weight_entropy` to wandb under a
  `psro/` panel — this is the **primary signal** that the switchover is behaving
  (see Monitoring).

`MetamonFIFODataset.on_end_of_collection` is **not** touched for writing; it
already calls `refresh_files()` every epoch, which is the hook the reweighting
below latches onto.

### 5. `metamon/rl/metamon_to_amago.py` `MetamonFIFODataset` — per-trajectory reweighting (v1, gated by `--psro_fifo_reweight`)

This is the direct fix for buffer lag and is shipped in v1 specifically because
there is no A/B to catch a lag-induced regression on the live run.

- `MetamonFIFODataset` already wraps a `MetamonDataset` (parsed-replay) whose
  files are the `{...}_vs_{opponent}_{WIN|LOSS}.json.lz4` trajectories.
- Add an optional `opponent_weight_provider: Callable[[], dict[str, float]] | None`.
  When set, `sample_random_trajectory` (and the indexed-file sampling path)
  draws a file in proportion to the current per-opponent weight instead of
  uniformly. Concretely: maintain an aligned `(filepaths, weights)` array
  rebuilt on each `refresh_files()` (already called every epoch in
  `on_end_of_collection`), and use `np.random.choice` over it.
- The weight provider reads the same `meta_weights.json` sidecar (keyed by agent
  name) and maps each file to its opponent's agent name by parsing the filename
  once at refresh time (cache the parse). Files whose opponent isn't in the
  sidecar get weight = uniform fallback (1/N).
- **Gated**: only active when `--psro_fifo_reweight` is passed; otherwise
  `sample_random_trajectory` is unchanged (today's uniform behavior).

This makes the *learner's* online 40% mixture track the meta-distribution as
fast as the buffer refreshes, not as slow as `dset_max_size` eviction. It is the
off-policy PSRO hybrid flagged by Offline FSP (arXiv:2403.00841); we accept that
regime because the alternative (lag) is worse and unmonitorable without an A/B.

---

## Buffer lag (the acute risk for a mid-run switchover)

At epoch 1000 the FIFO is full of uniformly-sampled games. Three levers, in
order of importance, make the switchover safe without an A/B:

1. **`--psro_fifo_reweight` (ship in v1)** — the learner's online mixture tracks
   the meta-dist on every buffer refresh instead of waiting for eviction. This
   is the real fix and is described in change #5 above.
2. **`--psro_buffer_trim` (optional, recommended for the live flip)** — at the
   start epoch, evict the FIFO down to a smaller size (e.g. trim 300k → 50k) so
   the backlog of uniform-sampled games turns over in ~50k collected games
   rather than ~300k. Trades a one-time data-volume dip (the offline 60% of the
   mixture is untouched, so the learner isn't starved) for a much faster
   convergence of the online 40% to the new distribution. Cheap and reversible
   (just raise `--dset_max_size` back on a future relaunch).
3. **`online_anneal_epochs` already ramps the online weight from 0** — but note
   on a *resume* the buffer is already full and AMAGO's legacy path sets
   `initial_weight=final_weight` for ready datasets. If we want the online
   fraction itself to re-ramp after the flip (so the learner leans on the
   fresher prioritized data gradually), relaunch with a fresh
   `--online_anneal_epochs` value. This is orthogonal to PSRO and optional.

## Stability & safety

- **Default-off + start-epoch gate**: every flag defaults to current uniform
  behavior; no existing or fresh run changes unless `--psro_weighting` is passed
  *and* `self.epoch >= --psro_start_epoch`.
- **Fallback to uniform** on any sidecar read error, missing file, empty buffer,
  or all-zero scores. The env and the FIFO sampler never hard-fail on weighting.
- **Diversity floor** prevents collapse onto one opponent (the main failure mode
  of greedy prioritization — well documented in the PSRO survey
  arXiv:2403.02227 and Conflux-PSRO arXiv:2410.22776).
- **EMA + min-games gate** prevent thrashing from per-epoch noise.
- **No learner-loss change**: the FIFO mixture ratios
  (`build_online_mixture_dataset`, 40/60 online/offline, `online_anneal_epochs`
  ramp) are untouched. The only thing that changes is *which opponents the
  online 40% comes from* and *how the online 40% is sampled within the FIFO*.
- **Clean revert**: relaunching without `--psro_weighting` returns the run to
  byte-for-byte today's behavior (the sidecar is simply never written/read
  again). No persistent state beyond the sidecar JSON, which can be deleted.

## Known limitations (acceptable for v1)

1. **Aggregate per agent, not per spec**: opponent `ckpt`/`temp` variants of the
   same agent share a weight in v1. Fine because within-agent variance << across.
2. **No Nash**: `prioritized` targets exploitability, not equilibrium. For
   non-transitive cycles this is weaker than Nash but still strictly better than
   uniform (Self-Play PSRO arXiv:2207.06541 shows uniform is exploitable). v3
   can flip `--psro_solver nash` once pool-vs-pool eval is wired.
3. **No controlled comparison**: with no A/B we cannot measure the *causal*
   effect of the switch at epoch 1000. Monitoring (below) tells us it isn't
   *broken*; it cannot prove it's *better*. That's an accepted cost of the
   wall-clock constraint.

## Monitoring & switchover (no A/B)

Since there is no baseline run to compare against, the wandb `psro/` panel is
the primary safety signal. Log every update:

- per opponent: `n_games_in_window`, `win_rate`, `weight` — confirm weights are
  concentrating on opponents the learner actually loses to, the floor is
  preventing collapse onto one opponent, and `n` is above `min_games`.
- `weight_entropy` (−Σ w log w) — a collapse to one opponent shows up as a
  sharp drop; alert if it goes below a threshold (e.g. log(2) for a 2-agent
  pool).
- `sidecar_write_ok` / `sidecar_read_ok` — booleans catching transport bugs.
- the existing `val/` win-rate panel (already logged by `--eval_during_training`)
  is the outcome signal: watch for a step-change at epoch 1000. A flat or rising
  val win rate says the switchover didn't break anything; a sustained drop says
  roll back (relaunch without `--psro_weighting`).

**Switchover procedure for the live run at epoch 1000** (the run is already
underway, so this is a resume, not a fresh start):

1. Let the current run reach epoch 1000 and checkpoint as usual
   (`--ckpt_interval` / `always_save_latest`).
2. Relaunch the **collector** process with the existing resume flags
   (`--prev_run_dir`/`--prev_checkpoint`, or `--resume_training_state` for the
   learner) **plus**:
   ```
   --psro_weighting --psro_start_epoch 1000 --psro_fifo_reweight \
   --psro_buffer_trim 50000
   ```
   (tune `--psro_buffer_trim` to taste; 50k ≈ 1/6 of the default 300k backlog.)
3. Relaunch the **learner** with `--psro_fifo_reweight` (it reads the sidecar
   via its `MetamonFIFODataset`) and resume from the epoch-1000 checkpoint so it
   doesn't restart training. The other `--psro_*` collection flags are no-ops on
   the learner (placeholder envs) but harmless to pass for config symmetry.
4. The **validator** is relaunch**ed** unchanged — it uses its own pool and is
   not affected by any `--psro_*` flag.
5. Watch the `psro/` and `val/` panels for the first ~`--psro_update_interval`
   × a few epochs. If `weight_entropy` collapses or `val/` win rate drops
   sustainedly, relaunch without `--psro_weighting` to revert (the run is
   otherwise unchanged, so revert is clean).

**Pre-flight (cheap, does not touch the live run):**

1. Unit test `compute_prioritized_weights` on a fake buffer dir of hand-named
   `.json.lz4` files asserting the win-rate→weight transform, EMA, floor,
   min-games gate, and uniform fallback when the buffer is empty.
2. Offline smoke: point `compute_prioritized_weights` at the **current live
   buffer** read-only and print the weights it *would* produce at epoch 1000.
   This validates the filename regex against real filenames and shows the
   starting weight distribution before we commit — no relaunch needed.
3. Sidecar round-trip test: write a synthetic `meta_weights.json`, confirm
   `ConfigBatchedOpponent.configure` parses it and
   `OpponentPoolConfig.set_weights` produces the expected sampling distribution
   over 10k draws.

## Staging

- **v1** (this plan): prioritized collection **+ per-trajectory FIFO
  reweighting** + sidecar JSON, all default-off, gated by `--psro_start_epoch`.
  The FIFO reweighting was promoted from v2 because the live mid-run switchover
  with no A/B makes buffer lag an acute, unmonitorable-without-it risk.
- **v2**: per-*spec* weighting (split an agent's weight across its
  checkpoint/temp variants by their individual win rates) — only worth it if
  within-agent variance turns out to matter (check the `psro/` panel).
- **v3**: `--psro_solver nash` over a pool-vs-pool matrix (reuse
  `build_payoff_matrix` + `solve_zero_sum_equilibrium`); adds a periodic eval
  sweep — only worth it if v1 plateaus.

---

## File touch summary

| File | Change |
|---|---|
| `metamon/rl/evaluate/opponent_pool.py` | `OpponentPoolConfig.set_weights` + weighted `sample_opponent` |
| `metamon/env/vectorized/opponent.py` | `ConfigBatchedOpponent` reads sidecar JSON in `configure` |
| `metamon/env/vectorized/vector_env.py` | thread `weights_path` kwarg through `make_metamon_env`/`ShowdownEnv` (default `None`) |
| `metamon/rl/psro_lite.py` | **new**: `compute_prioritized_weights` + the `PsroLite` stateful updater |
| `metamon/rl/metamon_to_amago.py` | `MetamonOnlineExperiment` accepts `psro_config`, overrides `collect_new_training_data`; `MetamonFIFODataset` gains optional `opponent_weight_provider` for per-trajectory reweighting |
| `metamon/rl/online_rl.py` | CLI flags + wire `weights_path`, `psro_config`, and `opponent_weight_provider` into the collector/learner experiment |
| `tests/` | unit test for `compute_prioritized_weights`; sidecar round-trip test; filename-regex test against real buffer samples |

No changes to: the learner's loss, `build_online_mixture_dataset` mixture
ratios, the offline dataset, validation, the AMAGO training loop, or the
team-construction pipeline.
