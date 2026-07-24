# Teamset Curriculum Proposal: Migrating from `gl_05_26` to a `smogon_pass2` / `smogon_pass2_selected` mix

## Status of the current run

| Component | Current value |
|---|---|
| Run | `mini_online_psro_v1` (PSRO-Lite split layout: learner + collector + validator in tmux) |
| Epoch | ~920 / 3950 target |
| Train team set (`--train_team_set`) | `gl_05_26` — **29,144 teams** (broad general-ladder sample, May 2026) |
| Opponent pool team set (`hl_gen1ou.yaml` defaults) | `gl_05_26` — same broad set, used by every opponent in the pool |
| Val team set (`--val_team_set`) | `competitive` (human-made Smogon forum teams) |
| Dataset | `online_selfplay.yaml`: 60% offline (pac-base/pac-exploratory/pac-tauros) + 40% online FIFO, **5% replay weight** |

### Target data — two competitive sets with distinct roles

| Set | Teams | Role in the curriculum | Description |
|---|---|---|---|
| `smogon_pass2` | 475 | Competitive **breadth** | Teams used by the strongest battlers on Showdown — the full range of what strong players bring, including off-meta experimentation and lead choices that didn't necessarily pan out. |
| `smogon_pass2_selected` | 283 | Competitive **sharpening** | Curated subset: every team has 246–617 logged battles and a win rate of 0.49–0.615, with per-lead variations (`_v2`, `_v3`, …) of the same roster. Battle-validated winners — the specialization target. |

Both are used throughout (not as alternatives). `smogon_pass2` is introduced
first and kept at a permanent floor so the model learns the *full* competitive
metagame before and alongside concentrating on the proven winners in
`smogon_pass2_selected`.

## The architectural constraint that shapes the plan

The team set is **loaded once at experiment creation** — `_make_collect_train_env()`
calls `get_metamon_teams(battle_format, team_set_name)`, producing a `TeamSet`
object that is embedded in the `make_metamon_env` partial. That partial is
instantiated during `experiment.init_envs()` and the resulting `TeamSet` persists
for the life of the process. `TeamSet.yield_team()` does a uniform
`random.choice(self.team_files)` on every battle reset, but the *file list* is
fixed at startup.

Crucially: **the team set is not part of the saved training state.** It is
reconstructed from CLI args (`--train_team_set`) every time the process starts.
This means the natural, low-risk way to change the team distribution is a
**phased restart**: stop the run, change the team-set argument, and resume from
the latest `training_states/<run>_epoch_<N>/` via `--resume_training_state`.

The FIFO buffer is append-only and the 5% replay weight means old `gl_05_26`
trajectories persist in the training mix for a long time after the team set
changes — a built-in (if mild) anti-forgetting mechanism.

## Methodology: phased three-way weighted-mix curriculum with permanent anti-forgetting floors

### Principle

Never go 100% on any single set. The competitive specialization is split across
two sets with distinct roles: `smogon_pass2` (475 teams — everything strong
players bring, including off-meta experimentation) provides competitive
*breadth*, while `smogon_pass2_selected` (283 teams — battle-validated winners,
win rate 0.49–0.615) provides competitive *sharpening*. The curriculum ramps
both competitive sets up over several restart boundaries while the broad
`gl_05_26` fraction shrinks, but `gl_05_26` never drops below a ~15% maintenance
floor and `smogon_pass2` never drops below ~25%. This is the RL analogue of
**experience replay for team diversity** — the policy keeps seeing the broad
distribution and the full competitive breadth while increasingly concentrating
on proven winners.

### Why a weighted mix instead of a merged directory

The three sets differ wildly in size: `gl_05_26` has 29,144 files,
`smogon_pass2` has 475, and `smogon_pass2_selected` has 283. A naive directory
merge would sample competitive teams ~2% of the time — useless. To get a
meaningful competitive weight you would need to duplicate the small files dozens
of times, wasting disk and locking the ratio until you rebuild the directory. A
`WeightedMixedTeamSet` that holds N underlying `TeamSet` objects and samples
between them with explicit weights is cleaner, uses zero extra disk, and lets you
set all three ratios from an env var on each restart.

### The schedule

Each row is one restart boundary. The run is stopped, the env vars below are set,
and the run is resumed from the latest training state. The epoch ranges are
illustrative — adjust based on validation signals (see Monitoring).

| Phase | Epochs (approx) | `gl_05_26` | `smogon_pass2` | `smogon_pass2_selected` | Rationale |
|---|---|---|---|---|---|
| 0 (done) | 0–920 | 100% | 0% | 0% | Broad coverage; learn the game. |
| 1 — acquaint | 920–1400 | 75% | 15% | 10% | Gentle intro; both competitive sets trickle in. `smogon_pass2` leads slightly (breadth-first). |
| 2 — broaden | 1400–2000 | 45% | 35% | 20% | `gl_05_26` drops below half; `smogon_pass2` is the larger competitive component — broaden competitive experience before sharpening. |
| 3 — sharpen | 2000–2800 | 25% | 30% | 45% | `smogon_pass2_selected` overtakes `smogon_pass2`; start concentrating on proven winners. |
| 4 — maintain | 2800–3950 | 15% | 25% | 60% | Final: majority proven winners, with permanent floors on broad-ladder (15%) and competitive-breadth (25%) distributions. |

Phase 4 leaves **two permanent floors**: 15% `gl_05_26` (broad-ladder
anti-forgetting — ~1 in 7 battles stays against the general distribution) and
25% `smogon_pass2` (competitive-breadth retention — the model keeps seeing the
full range of what strong players bring, not just the proven winners). Only 60%
of battles in the final phase come from `smogon_pass2_selected`. Combined with
the 5% replay weight in the dataset config (old `gl_05_26` trajectories persist
in the FIFO buffer) and the continued presence of broad-team opponents in the
pool, this gives multiple independent channels of non-specialized exposure.

### What changes at each boundary

Three things must move together so that the self-play distribution is coherent:

1. **Player team set** — `--train_team_set` → a weighted mix of `gl_05_26` +
   `smogon_pass2` + `smogon_pass2_selected` (via the new
   `WeightedMixedTeamSet`, see Implementation).

2. **Opponent pool team set** — `hl_gen1ou.yaml` currently defaults every
   opponent to `team_set: gl_05_26`. If the player starts bringing competitive
   teams but opponents keep bringing broad-ladder teams, the agent learns to
   *pilot* competitive teams against broad opponents — useful, but it never
   learns to *play against* competitive teams. To get both sides:
   - Add per-agent `team_set` overrides in `hl_gen1ou.yaml` so a subset of the
     pool uses `smogon_pass2` / `smogon_pass2_selected`.
   - Or (simpler, first pass) change the pool's `defaults: team_set` to the same
     weighted mix used for the player. This makes both sides of every self-play
     battle draw from the same three-way distribution, which is the cleanest
     signal.

3. **Validation team set** — keep `--val_team_set competitive` unchanged (it is
   the held-out eval distribution and should stay fixed for comparability across
   phases). **Add guard validation tracks against `gl_05_26` and (optionally)
   `smogon_pass2`** to detect broad-ladder forgetting and competitive-breadth
   over-fitting respectively (see Monitoring).

### Monitoring: validation tracks

The single `competitive` validation track tells you whether the model is
improving against held-out competitive teams. It does **not** tell you whether
the model is forgetting how to play against the broad ladder, nor whether it is
over-fitting to the curated winners. You need all three:

| Track | Team set | What it measures | Action if it drops |
|---|---|---|---|
| **Primary** | `competitive` (existing) | Specialization progress | Expected to rise; the whole point. |
| **Forgetting guard** | `gl_05_26` (new) | Broad-ladder retention | If it drops >X pts below its Phase-0 baseline, hold the current phase longer or **increase** the `gl_05_26` weight before advancing. |
| **Competitive-breadth guard** (optional) | `smogon_pass2` (new) | Competitive-metagame generality | If it diverges downward from the `competitive` track late in training, the model is over-fitting to `smogon_pass2_selected`; **increase** the `smogon_pass2` weight. |

Each guard track is a separate validator process pointed at its own team-set val
pool (the existing split-layout validator pattern makes this trivial — launch
additional `validator` processes with `VAL_TEAM_SET=gl_05_26` and
`VAL_TEAM_SET=smogon_pass2` respectively, each with a separate opponent pool).
All log to wandb as separate runs, so you get multiple win-rate curves to watch.
Note that `smogon_pass2` is *in* the training mix, so its guard track is not a
clean held-out eval — it is a diagnostic of how the model performs against the
broader competitive set relative to the selected winners, useful for spotting
late-stage over-fitting to `smogon_pass2_selected`.

### Why this avoids catastrophic forgetting

Four independent mechanisms, any one of which provides partial protection:

1. **Persistent broad-team sampling** — the 15% floor means the policy never
   stops seeing `gl_05_26` teams, on both sides of the board, for the rest of
   training. The gradient signal for broad-distribution play never goes to zero.

2. **FIFO replay buffer** — the 5% `replay_weight` in `online_selfplay.yaml`
   means the learner keeps sampling old `gl_05_26`-era trajectories from the
   buffer for many epochs after a phase transition. This is a temporal smoothing
   layer on top of the spatial (team-set) mix.

3. **Opponent pool diversity** — the pool already contains 7 distinct
   architectures (TaurosV0, Kakuna, Alakazam, SyntheticRLV2, V2ADataAblation,
   Superkazam, Kadabra3) plus the PSRO past-self. These opponents were trained on
   broad distributions and play diverse strategies. Even if their *teams* shift
   towards competitive, their *policies* retain broad-distribution behavior.

4. **Competitive-breadth retention** — the 25% `smogon_pass2` floor in Phase 4
   prevents over-specialization on the curated winners alone. Without it, the
   model could over-fit to the 283 `smogon_pass2_selected` teams and lose
   generality *within* the competitive metagame (e.g. fail against off-meta
   competitive picks that `smogon_pass2` covers but the selected set does not).

## Implementation

### 1. `WeightedMixedTeamSet` — a `TeamSet` subclass

Add to `metamon/env/wrappers.py`:

```python
class WeightedMixedTeamSet(TeamSet):
    """Sample teams from multiple underlying TeamSets with configurable weights.

    Unlike a merged directory (which fixes the ratio by file count and wastes
    disk duplicating small sets), this holds N TeamSet objects and picks one
    per battle according to ``weights``. Each underlying TeamSet then does its
    own uniform ``yield_team()`` draw.

    Args:
        team_sets: List of TeamSet objects to mix.
        weights:   Per-set sampling weights (need not sum to 1; normalized
                   internally). Must be same length as ``team_sets``.
    """

    def __init__(self, team_sets: list[TeamSet], weights: list[float]):
        # Deliberately skip TeamSet.__init__ (which scans a single directory).
        # We still need Teambuilder.__init__ for poke-env's team parsing.
        from poke_env.teambuilder.teambuilder import Teambuilder
        Teambuilder.__init__(self)
        if len(team_sets) != len(weights):
            raise ValueError("team_sets and weights must have equal length")
        if not team_sets:
            raise ValueError("WeightedMixedTeamSet requires at least one TeamSet")
        self.team_sets = team_sets
        total = sum(weights)
        self.weights = [w / total for w in weights]
        self.battle_format = team_sets[0].battle_format
        self._most_recent_team_file = None

    def yield_team(self) -> str:
        idx = random.choices(range(len(self.team_sets)), weights=self.weights)[0]
        team = self.team_sets[idx].yield_team()
        self._most_recent_team_file = self.team_sets[idx].most_recent_team_file
        return team

    def block_team(self, packed_team: str) -> bool:
        return any(ts.block_team(packed_team) for ts in self.team_sets)
```

### 2. CLI / env-var integration

Add a `--train_team_mix` option (or parse a special syntax in `--train_team_set`)
that lets the launch script specify the weighted mix without hardcoding it. The
simplest approach that fits the existing env-var pattern in
`launch_mini_online_v1.sh`:

```bash
# New env vars (all optional; if unset, fall back to --train_team_set single-set behavior)
TRAIN_TEAM_MIX="${TRAIN_TEAM_MIX:-}"          # e.g. "gl_05_26:0.45,smogon_pass2:0.35,smogon_pass2_selected:0.20"
```

In `online_rl.py`, if `TRAIN_TEAM_MIX` is provided, `_make_collect_train_env`
builds a `WeightedMixedTeamSet` from the named sets instead of calling
`get_metamon_teams` for a single set. The same env var can be forwarded to the
opponent-pool team-set resolution so both sides of self-play stay in sync.

### 3. Opponent pool update

Edit `metamon/rl/configs/opponent_pools/hl_gen1ou.yaml`:

```yaml
defaults:
  team_set: gl_05_26           # keep as the fallback / broad-distribution default
  battle_backend: metamon
  checkpoints: [null]
  temperatures: [1.0, 1.25, 1.5, 1.75, 2.0]
  num_agents: 1
```

To shift opponents towards competitive teams, either:
- **(simplest)** point `defaults: team_set` at the same mix name used for the
  player (requires the mix to be resolvable by `get_metamon_teams` — see below),
- **(per-agent)** add `team_set: smogon_pass2_selected` to specific agents (e.g.
  the PSRO past-self and TaurosV0) so a known fraction of the pool uses
  competitive teams while the rest stays broad.

### 4. Operational steps for each phase transition

```bash
# 0. Set the CPU governor (collection is CPU-sim-bound)
sudo cpupower frequency-set -g performance

# 1. Stop the current tmux session
tmux kill-session -t mini_online_psro_v1

# 2. Set the mix for the new phase (example: Phase 2 — broaden)
export EPOCHS=3950                          # MUST exceed resumed epoch
export TRAIN_TEAM_MIX="gl_05_26:0.45,smogon_pass2:0.35,smogon_pass2_selected:0.20"
# Opponent pool: edit hl_gen1ou.yaml defaults or per-agent team_set to match.

# 3. Relaunch the split layout (each role resumes from latest training_state)
bash scripts/launch_psro_v1.sh
# (The learner forwards --resume_training_state via --prev_run_dir internally;
#  collector + validator sync to the existing latest/policy.pt.)
```

The training state (model + optimizer + scheduler + PopArt + RNG) is restored
from the newest `training_states/<run>_epoch_<N>/`, so the optimizer momentum and
learning-rate schedule continue uninterrupted. The only thing that changes is
the team distribution the collector samples from — exactly what we want.

## Open questions / decisions for the user

1. **Relative growth of the two competitive sets** — the schedule above has
   `smogon_pass2` lead early (breadth-first) and `smogon_pass2_selected`
   dominate late (sharpen on proven winners). An alternative is to grow them in
   lockstep (e.g. Phase 2: 45/27.5/27.5, Phase 4: 15/42.5/42.5) if you don't
   want to privilege breadth-before-sharpening. The breadth-first ordering is
   safer — it avoids over-fitting to 283 teams before the model has seen the
   full competitive range — but the lockstep version converges faster if the
   `competitive` validation track is the only signal you care about.

2. **Phase length** — the ~500-epoch phase widths above are a guess. A
   data-driven alternative: advance to the next phase when the `competitive`
   validation win rate plateaus for ~100 epochs AND the `gl_05_26` forgetting
   guard has not dropped below its Phase-0 baseline.

3. **Opponent pool strategy** — simplest first pass is to point the pool's
   `defaults: team_set` at the same mix as the player. A more refined approach
   keeps some opponents permanently on `gl_05_26` so the agent always faces a
   fraction of broad-team opponents even in Phase 4. This is the
   "opponent-channel" anti-forgetting lever and is worth doing if the forgetting
   guard shows regression.

4. **Should the offline piles (pac-base/pac-exploratory/pac-tauros) also shift?**
   — Those are pre-generated and static; they were collected against broad
   distributions. Leaving them as-is adds another anti-forgetting channel (the
   60% offline mix always contains broad-distribution trajectories). No change
   recommended unless the forgetting guard fails.

---

## Migration guide: starting Phase 1 on the current run

The code changes are implemented and verified. Here is how to transition the
live `mini_online_psro_v1` run (currently at ~epoch 933) into Phase 1.

### What was implemented

| File | Change |
|---|---|
| `metamon/env/wrappers.py` | `WeightedMixedTeamSet` class + `parse_team_mix_spec` / `is_team_mix_spec` / `get_metamon_team_mix` / `get_metamon_team_set_or_mix` helpers. |
| `metamon/env/__init__.py` | Re-exports the new symbols. |
| `metamon/rl/evaluate/opponent_pool.py` | `team_set_for()` now dispatches on mix specs — a `team_set:` value containing `":"` is parsed as a mix instead of a directory name. |
| `metamon/rl/online_rl.py` | `--train_team_mix` / `--val_team_mix` CLI args; threaded through `_make_collect_train_env`, `_make_val_env`, and `create_online_experiment`. |
| `scripts/launch_mini_online_v1.sh` | `TRAIN_TEAM_MIX` / `VAL_TEAM_MIX` env vars forwarded as `--train_team_mix` / `--val_team_mix` when set. |
| `scripts/launch_psro_v1.sh` | Same env vars forwarded into all three tmux windows (learner, collector, validator). |

### Phase 1 step-by-step

Phase 1 target: 75% `gl_05_26` / 15% `smogon_pass2` / 10% `smogon_pass2_selected`.

**Step 0 — set the CPU governor** (collection is CPU-sim-bound):
```bash
sudo cpupower frequency-set -g performance
```

**Step 1 — update the opponent pool to use the same mix.**
Edit `metamon/rl/configs/opponent_pools/hl_gen1ou.yaml` — change the default
`team_set` from the bare name to the quoted mix spec (the quotes are required;
the colons would otherwise be read as YAML key-value separators):
```yaml
defaults:
  team_set: "gl_05_26:0.75,smogon_pass2:0.15,smogon_pass2_selected:0.10"
  battle_backend: metamon
  checkpoints: [null]
  temperatures: [1.0, 1.25, 1.5, 1.75, 2.0]
  num_agents: 1
```
This makes every opponent in the pool draw teams from the same three-way
distribution as the player. Both sides of every self-play battle see the mix.

**Step 2 — stop the current run.**
```bash
tmux kill-session -t mini_online_psro_v1
```
The learner's newest full training state
(`ckpts/training_states/mini_online_psro_v1_epoch_930/`) and the rolling
`latest/policy.pt` survive on disk and are the resume source.

**Step 3 — relaunch with the Phase 1 mix.**
```bash
EPOCHS=3950 \
TRAIN_TEAM_MIX="gl_05_26:0.75,smogon_pass2:0.15,smogon_pass2_selected:0.10" \
  bash scripts/launch_psro_v1.sh
```
`EPOCHS=3950` must exceed the resumed epoch (~930) or the learner exits
immediately ("0 epochs remaining"). The `launch_psro_v1.sh` script forwards
`TRAIN_TEAM_MIX` into all three tmux windows; each role picks it up via the
`launch_mini_online_v1.sh` → `--train_team_mix` → `online_rl.py` chain.

The learner resumes from `training_states/..._epoch_930/` (full accelerate
state: model + optimizer + scheduler + PopArt + RNG). The collector and
validator sync to the existing `latest/policy.pt` and keep going. The only thing
that changes is the team distribution — exactly what we want.

**Step 4 — verify the mix loaded.** Check the learner log for the
`WeightedMixedTeamSet:` banner:
```bash
grep 'WeightedMixedTeamSet' ~/metamon_runs/psro_learner.log
# Expected: WeightedMixedTeamSet: .../gl_05_26/gen1ou=75%, .../smogon_pass2/gen1ou=15%, .../smogon_pass2_selected/gen1ou=10%
```
The collector and validator logs will show the same banner (each role builds its
own `WeightedMixedTeamSet` at startup).

### Adding the forgetting-guard validator (recommended)

The existing validator evaluates against `competitive` teams (TaurosV0
opponent). To detect broad-ladder forgetting, launch a second validator that
evaluates against `gl_05_26` teams in a separate tmux session:
```bash
EPOCHS=3950 VAL_TEAM_SET=gl_05_26 RUN_NAME=mini_online_psro_v1 \
  bash scripts/launch_mini_online_v1.sh validator --log \
    --prev_run_dir ~/metamon_runs/mini_online_v1 \
    --prev_run_name mini_online_v1 --prev_checkpoint 700 \
  2>&1 | tee ~/metamon_runs/psro_forgetguard_validator.log &
```
This logs a separate wandb run with win-rate-vs-`gl_05_26` curves. Compare it to
the Phase-0 baseline (the validator's win rate before the mix change) — if it
drops more than a few points, hold Phase 1 longer before advancing to Phase 2.

### Phase transition cheat sheet

| Phase | `TRAIN_TEAM_MIX` value | `hl_gen1ou.yaml` default `team_set` |
|---|---|---|
| 1 | `gl_05_26:0.75,smogon_pass2:0.15,smogon_pass2_selected:0.10` | same |
| 2 | `gl_05_26:0.45,smogon_pass2:0.35,smogon_pass2_selected:0.20` | same |
| 3 | `gl_05_26:0.25,smogon_pass2:0.30,smogon_pass2_selected:0.45` | same |
| 4 | `gl_05_26:0.15,smogon_pass2:0.25,smogon_pass2_selected:0.60` | same |

At each transition: stop tmux → edit `hl_gen1ou.yaml` → `tmux kill-session` →
relaunch `launch_psro_v1.sh` with the new `TRAIN_TEAM_MIX`. The learner always
resumes from the newest training state; no data is lost.
