---
name: playtest-ladder
description: "Register a local training checkpoint as a pretrained model and battle it on the local Showdown ladder (metamon.rl.evaluate --eval_type ladder). Use when the user asks to playtest, play against, test, or evaluate a policy against themselves or other bots on the local showdown instance, register a new local model in pretrained.py, or use the --checkpoints -1 latest-checkpoint sentinel. Covers LocalPretrainedModel vs LocalFinetunedModel registration, the LATEST_CHECKPOINT mechanism, local server startup, and the evaluate CLI ladder flags."
---

# Playtest on the Local Showdown Ladder

Battle a local training checkpoint against yourself (a human logged into the
local Showdown web client) or other bots, using `metamon.rl.evaluate
--eval_type ladder`. The policy loads weights once at startup and queues for
ladder matches; training can continue uninterrupted (it only reads weights).

## Prerequisites

### 1. The local Showdown server must be running

The repo vendors a Showdown checkout at `server/pokemon-showdown/`. Start it
once (it keeps running across shell exits; wrap in tmux/systemd for reboots):

```bash
cd ~/repos/metamon/server/pokemon-showdown
node pokemon-showdown start --no-security
```

It listens on `ws://localhost:8000` (`LocalhostServerConfiguration`). Open
`http://localhost:8000` in a browser to log in as a human player (choose any
username; no password needed with `--no-security`). The bot connects to the
same server and queues for ladder matches — you'll be matched against it.

Check it's up:
```bash
pgrep -af 'pokemon-showdown start'   # should list the node process
ss -ltnp | grep ':8000'              # should show port 8000 listening
```

### 2. `METAMON_CACHE_DIR` must be set

The evaluate CLI and `pretrained.py` require it. The canonical location on
this machine:
```bash
export METAMON_CACHE_DIR=/home/eddie/metamon_cache
```
Team sets are resolved from here (`$METAMON_CACHE_DIR/teams/<set_name>` for
custom sets like `smogon_pass2`; built-in sets like `competitive` /
`gl_05_26` download from HF on first use).

### 3. flash-attn must match the installed torch

If you see `Missing flash attention 2 install` or an `undefined symbol` ABI
error from `flash_attn_2_cuda`, the torch/flash-attn combo is broken (a loose
`torch>=` + `uv sync` can bump torch across an ABI break). The fix is pinned in
`pyproject.toml`: `torch>=2.13,<2.14` and `flash-attn` as a direct-URL
prebuild wheel. If `uv run` still tries to build flash-attn from source, force
a re-sync:
```bash
uv sync --reinstall-package flash-attn
```
See the **flash-attn / torch ABI** section at the bottom for the full story.

## Step 1 — Register the checkpoint as a pretrained model

`metamon.rl.evaluate`'s `--agent` flag only accepts names from the pretrained
registry (`choices=get_pretrained_model_names()`). There is no raw-checkpoint
escape hatch in the CLI — you **must** register a class in
`metamon/rl/pretrained.py`. The registry is populated at import time by the
`@pretrained_model()` decorator, so `uv run` (editable install) picks up
edits immediately.

### Choose the right base class

| Class | When | What it provides |
|---|---|---|
| `LocalPretrainedModel` | from-scratch `rl.train` runs | Points `amago_ckpt_dir` at the run dir; you supply all config (gin, spaces, tokenizer, reward) |
| `LocalFinetunedModel` | finetunes / online-RL runs (`rl.finetune`, `rl.online_rl`) | Inherits config from a `base_model=` (gin, spaces, tokenizer, reward); you just point at the run dir + specify which base |

Online-RL runs (the `mini_online_psro_v*` lineage) use `LocalFinetunedModel`
with `base_model=<the from-scratch arch class>` — the base only supplies
architecture/spaces/tokenizer/reward config; **weights come from the run's own
checkpoints**, not the base.

### Where checkpoints live

For a run launched with `--save_dir $SAVE_DIR --run_name $RUN_NAME`:
```
$SAVE_DIR/$RUN_NAME/ckpts/
├── latest/policy.pt                    # rolling; learner rewrites each epoch
├── policy_weights/policy_epoch_<N>.pt  # named snapshots every ckpt_interval epochs
└── training_states/<run>_epoch_<N>/    # full accelerate state (resume source)
```
`LocalPretrainedModel.get_path_to_checkpoint(N)` →
`policy_weights/policy_epoch_<N>.pt`. To load `latest/policy.pt` instead, see
the `LATEST_CHECKPOINT` sentinel below.

### Template — `LocalFinetunedModel` (online-RL / finetune runs)

Mirror the existing `MiniOnlinePsroV1_3` / `MiniOnlinePsroV1_4` registrations.
Add near the other local-model classes in `metamon/rl/pretrained.py`:

```python
MY_RUN_SAVE_DIR = "/home/eddie/metamon_runs/my_run"


@pretrained_model()
class MyRunName(LocalFinetunedModel):
    """One-line description of the run."""

    def __init__(self):
        super().__init__(
            base_model=V2AGroupedV2DataAblation,   # the from-scratch arch class
            amago_ckpt_dir=MY_RUN_SAVE_DIR,
            model_name="my_run",                   # == --run_name used at training
            default_checkpoint=740,                # pin a frozen named snapshot
            train_gin_config="grouped_v2_large_isfilter.gin",
            dataset_config="online_selfplay.yaml",
        )

    def get_path_to_checkpoint(self, checkpoint: int) -> str:
        if checkpoint == LATEST_CHECKPOINT:
            return os.path.join(self.local_ckpt_dir, "latest", "policy.pt")
        return super().get_path_to_checkpoint(checkpoint)
```

Key points:
- **`@pretrained_model()`** registers by class name → `MyRunName` becomes a
  valid `--agent` choice. Use `@pretrained_model("CustomName")` to override.
- **`amago_ckpt_dir`** is the `--save_dir` from training (the parent, not the
  `.../<run_name>/ckpts` subpath — the class joins `model_name` + `ckpts`).
- **`model_name`** must exactly equal the `--run_name` used at training time
  (the class looks up `<amago_ckpt_dir>/<model_name>/ckpts/...`).
- **`base_model`** must be the architecture class the run was trained from
  (for the `mini_online_psro_v*` and `SmallG1OnlineV0` lineage that's
  `V2AGroupedV2DataAblation`). It supplies gin/spaces/tokenizer/reward only.

### Template — `LocalPretrainedModel` (from-scratch `rl.train`)

If the run was started from scratch (no base model), supply the full config:

```python
@pretrained_model()
class MyFromScratchRun(LocalPretrainedModel):
    def __init__(self):
        super().__init__(
            amago_ckpt_dir=MY_RUN_SAVE_DIR,
            model_name="my_from_scratch_run",
            model_gin_config="medium_multitaskagent.gin",
            train_gin_config="binary_rl.gin",
            default_checkpoint=40,
            action_space=metamon.interface.DefaultActionSpace(),
            observation_space=metamon.interface.TeamPreviewObservationSpace(),
            tokenizer=metamon.tokenizer.get_tokenizer("DefaultObservationSpace-v1"),
        )
```

### `default_checkpoint` — frozen vs latest

Two philosophies, both valid:

- **Frozen named checkpoint (recommended for playtesting):** set
  `default_checkpoint=740` (the latest `policy_epoch_740.pt`). Evaluations
  default to a stable snapshot that won't change under you mid-session. **Bump
  it manually** as the run advances and you want to playtest a newer snapshot.
  This is what `MiniOnlinePsroV1_4` does.

- **`LATEST_CHECKPOINT` (-1):** set `default_checkpoint=LATEST_CHECKPOINT` and
  add the `get_path_to_checkpoint` override (above). Evaluations auto-load the
  learner's rolling `latest/policy.pt` — the same file the validator reads each
  epoch. Convenient but a **moving target**: the learner rewrites it every
  epoch (~16 min), and the eval process loads it once at startup, so the
  epoch you get depends on when you launch. `MiniOnlinePsroV1_3` uses this.

You can always override the default at the CLI with `--checkpoints N` or
`--checkpoints -1`, regardless of what the class sets — but only if the class
defines the `get_path_to_checkpoint` override (otherwise `-1` looks for a
nonexistent `policy_epoch_-1.pt`).

### Verify the registration

```bash
METAMON_CACHE_DIR=/home/eddie/metamon_cache uv run python -c "
from metamon.rl.pretrained import get_pretrained_model, LATEST_CHECKPOINT
import os
m = get_pretrained_model('MyRunName')
print('default_checkpoint:', m.default_checkpoint)
print('default ->', m.get_path_to_checkpoint(m.default_checkpoint))
print('  exists:', os.path.exists(m.get_path_to_checkpoint(m.default_checkpoint)))
print('-1 ->', m.get_path_to_checkpoint(LATEST_CHECKPOINT))
"
```
Also confirm it appears in the CLI choices:
```bash
uv run python -m metamon.rl.evaluate --help | grep -A2 -- '--agent'
```

## Step 2 — Battle on the local ladder

### Command

```bash
METAMON_CACHE_DIR=/home/eddie/metamon_cache uv run python -m metamon.rl.evaluate \
    --agent MyRunName \
    --eval_type ladder \
    --gens 1 \
    --formats ou \
    --team_set competitive \
    --total_battles 250 \
    --username MyRunName
```

### What each flag does

| flag | meaning |
|---|---|
| `--agent` | registry name (must be registered, see Step 1) |
| `--eval_type ladder` | queue on the **local** Showdown server (`ws://localhost:8000`). `pokeagent` = the PokéAgent Challenge bot ladder; `heuristic` = vs built-in baselines; `metamon` = vectorized self-play vs another model; `challenge` = head-to-head vs a specific username |
| `--gens 1 --formats ou` | combines to `gen1ou` (one battle_format per gen×format pair) |
| `--team_set` | which teams the bot plays. Built-ins: `competitive`, `gl_05_26`, `hl_05_26`, `modern_replays`, `paper_replays`, `paper_variety`. Custom sets (e.g. `smogon_pass2`, `smogon_pass2_selected`) resolve from `$METAMON_CACHE_DIR/teams/<set_name>`. |
| `--total_battles` | battles before the process exits (total across all parallel actors; ladder uses 1 actor) |
| `--username` | the bot's Showdown username. Must be unique on the server. No password needed (`--no-security`). |
| `--checkpoints N` | (optional) override the default; `--checkpoints -1` loads `latest/policy.pt` if the class supports it |
| `--battle_backend` | `metamon` (default, latest), `poke-env` (deprecated, original paper), `pokeagent` (PokéAgent baselines). Defaults to the model's `battle_backend`. |
| `--save_trajectories_to` | (optional) save replays in parsed-replay format for later training |
| `--temperature` | sampling temperature (default 1.0; higher = more explorative) |
| `--no-agent_sample` | deterministic argmax instead of sampling |

### Play against it yourself

1. Start the local server (see Prerequisites) if not already running.
2. Launch the bot with the command above in one terminal.
3. Open `http://localhost:8000` in your browser, log in with a **different**
   username, and queue for the gen1ou ladder. The bot will be matched against
   you. (Usernames must be unique; the bot took the one you gave it.)

### Bot-vs-bot on the same server

Launch two instances with different `--agent`/`--username`; both queue on the
ladder and get matched against each other (or any other logged-in player).

## How it works under the hood

- `metamon/rl/evaluate/__main__.py` → `_run_default_evaluation` →
  `get_pretrained_model(args.agent)` → `pretrained_model.initialize_agent()`.
- `initialize_agent` builds the AMAGO experiment from the gin config, starts
  it (constructs the policy net — this is where FlashAttention is
  instantiated), then `torch.load`s the checkpoint and `load_state_dict`s it
  (strict, with a key/shape validation that prints
  `Checkpoint validated: N keys, M params`).
- `--eval_type ladder` → `pretrained_vs_local_ladder` →
  `_pretrained_on_ladder` → `make_local_ladder_env` builds a
  `QueueOnLocalLadder` env (`metamon/env/wrappers.py`) whose
  `server_configuration` is `LocalhostServerConfiguration` (hardcoded to
  `ws://localhost:8000`). The env queues for ladder battles; the agent steps
  through them.
- The server is a vendored Pokemon Showdown checkout at
  `server/pokemon-showdown/`, started with `node pokemon-showdown start
  --no-security`.
- **No GPU training is disturbed.** The eval process loads weights read-only
  once at startup. The learner keeps publishing `latest/policy.pt` each epoch
  unaffected.

## Reference

- `metamon/rl/pretrained.py` — the registry (`@pretrained_model`,
  `ALL_PRETRAINED_MODELS`, `get_pretrained_model`, `get_pretrained_model_names`),
  `PretrainedModel.initialize_agent` (the load+validate flow),
  `LocalPretrainedModel` / `LocalFinetunedModel`, the `LATEST_CHECKPOINT = -1`
  sentinel, and every registered model (TaurosV0, the mini_online_psro_v*
  lineage, etc.).
- `metamon/rl/evaluate/__main__.py` — the CLI (`add_cli`), the
  `_get_default_eval` dispatch, `pretrained_vs_local_ladder`, and the
  `pretrained_vs_*` helpers for every eval type.
- `metamon/env/wrappers.py` — `QueueOnLocalLadder` (the local-ladder env;
  `server_configuration` → `LocalhostServerConfiguration`),
  `ChallengeByUsername`, `PokeAgentLadder` (the only non-localhost ladder,
  pointed at `wss://battling.pokeagentchallenge.com`), and
  `get_metamon_teams` (team-set resolution, incl. custom sets from
  `METAMON_CACHE_DIR`).
- `server/pokemon-showdown/` — the vendored Showdown server. Start with
  `node pokemon-showdown start --no-security` (listens on port 8000).
- `examples/evaluate_custom_models.py` — worked examples of registering both
  `LocalPretrainedModel` and `LocalFinetunedModel` classes.

## flash-attn / torch ABI (the common breakage)

amago's transformer policy (`TformerTrajEncoder`) uses `FlashAttention`, which
requires the `flash_attn` package. `flash_attn` ships **source-only on PyPI**
(a 30+ min CUDA compile), so this repo installs a prebuilt wheel from
`mjun0812/flash-attention-prebuild-wheels` matched to a specific torch version.

### The failure mode

A loose `torch>=2.6` in `pyproject.toml` lets `uv sync`/`uv run` bump torch
across a minor-version C++ ABI break (e.g. 2.12 → 2.13 changed
`c10::impl::cow::materialize_cow_storage`). The compiled `flash_attn_2_cuda.so`
then fails to import with `undefined symbol: _ZN3c104...`. **Already-running
training survives** (old torch is in RAM), but any **new** process — including
a training resume after a crash — hits the error. So this is a latent
training-continuity bug, not just an eval issue.

### The pin (in `pyproject.toml`)

- `torch>=2.13,<2.14` — bounded to the major.minor the flash wheel was built for.
- `flash-attn @ https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.47/flash_attn-2.8.3+cu130torch2.13-cp312-cp312-linux_x86_64.whl` — direct-URL dependency (so `uv sync` installs the wheel, not the PyPI sdist, and it survives syncs).
- `requires-python = ">=3.12"` — the wheel is cp312-only; uv refuses to resolve while it tries to satisfy older Pythons too.

### If `uv run` still tries to build flash-attn from source

`uv` caches resolutions; a stale cache can still pull `flash-attn 2.8.3.post1`
(the PyPI sdist) and try to compile it (fails: `No module named 'torch'` in
the build env). Force a re-sync from the lock:
```bash
uv sync --reinstall-package flash-attn
```
Verify the fix:
```bash
uv run python -c "import torch, flash_attn; print(torch.__version__, flash_attn.__version__)"
```

### When upgrading torch

If you want a newer torch, find a matching prebuild wheel first (same
`cu13XtorchY.ZZ` + `cp312` + `linux_x86_64` tag) at
`https://github.com/mjun0812/flash-attention-prebuild-wheels/releases`, update
the direct-URL pin in `pyproject.toml`, then bump the `torch` pin to match.
Re-run `uv lock && uv sync`.
