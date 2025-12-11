# Metamon Repository Overview

## Project Description
Metamon is a research codebase for training reinforcement learning agents to play Pokémon Showdown at a human-competitive level. The project enables RL research on Pokémon Showdown by providing:

1. A standardized suite of teams and opponents for evaluation
2. A large dataset of RL trajectories reconstructed from real human battles (~4M trajectories)
3. Pretrained IL and RL policies available on HuggingFace
4. Training infrastructure using the AMAGO RL framework

The work is published as "Human-Level Competitive Pokémon via Scalable Offline RL and Transformers" (RLC, 2025).

## Environment Setup

This project uses **`uv`** for package management.

Before running any commands, activate the virtual environment and set the cache directory:

```bash
# Activate virtual environment
source .venv/bin/activate

# Set cache directory
export METAMON_CACHE_DIR=/home/eddie/metamon_cache
```

To start local Pokémon Showdown server:

```bash
cd server/pokemon-showdown && node pokemon-showdown start --no-security
```

## Repository Structure

### Core Components

- **`metamon/env/`** - Environment wrappers for Pokémon Showdown
  - `wrappers.py` - Gymnasium-compatible environment wrappers
  - `metamon_battle.py` - Battle logic integration
  - `metamon_player.py` - Player interface

- **`metamon/interface.py`** - Core abstractions
  - `ObservationSpace` - Defines how game state is represented to agents
  - `ActionSpace` - Converts agent outputs to game actions (13 discrete: 4 moves, 5 switches, 4 tera moves)
  - `RewardFunction` - Defines reward shaping
  - `UniversalState` / `UniversalAction` - Common representations

- **`metamon/rl/`** - Training and evaluation
  - `train.py` - Train agents from scratch on offline datasets
  - `finetune_from_hf.py` - Finetune pretrained models from HuggingFace
  - `evaluate.py` - Evaluate against baselines or ladder opponents
  - `pretrained.py` - Registry of all pretrained models
  - `metamon_to_amago.py` - Bridges Metamon environments to AMAGO framework

- **`metamon/data/`** - Dataset management
  - `parsed_replay_dset.py` - PyTorch dataset for reconstructed human battles
  - `download.py` - Downloads datasets from HuggingFace

- **`metamon/baselines/`** - Heuristic and learned baseline opponents
  - Includes: RandomBaseline, Grunt, GymLeader, EmeraldKaizo, etc.

- **`metamon/tokenizer/`** - Text tokenization for observations

- **`server/`** - Local Pokémon Showdown server (submodule)

## Available Pretrained Models

### Paper Models (Gens 1-4)
All available on HuggingFace at `jakegrigsby/metamon`:

| Model | Parameters | Description | Default Ckpt |
|-------|-----------|-------------|--------------|
| **SyntheticRLV2** | 200M | **Best model**: Actor-critic with value classification, 1M human + 4M synthetic self-play | 48 |
| SyntheticRLV1++ | 200M | SyntheticRLV1 finetuned on 2M battles vs diverse opponents | 40 |
| SyntheticRLV1_SelfPlay | 200M | SyntheticRLV1 finetuned on 2M self-play battles | 40 |
| SyntheticRLV1 | 200M | 1M human + 2M diverse self-play | 40 |
| SyntheticRLV0 | 200M | 1M human + 1M diverse self-play | 40 |
| LargeRL / LargeIL | 200M | Trained on 1M human battles only | 40 |
| MediumRL / MediumIL | 50M | Trained on 1M human battles only | 40 |
| SmallRL / SmallIL | 15M | Trained on 1M human battles only | 40 |
| Minikazam | 4.7M | Small RNN trained on parsed-replays v4 + self-play (best for finetuning on limited GPU) | varies |

### PokéAgent Challenge Models (Gens 1-4 + 9)
- **Abra** (57M), **Kadabra**, **Alakazam** - Gen 9-compatible models
- **SmallRLGen9Beta** (15M) - Prototype Gen 9 model

## Reward Functions

Located in `metamon/interface.py`:

1. **DefaultShapedReward** - Used by paper models
   - +/- 100 for win/loss
   - Light shaping for damage dealt, health recovered, status inflicted/received
   - **Contains annealing values** for these shaping terms

2. **AggressiveShapedReward** - Removes status shaping, makes rewards +200/+0

3. **AggressiveShapedRewardSleep** - Variant of AggressiveShapedReward
   - +200/0 for win/loss (aggressive, no loss penalty)
   - +1 reward for putting opponent to sleep
   - Keeps damage/HP shaping (±1.0) and KO differential (±2.0)

4. **BinaryReward** - Sparse: only +/- 100 for win/loss

## Observation Spaces

1. **DefaultObservationSpace** - Original paper space (text + numerical features)
2. **ExpandedObservationSpace** - Improved version with Gen 9 tera types
3. **TeamPreviewObservationSpace** - Adds opponent team preview
4. **OpponentMoveObservationSpace** - Includes revealed opponent moves

All can be tokenized using vocabs in `metamon/tokenizer/`.

---

## Experiment Skills System

**Location**: `.claude/skills/` - Team memory for experiment knowledge

The metamon repository uses a **skills-based knowledge preservation system** to capture experiment insights and prevent knowledge loss. Skills document successful approaches, failed attempts, and hyperparameter decisions to accelerate future research.

### Using Skills

**Before starting experiments**:
```bash
/advise <your experiment description>
```
Claude will search the skills registry and surface relevant past experiments, failure patterns, and recommended configurations.

**After completing experiments**:
```bash
/retrospective
```
Claude will extract insights from the conversation and generate a structured skill file documenting what worked, what failed, and key learnings.

### Available Skills

**Training Workflows**:
- `selfplay-loop-workflow` - Gen1 OU self-play loop (Generate → Filter → Train → League)

**Configuration**:
- `dynamic-damping-config-selection` - Choosing gin configs for dynamic damping
- `reward-scale-matching` - Matching reward_multiplier to reward function scale

**Troubleshooting**:
- `format-filtering-troubleshooting` - Fixing data loading / format filtering issues

### Creating New Skills

Use `/retrospective` after experiments to auto-generate skills, or manually create following `.claude/SKILL_TEMPLATE.md`. Skills should:
- Document specific failures in detail (most valuable)
- Include copy-paste ready commands
- Specify exact hyperparameters tested
- Provide concrete success criteria

---

## Current Work: Nash Equilibrium Training (PSRO)

**Goal**: Develop Gen 1 OU Nash equilibrium policy using Policy Space Response Oracles (PSRO).

**Status**: Phase 1 in progress
- Phase 0 complete: Baseline population established (exploitability = 0.44)
- Phase 1: Iterative best-response training to reduce exploitability to ~0.1-0.2

**Quick Start**:
```bash
python -m metamon.nash.run_psro \
    --phase0_dir ~/nash_phase0 \
    --save_dir ~/nash_phase1 \
    --num_iterations 5 \
    --battle_format gen1ou \
    --team_set modern_replays_v2 \
    --collection_battles 500 \
    --oracle_epochs 3 \
    --oracle_model_config synthetic_multitaskagent.gin \
    --oracle_train_config psro_oracle.gin \
    --init_from_checkpoint SyntheticRLV2 \
    --parsed_replay_dir ~/metamon_cache/parsed-replays \
    --formats gen1ou \
    --log
```

**Documentation**:
- **`NASH.md`**: Complete Nash-first training roadmap (Phase 0-4)
- **`PSRO_GUIDE.md`**: Step-by-step PSRO training guide
- **`LESSONS_LEARNED.md`**: Key findings from Phase 0 experiments
- **`metamon/nash/README.md`**: Package technical documentation
- **`metamon/nash/claude.md`**: Implementation details and status

---
---

## Self-Play Training Infrastructure

### Existing Production Scripts (DON'T REIMPLEMENT)

Located in `scripts/`:
- **`generate_selfplay_data.py`** - Parallel self-play data collection (model vs itself)
- **`filter_selfplay_data.py`** - Quality filtering (invalid actions, win/loss balance)
- **`self_play_tournament.py`** - Round-robin evaluation for checkpoint comparison
- **`calculate_elo.py`** - ELO rating calculation

**Workflow**: Generate → Filter → Train (via `metamon.rl.finetune_from_hf`) → Evaluate → Repeat

See `scripts/README.md` for usage details.

### Dynamic Damping (NEW)

**Location**: `metamon/rl/dynamic_damping.py`

**What**: Prevents policy collapse in self-play via:
- Reverse-KL regularization: KL(π_new || π_ref)
- Power-law coefficient schedules (smooth decay)
- Adaptive LR/KL control based on observed KL

**Integration**: Works with existing training via gin config

**Enable**: Use `vanilla_selfplay_damped.gin` instead of `vanilla_selfplay_baseline.gin`

**Docs**: See `GEN1OU_SELFPLAY_GUIDE.md` for complete Gen1 OU workflow

### Critical: Format Filtering

**ALWAYS use `--formats gen1ou`** in training to prevent loading other formats:
```bash
python -m metamon.rl.finetune_from_hf --formats gen1ou ...
```

Data must be in format-specific subdirectories: `data_dir/gen1ou/*.json.lz4`

### Existing Gen1 OU Data

- Nash Phase 0: `/home/eddie/metamon/other/nash_phase0/trajectories/gen1ou/` (1,104 replays)
- Super Dataset Loop 3: `/home/eddie/metamon/trajectories/super_dataset_loop3/gen1ou/` (25,072 replays)

---

## Recent Training: Aggressive Sleep Strategy

**Goal**: Train a Gen1 OU specialist that prioritizes sleep-inducing moves

**Setup**:
- Base model: `DampedBinarySuperV1_Epoch4`
- Dataset: 25,072 offline trajectories from super_dataset_loop3
- Reward: `AggressiveShapedRewardSleep` (+200/0 win/loss, +1 sleep bonus)
- Observation: `ExpandedObservationSpace` (includes PP, tera types, sleep/freeze flags)
- Training: 5 epochs with BC-heavy offline finetuning

**Training Command**:
```bash
python -u -m metamon.rl.finetune_from_hf \
    --run_name "aggressive-sleep-loop3v1" \
    --finetune_from_model DampedBinarySuperV1_Epoch4 \
    --custom_replay_dir ~/metamon/trajectories/super_dataset_loop3/ \
    --custom_replay_sample_weight 1.0 \
    --formats gen1ou \
    --train_gin_config selfplay_damped_aggressive.gin \
    --reward_function AggressiveShapedRewardSleep \
    --epochs 5 \
    --save_dir ~/metamon/models/gen1_aggressive_sleep_loop3 \
    --eval_gens 1 \
    --log
```

**Key Configuration** (`selfplay_damped_aggressive.gin`):
- **Reward scale**: `reward_multiplier = 0.05` (handles +200/0 scale)
- **Data mix**: BC-heavy (75%) with small DPG (25%) for offline stability
- **KL target**: 0.015 (slightly looser than conservative 0.01)
- **Damping**: Starts at 0.20 with controller-driven adaptation (no decay)

**What to Monitor**:
- KL divergence should stay around 0.015 (range 0.01-0.0225)
- Sleep move usage should increase over training
- Policy entropy should stay > 1.0 (no collapse)
- Win rate should improve gradually

---

## Notes

- All pretrained models support Gens 1-4, but have bias toward training distribution
- Gen 9 support is experimental/beta
- Parsed replays location: `~/metamon_cache/parsed-replays` (set via `METAMON_CACHE_DIR` env var)
- Best practice: Offline pretraining → Online finetuning
