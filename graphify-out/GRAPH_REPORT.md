# Graph Report - metamon  (2026-07-20)

## Corpus Check
- 168 files · ~1,314,978 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 2840 nodes · 0 edges · 151 communities (124 shown, 27 thin omitted)
- Extraction: 0% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Backend Team Construction Matchup
- Vectorized AMAGO Policy & Opponent
- Interface & Pretrained Models
- Replay Parser Forward Simulation
- Env Wrappers
- RL Evaluate Harness
- Replay Parser State
- Pretrained Model Registry
- Model-Based Baselines
- Backend Team Prediction #9
- Vectorized Showdown Sim
- Base Baselines
- Backend Replay IO
- Replay Parser Reconstruction
- Metamon Battle Environment
- RL Experimental Ensemble
- Team Construction Serving
- Vectorized Env Wrapper
- Backend Team Prediction #18
- Baselines Heuristic Basic #19
- Backend Team Construction #20
- Backend Team Prediction #21
- Imitation Learning Model
- RL Metamon Tokenization
- Backend Team Prediction #24
- Backend Team Prediction #25
- RL Custom Agent
- Backend Replay Parser #27
- Backend Team Construction #28
- RL Online Training
- Backend Team Prediction #30
- Backend Team Prediction #31
- Backend Team Construction #32
- Backend Replay Parser #33
- Backend Team
- Backend Team Prediction #35
- Backend Team Prediction #36
- Interface Observation Space
- Vectorized AMAGO Config
- Rl Dataset Config #39
- Backend Team Prediction #40
- Backend Team Prediction #41
- Universal Pokemon State
- Default Observation Space
- Rl Evaluate Results
- Backend Team Construction #45
- Env Vectorized Battle
- PokeEnv Wrapper
- Rl Online Rl
- Backend Replay Parser #49
- Data Download Rationale
- Backend Team Prediction #51
- Backend Team Prediction #52
- Rl Metamon To #53
- Grouped Observation Space
- Backend Replay Parser #55
- Backend Team Prediction #56
- Backend Team Prediction #57
- Rl Evaluate Common #58
- Parsed Replay Dataset
- Env Vectorized Lane
- RL Evaluate Ladder Self-Play
- Backend Replay Parser #62
- Backend Team Construction #63
- Backend Team Construction #64
- Backend Team Prediction #65
- Backend Team Prediction #66
- Rl Metamon To #67
- IL Train Runner
- RL Evaluate Team Preview
- Rl Metamon To #70
- Backend Replay Parser #71
- Backend Replay Parser #72
- Showdown Dex Data
- Baselines Heuristic Basic #74
- Data Parsed
- Rl Evaluate Common #76
- Pokemon Tokenizer
- Backend Replay Parser #78
- Env Vectorized Action
- Rl Metamon To #80
- Rl Pretrained #81
- Rl Pretrained #82
- Backend Team Prediction #83
- Backend Team Prediction #84
- Heuristic Kaizo Baseline
- Env Vectorized Team
- Backend Team Prediction #87
- Env Metamon Battle #88
- Env Metamon Player
- Env Vectorized Package
- Rl Evaluate
- Backend Team Prediction #92
- Baselines Init
- Data Parsed Replay #94
- Rl Metamon To #95
- Backend Team Prediction #96
- Baselines Heuristic Basic #97
- Rl Metamon To #98
- Backend Team Prediction #99
- Baselines Heuristic Basic #100
- Rl Metamon To #101
- Backend Team Prediction #102
- Backend Team Prediction #103
- Data Parsed Replay #104
- Rl Metamon To #105
- Backend Team Prediction #106
- Backend Replay Parser #107
- Heuristic Kaizoplus Baseline
- Config
- Discrete
- Env Vectorized Vector
- Rl Pretrained #112
- Backend Replay Parser #113
- Backend Team Prediction #114
- Backend Team Prediction #115
- Rl Dataset Config #116
- Backend Replay Parser #117
- Backend Team Prediction #118
- Data Raw Replay
- Env Metamon Battle #120
- Baselines Heuristic Basic #121
- Env Metamon Battle #122
- Backend Replay Parser #123
- Env Metamon Battle #124
- Env Metamon Battle #125
- Env Metamon Battle #126
- Env Metamon Battle #127
- Backend Replay Parser #128
- Env Metamon Battle #129
- Env Metamon Battle #130
- Env Metamon Battle #131
- Env Metamon Battle #132
- Env Metamon Battle #133
- Rl Evaluate Ladder #134

## God Nodes (most connected - your core abstractions)
1. `PokemonTokenizer` - 149 edges
2. `ObservationSpace` - 119 edges
3. `ActionSpace` - 118 edges
4. `RewardFunction` - 116 edges
5. `Pokemon` - 109 edges
6. `TokenizedObservationSpace` - 82 edges
7. `PretrainedModel` - 73 edges
8. `MetamonDiscrete` - 70 edges
9. `MetamonBackendBattle` - 69 edges
10. `UniversalState` - 68 edges

## Surprising Connections (you probably didn't know these)
- `MetamonBackendBattle` --uses--> `Nothing`  [INFERRED]
  env/metamon_battle.py → backend/replay_parser/replay_state.py
- `PokeAgentBackendBattle` --uses--> `Nothing`  [INFERRED]
  env/metamon_battle.py → backend/replay_parser/replay_state.py
- `ActionSpace` --uses--> `Nothing`  [INFERRED]
  interface.py → backend/replay_parser/replay_state.py
- `DefaultActionSpace` --uses--> `Nothing`  [INFERRED]
  interface.py → backend/replay_parser/replay_state.py
- `DefaultObservationSpace` --uses--> `Nothing`  [INFERRED]
  interface.py → backend/replay_parser/replay_state.py

## Import Cycles
- None detected.

## Communities (151 total, 27 thin omitted)

### Community 0 - "Backend Team Construction Matchup"
Cohesion: 0.07
Nodes (8): Path, Popen, Path, Namespace, Path, Random, Path, Random

### Community 1 - "Vectorized AMAGO Policy & Opponent"
Cohesion: 0.06
Nodes (32): AmagoLadderPolicyDriver, Batched policy forward matching AMAGO ladder rollouts., Record env feedback after a committed decision (matches AMAGOEnv.step)., Vectorized Showdown simulation env for Metamon.  Runs many battles in parallel i, AmagoBatchedOpponent, BatchedOpponent, ConfigBatchedOpponent, ABC (+24 more)

### Community 2 - "Interface & Pretrained Models"
Cohesion: 0.08
Nodes (32): Get an instantiated observation space object by name., Get an instantiated action space object by name., Get an instantiated reward function object by name., Alakazam, Kadabra2, Kakuna, Minikazam, Grouped V2 arch trained on the V2A baseline data mix (pac-base 60%, pac-explorat (+24 more)

### Community 3 - "Replay Parser Forward Simulation"
Cohesion: 0.05
Nodes (24): |-transform|POKEMON|SPECIES, |-fieldstart|CONDITION or |-fieldend|CONDITION, # |-start|POKEMON|EFFECT or # |-end|POKEMON|EFFECT, |-setboost|POKEMON|STAT|AMOUNT, |-clearpositiveboost|TARGET|POKEMON|EFFECT, |-clearnegativeboost|POKEMON, |-copyboost|SOURCE|TARGET, |-restoreboost|p2a: Gorebyss|[silent] (+16 more)

### Community 4 - "Env Wrappers"
Cohesion: 0.07
Nodes (31): datetime, MetamonPlayer, PokeAgentPlayer, Player, Extended Player with optional team preview prediction model., Initialize MetamonPlayer.          Args:             team_preview_model: Optiona, BattleAgainstBaseline, ChallengeByUsername (+23 more)

### Community 5 - "RL Evaluate Harness"
Cohesion: 0.06
Nodes (25): Dataset, Team preview prediction: predict which pokemon to lead with given both teams. 12, Perceiver-style model: 12 pokemon + optional additional info -> predict lead (1, Dataset for team preview prediction from parsed replays., load model from checkpoint. tokenizer auto-loaded if not provided., train team preview model with early stopping, TeamPreviewDataset, TeamPreviewModel (+17 more)

### Community 6 - "Replay Parser State"
Cohesion: 0.06
Nodes (7): PokedexMissingEntry, Pokemon, Update this Pokemon's info based on a version of itself from later in the battle, Turn, PEEffect, PESideCondition, Replacement

### Community 7 - "Pretrained Model Registry"
Cohesion: 0.08
Nodes (27): Experiment, An observation space that tokenizes specified keys of the default observation sp, TokenizedObservationSpace, MetamonDiscrete, Discrete policy with temperature-based sampling.      Extends AMAGO's Discrete P, LargeIL, LargeRL, MediumIL (+19 more)

### Community 8 - "Model-Based Baselines"
Cohesion: 0.07
Nodes (19): BaseRNN, BCRNNBaseline, MiniRNN, PretrainedOnCPU, ABC, Battle, What do we do when the model recommends an invalid move?, # TODO: force old models to old action space (+11 more)

### Community 9 - "Backend Team Prediction #9"
Cohesion: 0.09
Nodes (11): PersistentShowdownValidator, Path, Popen, NODE_PATH entries so tools/persistent_showdown_validator.js can require pokemon-, List team files once each, in deterministic order before optional shuffle., ValidateFileResult, RandomBaseline, picks a totally random move (+3 more)

### Community 10 - "Vectorized Showdown Sim"
Cohesion: 0.07
Nodes (13): LaneHandler, Any, Python transport for the vectorized Showdown Node host.  Spawns ``battle_host.js, Send many lane choices in one stdin write (one flush)., Dispatch host chunks until ``predicate()`` is True., Fan-out/fan-in coordinator over several :class:`ShowdownSimProcess` workers., Anything that can consume a host chunk for a single lane., Owns the Node host subprocess and the host transport.      Args:         node_pa (+5 more)

### Community 11 - "Base Baselines"
Cohesion: 0.08
Nodes (25): Baseline, ABC, Battle, BattleOrder, Player, Recreate current stat given base stats and level -- assuming best DV / StatExp, Recreate current stat given a pokemon's base stats, level, iv, and ev         (U, Heuristic to determine our chances of outspeeding an opponent given         that (+17 more)

### Community 12 - "Backend Replay IO"
Cohesion: 0.06
Nodes (27): Action, BackwardMarkers, Nothing, Any, Enum, `None` means "unknown", while these values mean     "Known to be missing or N/A", Mark info with "all we know is that we definitely can't know this", Check if a value is considered "unknown" (+19 more)

### Community 13 - "Replay Parser Reconstruction"
Cohesion: 0.07
Nodes (21): ActionIndexError, CalledForeignConsecutive, ForwardException, IncompleteEffectLogic, MimicMiss, MovedexMissingEntry, MoveInfoNotFound, MultipleTera (+13 more)

### Community 14 - "Metamon Battle Environment"
Cohesion: 0.06
Nodes (15): MetamonBackendBattle, Any, Replace poke-env's Showdown protocol/sim interpreter with Metamon's verison., :return: How many turns of dynamax are left. None if dynamax is not active, :return: The format of the battle, in accordance with Showdown protocol, :return: The generation of the battle; will be the parameter with which the, The last request received from the server. This allows players to track, :return: The maximum acceptable size of the team to return in teampreview, if (+7 more)

### Community 15 - "RL Experimental Ensemble"
Cohesion: 0.11
Nodes (14): _AnchorDeviationMetrics, _EnsembleHiddenState, _EnsembleMemberRuntime, _EnsembleTrajEncoderProxy, HeuristicRouterEnsemblePolicy, _ProposerVariant, Any, device (+6 more)

### Community 16 - "Team Construction Serving"
Cohesion: 0.11
Nodes (12): Any, Path, ArgumentParser, Namespace, ndarray, Path, Team, PairEvaluator (+4 more)

### Community 17 - "Vectorized Env Wrapper"
Cohesion: 0.09
Nodes (13): ndarray, Space, Seed per-lane trajectory buffers with the eval POV at battle start., Append one lane's choices in physical Showdown order (p1, then p2)., Repair an illegal action against the legal mask, then build a choice.          M, Re-answer from the action already committed this step.          Used for trap ``, Answer a single opponent-side request with the batched opponent policy., Re-answer a single side after an ``|error|`` re-prompt.          Used by :meth:` (+5 more)

### Community 18 - "Backend Team Prediction #18"
Cohesion: 0.08
Nodes (20): EvaluationAccumulator, Tensor, Top-k accuracy: is the correct token in the top-k predictions?, Average confidence (max probability) of predictions., Expected Calibration Error (ECE)., Accumulates evaluation statistics across batches for per-generation metrics., Compute all evaluation metrics., Extract generation number for each sample from the format token. (+12 more)

### Community 19 - "Baselines Heuristic Basic #19"
Cohesion: 0.06
Nodes (17): BasicSwitcher, Gen1Trainer, Gen1TrainerGoodSwitching, Grunt, GruntRandomSwitching, GymLeader, NotRiskTaker, PreferPriority (+9 more)

### Community 20 - "Backend Team Construction #20"
Cohesion: 0.12
Nodes (7): Team, InteractionFeatureLayout, Team, Convert sparse dict rows into a scipy CSR matrix., ndarray, Add swapped team/order examples (x, y) -> (swap(x), 1-y)., Split originals first so augmented pairs stay in the same split.

### Community 21 - "Backend Team Prediction #21"
Cohesion: 0.09
Nodes (12): PokemonSet, Counts the number of details revealed in this PokemonSet.          This is used, Counts the number of moves revealed in this PokemonSet., Canonical (species, moves, item, ability) key — the competitive definition of a, Check if this Pokemon is actually present (not a placeholder for unknown mon)., Get list of (key, subkey) tuples for revealed attributes that can be masked., Determines whether this Pokemon is "consistent" with another Pokemon,         wh, Used to convert between the Pokemon we are filling in the replay parser (+4 more)

### Community 22 - "Imitation Learning Model"
Cohesion: 0.07
Nodes (11): CrossAttentionBlock, FFTurnEmbedding, FixedPosEmb, ABC, LongTensor, Tensor, Take the multimodal sequence, add on a few blank ("scratch") tokens     like it', Map the multimodal (text, numerical) features of each turn in a battle to a (+3 more)

### Community 23 - "RL Metamon Tokenization"
Cohesion: 0.09
Nodes (14): MetamonAMAGOWrapper, PSLadderAMAGOWrapper, Battle on the local Showdown ladder!, Battle on the NeurIPS 2025 PokéAgent Challenge ladder!, Battle a specific opponent by username (head-to-head challenge mode)., Battle against a built-in baseline opponent, AMAGOEnv wrapper for single-env Showdown / pokepy / poke-env gym envs.      Use, AMAGOEnv wrapper for batched Showdown / pokepy vector envs (already_vectorized). (+6 more)

### Community 24 - "Backend Team Prediction #24"
Cohesion: 0.09
Nodes (21): IterativeDecodingStats, Any, device, Module, Path, TeamSet, High-level wrapper: TeamSet <-> tensors for training and inference.      Decoder, The underlying nn.Module. (+13 more)

### Community 25 - "Backend Team Prediction #25"
Cohesion: 0.09
Nodes (16): Which Pokemon (0-5) does each position belong to? Format (pos 0) → -1., Get the sequence position of each Pokemon's name. Index -1 → 0., Get sequence positions of all 6 Pokemon names., Get the 4 move sequence positions for a specific Pokemon (0-5)., Encode a team to token IDs for inference.          Returns:             (tokens,, Encode (masked, ground_truth) pair to token IDs for training.          Returns:, Convert a single Pokemon to sequence tokens with given move order., Convert Pokemon pair to sequences with coordinated move ordering. (+8 more)

### Community 26 - "RL Custom Agent"
Cohesion: 0.09
Nodes (14): MultiTaskAgent, ISAdvantageFilter, MetamonFinetuneAgent, Any, Batch, Tensor, Finetuning agent with a slow-EMA tortoise shadow and optional IS correction.  Im, Inject boolean mask for BN statistics; cleared after use. (+6 more)

### Community 27 - "Backend Replay Parser #27"
Cohesion: 0.10
Nodes (10): POVReplay, date, TeamSet, # TODO: v3-beta lets movesets go over 4... forcing a fix on the interface side., Team prediction works by:      1. Converting the team we've gathered here in the, InconsistentTeamPrediction, Enum, Indicate that a replay contains The Game's Most Annoying Mechanics™,     which a (+2 more)

### Community 28 - "Backend Team Construction #28"
Cohesion: 0.17
Nodes (9): PokemonSet, One canonical set used to represent a Pokemon species in team search/simulation., Any, date, Path, PokemonSet, Team, Create a standardized per-species Showdown export block.      In early generatio (+1 more)

### Community 29 - "RL Online Training"
Cohesion: 0.11
Nodes (5): Legal agent-action indices, mirroring ``PokeEnvWrapper._update_legal_actions``., Any, UniversalState, Online-only wrapper that randomly hides computed battle stats.      Newly collec, StatsDropoutObservationSpace

### Community 30 - "Backend Team Prediction #30"
Cohesion: 0.09
Nodes (13): FilteredTeamsFromReplaysDataset, Dataset, Filters TeamDataset by assuming that the poke-paste team files were generated by, Dataset for team prediction using index_scored.csv.      Supports generation-wei, Get the max index to sample from for a generation. Base class uses all files., Load a team file and process it through masker and Team2Seq., Override to restrict sampling to top percentile of files by score.          Retu, ReplayTeamFilenameMeta (+5 more)

### Community 31 - "Backend Team Prediction #31"
Cohesion: 0.11
Nodes (17): Dataset with percentile-based filtering and curriculum support.      Inherits ge, Enable curriculum learning with a shared percentile that can be updated., Update the curriculum percentile (call from main process)., Get current percentile threshold., ScoredTeamPredictionDataset, Accumulates semantic metrics across batches with per-generation tracking., SemanticMetricsAccumulator, EvalResults (+9 more)

### Community 32 - "Backend Team Construction #32"
Cohesion: 0.15
Nodes (14): BattleExample, One supervised training example for the team-vs-team win model., Path, Random, Team, Uniform metagame sampler used by the paper-style training setup., Model-guided active sampler that prioritizes uncertain team-vs-team pairs., Deterministic fallback simulator useful for CI and correctness tests. (+6 more)

### Community 33 - "Backend Replay Parser #33"
Cohesion: 0.15
Nodes (5): StrParsingException, |-block|POKEMON|EFFECT|MOVE|ATTACKER, |move|POKEMON|MOVE|TARGET, |-activate|EFFECT          also the catch-all message PS sends for minor edge ca, for the misc. "[from] item/ability/move [of] id: pokemon" messages

### Community 34 - "Backend Team"
Cohesion: 0.11
Nodes (10): Tensor, returns (team_tokens, additional_info_tokens, lead_idx, format_token), Predict lead from a UniversalState (should be first state with teampreview)., Predict which pokemon to lead with.          Returns:             Tuple of (pred, Sorts Pokémon alphabetically according to their active species     name (which w, Sorts moves alphabetically according to their name in lowercase     with special, An object that represents a move in the backend-agnostic "Universal" format., UniversalMove (+2 more)

### Community 35 - "Backend Team Prediction #35"
Cohesion: 0.14
Nodes (10): date, PokemonSet, TeamSet, NaiveUsagePredictor *independently* samples missing details from PS usage stats., Sample from the top k choices, weighted by their probabilities., Score a candidate roster based on how likely it is to have been made from the cu, Scores a diff between a current Pokemon and a candidate predicted Pokemon., The old system would be equivalent to:         return EARLIEST_USAGE_STATS_DATE, (+2 more)

### Community 36 - "Backend Team Prediction #36"
Cohesion: 0.09
Nodes (9): Decode token IDs back to a TeamSet., Returns a dictionary of the details in `other` that are not in `self`,         a, Randomly sets some of the known attributes of this PokemonSet to be missing,, Represents an entire team during team prediction.      Mostly splits the functio, Note that in gen1-4 the leads need to match, but the rest of the roster can be i, Determines whether this team is "consistent" with another team,         where "c, Outputs the poke-paste-style string., Creates a TeamSet from a showdown file. (+1 more)

### Community 37 - "Interface Observation Space"
Cohesion: 0.09
Nodes (10): AMAGOEnv, ObservationSpace, Clear any internal state (between battles)., Return a dictionary of tokenizable keys and their expected (max) length., Create an environment that does nothing. Can be used to initialize a policy, Initialize an AMAGO experiment that will be used to load a pretrained checkpoint, Prototype for Gen9 agents. Trained entirely on human replays (parsed-replays v3), SmallRLGen9Beta (+2 more)

### Community 38 - "Vectorized AMAGO Config"
Cohesion: 0.12
Nodes (12): callable, Any, device, Module, ndarray, AMAGO policy drivers that mirror QueueOnLocalLadder / MetamonAMAGOWrapper semant, Run vectorized Showdown eval with symmetric ladder-style policy drivers.      Th, Return action indices; only ``active`` lanes advance recurrent state. (+4 more)

### Community 39 - "Rl Dataset Config #39"
Cohesion: 0.14
Nodes (15): CustomReplaySource, DatasetConfig, RLDataset, YAML-based dataset configuration for metamon RL training.  A DatasetConfig captu, Load a DatasetConfig from a YAML file., Write a DatasetConfig to a YAML file., Recursively flatten ``prev_dataset`` references into a flat entry list.      Whe, Resolve a config and convert it back to a flat DatasetConfig.      The returned (+7 more)

### Community 40 - "Backend Team Prediction #40"
Cohesion: 0.14
Nodes (10): IterativeTeamDecoder, Tensor, Apply nucleus (top-p) filtering and renormalize., MaskGIT-style iterative decoding with re-sorting after each fill.      After eac, Mask ratio schedule: gamma(r) gives fraction of tokens still masked at progress, Compute permutation to put tokens in canonical order., Reset duplicate tokens at given positions, keeping highest confidence., Enforce unique pokemon names and moves per pokemon (reset duplicates to $missing (+2 more)

### Community 41 - "Backend Team Prediction #41"
Cohesion: 0.12
Nodes (12): LocalGlobalTeamTransformer, LongTensor, Tensor, (batch, 6*attrs, d) + format -> (batch*6, 1+attrs, d)., (batch*6, 1+attrs, d) -> (batch, 6*attrs, d), dropping format token., Transformer encoder for team prediction (token + type + position embeddings)., Training forward: logits for loss computation., One-shot decode for comparison with iterative decoding during eval. (+4 more)

### Community 42 - "Universal Pokemon State"
Cohesion: 0.16
Nodes (7): Effect, Returns a teampreview order for the given battle.          If a team_preview_mod, Status, Straight-through conversion from metamon replay parser Pokemon object         to, UniversalPokemon, ReplayNothing, ReplayPokemon

### Community 43 - "Default Observation Space"
Cohesion: 0.12
Nodes (6): DefaultObservationSpace, ExpandedObservationSpace, OpponentMoveObservationSpace, The default observation space used by the paper.      Observations become a dict, Adds PP, the opponent's revealed party, and edge case sleep/freeze flags to Defa, Trades some text tokens to make space for the opponent's revealed moves.

### Community 44 - "Rl Evaluate Results"
Cohesion: 0.12
Nodes (11): MatchupResult, Results tracking, crash recovery, and win matrix output for auto-evaluation.  Us, Build a win-rate matrix from all completed matchups.          Returns:, Print a formatted win matrix to the terminal using rich., Result of a single head-to-head matchup., Write the win matrix to a CSV file., Track matchup results with crash recovery via append-only JSONL.      Args:, Load previously completed matchups for crash recovery. (+3 more)

### Community 45 - "Backend Team Construction #45"
Cohesion: 0.17
Nodes (9): Team, Canonical team representation: sorted tuple of unique Pokemon IDs., Parse comma/space-separated team IDs or pass-through existing integer sequences., ndarray, PairEvaluator, Team, Expand restricted strategies via best response to the current equilibrium mixtur, BestResponseFn (+1 more)

### Community 46 - "Env Vectorized Battle"
Cohesion: 0.13
Nodes (8): Lane, lanes, MSG, NOTE: we must NOT also iterate the raw `battleStream` here., readline, rl, Showdown, STREAM

### Community 47 - "PokeEnv Wrapper"
Cohesion: 0.14
Nodes (6): PokeEnvWrapper, Any, Battle, Space, A thin wrapper around poke-env's OpenAIGymEnv that handles the observation space, OpenAIGymEnv

### Community 48 - "Rl Online Rl"
Cohesion: 0.11
Nodes (8): Download a set of teams from huggingface (if necessary) and return a TeamSet., TeamSet, Worker script: runs one side of a head-to-head matchup.  Loads a pretrained mode, Online RL finetuning from a registered pretrained model.  Architecture, observat, Newest epoch with a full accelerate state under ckpts/training_states/., Return the path to the policy weights file to load., Return kwargs for ``make_metamon_env`` opponent configuration., Configure distributed roles (AMAGO only ships collect/learn helpers).

### Community 49 - "Backend Replay Parser #49"
Cohesion: 0.11
Nodes (5): RareValueError, Cancel an opponent's switch if the user's item was activated by a switch-out mov, |player|PLAYER|USERNAME|AVATAR|RATING, |-item|POKEMON|ITEM|[from]EFFECT or |-enditem|POKEMON|ITEM|[from]EFFECT, Boosts

### Community 50 - "Data Download Rationale"
Cohesion: 0.13
Nodes (9): Get the current version of a dataset., Download the parsed replays for a given battle format.      Args:         battle, Download the teams for a given battle format and set name.      Args:         ba, Download the "replay stats" for a given version.      Replay stats are json stat, Download the "raw" (unprocessed) replays.      We maintain a dataset of replays, Download self-play data from the metamon-parsed-pile dataset.      Args:, Formats published on HF for a self-play subset., Resolve (subset, format) pairs to download for self-play data. (+1 more)

### Community 51 - "Backend Team Prediction #51"
Cohesion: 0.11
Nodes (10): Decoder, IterativeStatsAccumulator, OneShotDecoder, ABC, Abstract base class for team prediction decoders., Aggregate batch stats into a summary dict (mask ratios, commits, confidences)., Number of decoding iterations., Single-pass decoding with temperature and nucleus sampling.      Used for one-sh (+2 more)

### Community 52 - "Backend Team Prediction #52"
Cohesion: 0.24
Nodes (6): date, Exception, List available baseline/rank subdirectories in the processed usage-stats dataset, Raised when usage stats cannot be loaded at the requested rank/time window., UsageStatsLoadError, Download the usage stats for a given battle format and year/month.      Usage st

### Community 53 - "Rl Metamon To #53"
Cohesion: 0.14
Nodes (8): _Categorical, MetamonGroupedTstepEncoderV2, MetamonMaskedActor, Any, Tensor, Args:             x: ``(B, n_groups * group_seq_len, d_model)`` — all groups con, Timestep encoder for GroupedObservationSpace.      Three-stage architecture:, Default AMAGO Actor with optional logit masking of illegal actions.      Note th

### Community 54 - "Grouped Observation Space"
Cohesion: 0.16
Nodes (5): GroupedObservationSpace, GroupedStatsObservationSpace, ndarray, Groups observations by entity for use with a shared Pokemon encoder.      Unlike, GroupedObservationSpace variant that also encodes computed battle stats.      On

### Community 55 - "Backend Replay Parser #55"
Cohesion: 0.18
Nodes (6): UnfinishedMessageException, Cancel an opponent's switch if the user's ability was activated by a switch-out, |-damage|POKEMON|HP STATUS or |-heal|POKEMON|HP STATUS, |-boost|POKEMON|STAT|AMOUNT or |-unboost|POKEMON|STAT|AMOUNT, |-ability|POKEMON|ABILITY|[from]EFFECT, |-sidestart|SIDE|CONDITION or |-sideend|SIDE|CONDITION or |-swapsideconditions

### Community 56 - "Backend Team Prediction #56"
Cohesion: 0.22
Nodes (3): Any, Path, Build replay-derived team sets from cached revealed_teams.  Set definitions live

### Community 57 - "Backend Team Prediction #57"
Cohesion: 0.25
Nodes (7): Path, index.csv helpers for Metamon team set directories., Yield (format_dir, battle_format) under a team set directory., Return the directory containing team files for a format., List team filenames relative to format_dir (sorted)., Load absolute paths to team files from index.csv if present, else scan disk., PathLike

### Community 58 - "Rl Evaluate Common #58"
Cohesion: 0.13
Nodes (11): CompletedProcess, MatchupPairResult, Shared utilities for auto-evaluation launchers (h2h, sweep, ladder_self_play)., Replace ``${var}`` / ``${var:default}`` in *text* with provided values.      YAM, Expand a config value into a flat Python list.      Supports the following forms, Draw a single random element from a config value.      Accepts all forms underst, Human-readable summary of a raw config value (for preview tables)., Round-robin distribute items across GPUs. (+3 more)

### Community 59 - "Parsed Replay Dataset"
Cohesion: 0.14
Nodes (10): MetamonDataset, Dataset, ndarray, Detect whether each format is stored as flat directory or tar archive., Load JSON data from either tar archive or disk file., TAR: Read file using ratarmountcore (O(1) random access)., Base dataset class for loading parsed Pokémon battle trajectories.      Parsed r, DIRECTORY: Read file from disk. (+2 more)

### Community 60 - "Env Vectorized Lane"
Cohesion: 0.15
Nodes (7): Record that ``side``'s parsed battle state changed., Whether ``side`` has a fresh, *fully-materialized* decision available., True once a new, fully-synchronized decision cycle is available.          Each `, Record that the env has consumed/acted on the current cycle., Two-POV view of one Showdown battle driven by raw protocol text., Create fresh per-POV battles for a new game in this lane., StreamBattleLane

### Community 61 - "RL Evaluate Ladder Self-Play"
Cohesion: 0.16
Nodes (7): Load and validate a YAML config file.      If *template_vars* is provided, ``${v, Merge per-policy overrides on top of defaults., Expand one merged agent config into weighted pool rows.      ``num_agents: 1`` (, Get full agent details for preview. Returns list of dicts with username, model_n, Thread-safe counter for battle-generation throughput., Expand agents based on num_agents field, _StatsTracker

### Community 62 - "Backend Replay Parser #62"
Cohesion: 0.19
Nodes (3): ActionMisaligned, ForceSwitchMishandled, ForwardVerify

### Community 63 - "Backend Team Construction #63"
Cohesion: 0.24
Nodes (5): ndarray, BaselineScorer, InteractionScorer, ndarray, Team

### Community 64 - "Backend Team Construction #64"
Cohesion: 0.23
Nodes (6): Path, Parse repeated --moveset-spec values.      Format: "PokemonName:Move1|Move2|Move, Return best team using (name match count, moves matched ratio) ranking., Load pokemon -> team_ids mapping from CSV.      Returns:         teams_by_name:, Load moveset index as team_id -> pokemon_name -> list[move-set]., Map team IDs (e.g. '0001') to concrete team files under team_dir.

### Community 65 - "Backend Team Prediction #65"
Cohesion: 0.15
Nodes (7): Compute revealed scores for all teams in the dataset.  Output:   - index_scored., Print per-generation stats and example teams., Process all team files and compute revealed scores., Get the maximum number of relevant attributes per Pokemon for a generation., Count the number of revealed attributes that are "relevant" for this Pokemon., Compute the revealed score for this Pokemon.          Score = revealed_relevant_, Compute the revealed score for the entire team.          Score = (total revealed

### Community 66 - "Backend Team Prediction #66"
Cohesion: 0.12
Nodes (7): CurriculumMasker, NamesOnlyMasker, TeamSet, Masker classes for creating (x, y) training pairs by masking team attributes., Masker with curriculum: masking rate anneals from min to max over warmup steps., Mask a team. Returns (masked_x, ground_truth_y)., Toy masker: only masks Pokemon names.

### Community 67 - "Rl Metamon To #67"
Cohesion: 0.14
Nodes (7): PerceiverTurnEmbedding, TransformerTurnEmbedding, MetamonPerceiverTstepEncoder, MetamonTstepEncoder, Randomly set entries in the text component of the observation space to UNKNOWN_T, Token + numerical embedding for Metamon.      Fuses multi-modal input with atten, Efficient attention scheme for processing turn token inputs.      Uses latent cr

### Community 69 - "RL Evaluate Team Preview"
Cohesion: 0.18
Nodes (10): MatchupSpec, A single head-to-head matchup to run.      Attributes:         policy_a: The fir, Dry-run visualization for auto-evaluation configs.  Parses a config file and pri, Print the h2h matchup matrix using numbered indices., Print sweep matchups as a formatted table., Print self-play ladder agents as a formatted table.      Args:         agents: L, Format a raw config value (list / dict / scalar) as a compact string., Rich-formatted one-line detail string for a policy. (+2 more)

### Community 70 - "Rl Metamon To #70"
Cohesion: 0.15
Nodes (7): MetamonAMAGOExperiment, MetamonOnlineExperiment, Adds actions masking to the main AMAGO experiment, and leaves room for further t, Main-process-only atomic write so multi-GPU learners do not race on NFS., Copy ``MetamonAMAGOExperiment`` gin scope onto ``MetamonOnlineExperiment``., Online RL experiment with per-lane AMAGO policy temperature during collection., Re-apply gin after env init (opponent ``initialize_agent`` clears gin).

### Community 71 - "Backend Replay Parser #71"
Cohesion: 0.15
Nodes (6): CantIDSwitchIn, |detailschange|POKEMON|DETAILS|HP STATUS or |-formechange|POKEMON|SPECIES|HP STA, |replace|POKEMON|DETAILS|HP STATUS, |poke|PLAYER|DETAILS|ITEM, |switch|POKEMON|DETAILS|HP STATUS or |drag|POKEMON|DETAILS|HP STATUS, pokemon info from showdown `DETAILS` arg          https://github.com/smogon/poke

### Community 72 - "Backend Replay Parser #72"
Cohesion: 0.16
Nodes (6): Move, A wrapper around poke-env's Move object with its own pp counter, TODO/Dev note: Mimic is really hard from the replay POV and this isn't perfect., Fill unknown details based on the outputs of our team prediction module., Rank the available_moves by the fraction of health they recover, PEMove

### Community 73 - "Showdown Dex Data"
Cohesion: 0.19
Nodes (4): Dex, Any, This code is copied from poke-env:     https://github.com/hsahovic/poke-env/blob, Used to merge the results of a team prediction into existing team info.

### Community 74 - "Baselines Heuristic Basic #74"
Cohesion: 0.12
Nodes (9): BugCatcher, MaxBPBaseline, PokeEnvHeuristic, PreferPrioritySmart, picks fast moves when they can kill, (usually) picks the move with the highest base power, Calls the heuristic agent included in the poke-env repo, An actively bad trainer that always picks the least     damaging move. When forc (+1 more)

### Community 75 - "Data Parsed"
Cohesion: 0.18
Nodes (10): ParsedReplayDataset, PyTorch datasets for loading parsed Pokémon battle trajectories.  Classes:     M, Human replay dataset from jakegrigsby/metamon-parsed-replays.      Auto-download, Self-play dataset from jakegrigsby/metamon-parsed-pile.      Auto-downloads from, SelfPlayDataset, Profile VectorizedShowdownEnv step() throughput at high lane counts.  Usage:, throwaway: end-to-end VectorizedShowdownEnv smoke with a random opponent., DefaultActionSpace (+2 more)

### Community 76 - "Rl Evaluate Common #76"
Cohesion: 0.13
Nodes (8): ArgumentParser, Discover template variables in *config_path* and add them to *parser*.      Call, Extract resolved template variable values from parsed args., Scan a config file for ``${var}`` and ``${var:default}`` placeholders.      Only, Head-to-head evaluation launcher.  Usage:     python -m metamon.rl.evaluate.h2h, Shared launcher for challenge-based evaluation (h2h and sweep).  Takes a list of, Run a list of matchups with crash recovery and a thread pool.      Args:, Sweep evaluation launcher.  Usage:     python -m metamon.rl.evaluate.sweep \\

### Community 77 - "Pokemon Tokenizer"
Cohesion: 0.12
Nodes (4): Abra, First of a new series of training runs replicating the "Synthetic" agents from t, PokemonTokenizer, ndarray

### Community 78 - "Backend Replay Parser #78"
Cohesion: 0.23
Nodes (6): BackwardException, InvalidActionIndex, Exception, datetime, ReplayParser, ReplayState

### Community 79 - "Env Vectorized Action"
Cohesion: 0.16
Nodes (8): Battle, BattleOrder, Map metamon agent actions to Showdown ``BattleStream`` choice strings.  Because, Strip the ``/choose `` prefix to get a raw ``BattleStream`` choice., Rewrite ``switch <species>`` to ``switch <1-based slot>`` via the request., Convert an agent action index to a choice string, or ``None`` if invalid.      `, Per-lane battle state for the vectorized Showdown env.  A :class:`StreamBattleLa, throwaway: drive battles through StreamBattleLane using legal random actions

### Community 80 - "Rl Metamon To #80"
Cohesion: 0.25
Nodes (12): LearnablePosEmb, MultiModalEmbedding, PerceiverEncoder, Map integer tokens to a sequence of vector representations.      Just an `nn.Emb, Take the text embedding and add on a representation of the numerical     part of, TokenEmbedding, _BlockDiagPerceiverEncoder, _FastPerceiverEncoder (+4 more)

### Community 81 - "Rl Pretrained #81"
Cohesion: 0.22
Nodes (7): EnsembleMemberSpec, Configuration for one inference-only ensemble member., KakunaEnsemble, A prototype version of ensembling metamon policies.      Notably became the firs, Followup to KakunaEnsemble that also reached #1 on the Showdown leaderboard., Decorator to register pretrained model classes.      Args:         name: Optiona, TaurosEnsemble

### Community 82 - "Rl Pretrained #82"
Cohesion: 0.13
Nodes (9): LocalFinetunedModel, LocalPretrainedModel, OnlineTaurosV1, Online RL finetune of TaurosV0 from ``metamon.rl.online_rl`` run ``online_tauros, Online RL run ``mini_online_v1`` from ``metamon.rl.online_rl``.      Unlike ``On, Evaluate a model from a custom training run.      Args:         amago_ckpt_dir:, Evaluate a model from a finetuning run.      Same as LocalPretrainedModel but ta, SmallG1OnlineV0 (+1 more)

### Community 83 - "Backend Team Prediction #83"
Cohesion: 0.13
Nodes (6): PokemonSet, TeamSet, A simplified version of a Team that only tracks Pokemon names.      Used for sim, Returns a set of the Pokemon names in `other` that are not in `self`,         a., Fill in missing Pokemon names from a Roster., Roster

### Community 84 - "Backend Team Prediction #84"
Cohesion: 0.18
Nodes (4): ndarray, # TODO: speedup, Cached Vocabulary singleton (read-only after init)., TeamTokenizer

### Community 85 - "Heuristic Kaizo Baseline"
Cohesion: 0.18
Nodes (5): EKRisky, EmeraldKaizo, Battle, Based on Kaizo's Risky AI,     same as the normal AI but with preference for ris, Based on Emerald Kaizo AI, with bug fixes and modifications

### Community 86 - "Env Vectorized Team"
Cohesion: 0.17
Nodes (6): Any, Teambuilder, Build a ``>player`` spec dict and return the source team file (if any)., Vectorized Showdown battle env with an in-the-loop batched NN opponent.  N battl, Factory: one shared opponent sampled from an opponent pool config.      Each ful, Factory: vectorized Showdown env vs a metamon ``PretrainedModel`` opponent.

### Community 88 - "Env Metamon Battle #88"
Cohesion: 0.17
Nodes (6): Status, The main advantage that the online version has over the offline         (replay, Reveal information about (our team's) pokemon from a "side" request, Update the turn from a "side" request, and create self.available_switches, Update the turn from an "active" request, and create self.available_moves, Battle boilerplate, then call _update_turn_from_request

### Community 89 - "Env Metamon Player"
Cohesion: 0.17
Nodes (6): AbstractBattle, PokeAgentBackendBattle, # NOTE: we don't implement this turn counting system, Battle that will attempt to maintain backwards compatibility with the MetamonBac, Override the default battle creation logic to use our own MetamonBackendBattle., Override the default battle message handling logic to use our own MetamonBackend

### Community 90 - "Env Vectorized Package"
Cohesion: 0.17
Nodes (11): dependencies, pokemon-showdown, description, main, name, private, scripts, start (+3 more)

### Community 91 - "Rl Evaluate"
Cohesion: 0.17
Nodes (6): Build a PolicySpec from a per-policy config dict + defaults.      The policy nam, Expand a policy entry that may have a 'variants' list.      Without variants: re, Config parsing for head-to-head evaluation.  Example config:      battle_format:, Parse a head-to-head YAML config into a list of MatchupSpecs.      Generates all, Config parsing for sweep evaluation.  Sweep mode evaluates one policy across a p, Parse a sweep YAML config into a list of MatchupSpecs.

### Community 93 - "Baselines Init"
Cohesion: 0.24
Nodes (3): MatchupResult, Get all registered baseline names., Get a registered baseline class by name.

### Community 94 - "Data Parsed Replay #94"
Cohesion: 0.18
Nodes (5): Apply rating, date, and win/loss filters to a filename., Parse date string from filename., Partition filenames into train/test via deterministic seeded shuffle., Build the list of files to load, applying filters., DIRECTORY: Index and filter files from a flat directory.

### Community 95 - "Rl Metamon To #95"
Cohesion: 0.22
Nodes (5): MetamonFIFODataset, Online replay buffer backed by metamon ``json.lz4`` trajectories on disk.      V, Single ``scandir`` pass over the on-disk format dirs.          Returns ``(mtime,, Delete oldest on-disk trajectories until ``dset_max_size`` is satisfied., Mirror ``DiskTrajDataset.on_end_of_collection`` (amago.loading).          All ra

### Community 97 - "Baselines Heuristic Basic #97"
Cohesion: 0.20
Nodes (4): PreferPhysical, PreferSpecial, PreferStatus, (usually) picks the physcial move with the highest base power     otherwise, pic

### Community 99 - "Backend Team Prediction #99"
Cohesion: 0.39
Nodes (3): PokemonSet, TeamSet, A questionably efficient script that loads a set of team files and finds all uni

### Community 100 - "Baselines Heuristic Basic #100"
Cohesion: 0.31
Nodes (3): EasyPokeEnvHeuristic, _ParameterizedPokeEnvHeuristic, Makes the poke-env heuristic agent exploitable by introducing randomness in its

### Community 101 - "Rl Metamon To #101"
Cohesion: 0.28
Nodes (4): MetamonAMAGODataset, RLDataset, A wrapper around the ParsedReplayDataset that converts to an AMAGO RLDataset., RLData

### Community 103 - "Backend Team Prediction #103"
Cohesion: 0.25
Nodes (4): PokemonStatsLookupError, date, # TODO: this is probably not needed anymore but further changes will, KeyError

### Community 104 - "Data Parsed Replay #104"
Cohesion: 0.32
Nodes (4): Get path to our cached filename list for a tar archive., TAR: List all json files in archive using ratarmountcore.          This opens th, TAR: Load cached filename list from .txt file., TAR: Index and filter files from a tar archive.

### Community 105 - "Rl Metamon To #105"
Cohesion: 0.25
Nodes (4): BatchNormalizedExpFilter, Batch, Set the boolean mask for the next ``__call__``.          Args:             mask:, Batch-normalized exponential weighting for filtered behavior cloning.      Z-sco

### Community 106 - "Backend Team Prediction #106"
Cohesion: 0.43
Nodes (4): NaiveUsagePredictor, ABC, The original paper strategy. We use the names of the pokemon we have already kno, TeamPredictor

### Community 110 - "Discrete"
Cohesion: 0.29
Nodes (3): Discrete, Space, Return the observation space for this observation type.

### Community 111 - "Env Vectorized Vector"
Cohesion: 0.33
Nodes (3): Any, Refresh per-lane opponent obs/action spaces when the pool opponent changes., Swap the shared in-the-loop opponent (full env reset only).

### Community 112 - "Rl Pretrained #112"
Cohesion: 0.29
Nodes (4): Kadabra, Kadabra3, Kadabra4, A second attempt at self-play on gens1-4 & 9 that was featured in the PokéAgent

### Community 113 - "Backend Replay Parser #113"
Cohesion: 0.33
Nodes (3): UnimplementedMessage, |-burst|POKEMON|SPECIES|ITEM, |swap|POKEMON|POSITION

### Community 114 - "Backend Team Prediction #114"
Cohesion: 0.33
Nodes (3): PokemonSet, Mask a random subset of attributes (possibly all of them)., This loads large json files that are created by the `generate_replay_stats` scri

### Community 116 - "Rl Dataset Config #116"
Cohesion: 0.40
Nodes (3): MixtureOfDatasets with per-dataset initial/final weight control.      Used for i, Bidirectional linear anneal that clamps correctly for both         increasing (n, TransitionMixtureOfDatasets

### Community 120 - "Env Metamon Battle #120"
Cohesion: 0.40
Nodes (3): :return: The opponent's side conditions. Keys are SideCondition objects, values, :return: The player's side conditions. Keys are SideCondition objects, values, SideCondition

## Knowledge Gaps
- **304 isolated node(s):** `POVReplay`, `WarningFlags`, `ForwardException`, `ZoroarkException`, `UnfinishedMessageException` (+299 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **27 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What connects `POVReplay`, `WarningFlags`, `ForwardException` to the rest of the system?**
  _304 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Backend Team Construction Matchup` be split into smaller, more focused modules?**
  _Cohesion score 0.0 - nodes in this community are weakly interconnected._
- **Should `Vectorized AMAGO Policy & Opponent` be split into smaller, more focused modules?**
  _Cohesion score 0.0 - nodes in this community are weakly interconnected._
- **Should `Interface & Pretrained Models` be split into smaller, more focused modules?**
  _Cohesion score 0.0 - nodes in this community are weakly interconnected._
- **Should `Replay Parser Forward Simulation` be split into smaller, more focused modules?**
  _Cohesion score 0.0 - nodes in this community are weakly interconnected._
- **Should `Env Wrappers` be split into smaller, more focused modules?**
  _Cohesion score 0.0 - nodes in this community are weakly interconnected._
- **Should `RL Evaluate Harness` be split into smaller, more focused modules?**
  _Cohesion score 0.0 - nodes in this community are weakly interconnected._