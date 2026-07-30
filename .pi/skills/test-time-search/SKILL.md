---
name: test-time-search
description: "Correctness-first research and engineering guide for test-time Monte Carlo search on a frozen Gen1 OU Metamon policy (metamon.rl.experimental.test_time_search). Use when implementing, auditing, extending, benchmarking, or evaluating oracle root rollouts; Showdown snapshot/fork; branch RNG; Transformer recurrent-state branching; critic-backed rollout values; KL or magnetic policy improvement; paired search evaluation; opponent-model diagnostics; search gating; or the path from the current MiniOnlinePsroV1_4 prototype toward a credible superhuman Gen1 OU agent. Preserves the validated simulator-cloning work, identifies unresolved estimator risks, and defines the required tests, experiments, logging, and go/no-go gates for the next agent."
---

# Test-Time Search for a Frozen Gen1 OU Metamon Policy

This skill is the successor handoff for the **eval-only oracle root Monte Carlo
search** system implemented in:

```text
metamon/rl/experimental/test_time_search/
```

It is intentionally both a technical reference and an execution plan. A new
agent should be able to enter the repository, verify the current implementation,
make the required correctness changes, run the next experiments, and leave a
reproducible research report without re-deriving the project from scratch.

The guiding research objective is:

> Bolt test-time search onto a frozen, already-strong Metamon policy and use it
> to move toward a superhuman Gen1 OU agent, while preserving the existing fast
> Showdown simulator stack and avoiding a full game solver, MCTS, poke-engine,
> or policy retraining.

Gen1 OU is a **zero-sum, stochastic, hidden-information game**. Those three
properties must remain explicit throughout the work:

- **Zero-sum:** branch values and policy states must always use the correct
  player's perspective.
- **Stochastic:** search must integrate over future battle chance rather than
  accidentally seeing the simulator's hidden future random stream.
- **Hidden information:** the current search is intentionally an oracle over
  hidden teams and moves, but that oracle must not be conflated with an oracle
  over future randomness.

---

## 1. Executive state of the project

### What is already strong

The current implementation solved the hardest systems problem: safely branching
both the official Pokémon Showdown simulator and Metamon's recurrent policy
state from a live vectorized evaluation battle.

Validated components include:

- official `Battle.toJSON()` / `Battle.fromJSON()` simulator serialization;
- JSON-string snapshots to avoid aliased log arrays;
- Python `StreamBattleLane` deep copies combined with JS
  `replay_log=false` forks;
- snapshot/fork synchronization with `ping`/`pong`;
- batched branch-lane creation;
- Transformer KV-cache, RL² state, and time-index branching;
- policy-guided root rollouts;
- critic bootstrap with PopArt denormalization;
- a KL-anchored policy-improvement operator;
- an eval-only CLI with baseline and search modes;
- simulator fork and improvement-math tests.

This engineering should be preserved. Do not replace the validated clone path
with an approximate simulator or a new external engine.

### What is not yet established

There is **no credible evidence yet that search improves playing strength**.
The current headline comparison is:

| configuration | battles | win rate | interpretation |
|---|---:|---:|---|
| frozen baseline, no search | 200 | 0.50 | expected self-play reference |
| search: K=4, D=0, beta=1, prune=0.05, every fifth decision | 100 | 0.47 | statistically indistinguishable from baseline |
| same search configuration | 40 | 0.60 | too small to interpret |

The 100-game result had approximately:

- 1,026 searched roots;
- 23% root-argmax changes;
- mean policy KL around 0.18;
- roughly 195 ms latency per searched root after warmup.

That is enough to show that the system changes actions and runs at usable
latency. It is not enough to show that it changes actions correctly.

### Phase 0 implementation status (correctness-first pass)

A correctness-first implementation pass has been completed against the actual
codebase and **verified on GPU** (CUDA + checkpoint epoch 740). See
`metamon/rl/experimental/test_time_search/PROGRESS.md` for the full status and
`HANDOFF_GPU.md` for the remaining-item runbook. Summary:

**VERIFIED (tests pass, 71 passed on GPU / 61 passed + 11 skipped on CPU):**

- **Branch RNG (§7):** the future-chance oracle is fixed. `battle_host.js` +
  `sim_process.py` carry an optional per-branch 4×uint16 Showdown PRNG seed;
  `rng.py` `RootSeedBank` gives common-random-number seeds (shared across
  candidate actions per rollout index `k`, distinct across `k`,
  action-identity-independent); opponent root action is coupled per `k`;
  `inherited_trunk_rng` retained as a labeled diagnostic. Sim-level tests prove
  reseeding diverges on stochastic positions, the trunk PRNG is untouched, and
  the inherited mode reproduces the trunk future while resampled does not.
- **Exact leaf value (§10):** `V_pi = Σ_a π(a|h)·Q(h,a)` via one fixed-shape
  all-action critic call (compile-guarded); brute-force-equivalence + illegal
  masking + critic-disagreement tests pass.
- **Return accounting (§5):** three critical bugs fixed — terminal victory
  reward recorded once (was dropped), `reward_multiplier=10.0` applied (was
  missing), discount exponent `γ^(D+1)` (was off-by-one). `reward_multiplier` is
  now sourced from `agent.policy` (the MultiTaskAgent), where it actually lives.
- **Improvement operators (§11/§12):** `single_anchor_kl` (canonical, `kl_anchor`
  alias) + new `magnetic_kl` (uniform magnet, α/β); global advantage scale;
  per-root z-scoring demoted to a legacy mode. Constant-shift invariance,
  α=0 equivalence, and global-scale β stability all tested.
- **Policy-state fork (§8):** VERIFIED on GPU — forked branch == trunk actor
  probs, trunk never mutated by a fork advance, independent copies, correct
  sides, batched == scalar, seq_len saturates at 127 (sliding window).
- **Search plumbing (§8F/§19):** VERIFIED on GPU — `search_mode=none` ==
  baseline; `base_only` through-infra returns `pi_base`; no branch/snapshot
  leaks; `error_policy=raise` propagates after cleanup, `base_fallback` logs.
- **Config (§15):** research-safe defaults; `--legacy_prototype` restores the
  pre-correction config for reproducibility.

**Phase 0 is COMPLETE — the go/no-go gate (§21) is passed (VERIFIED on GPU).**

The last blocker — a `_pump_branches` settle timeout — is FIXED and VERIFIED:

- The §18G smoke eval (`error_policy=raise`, `all_legal`, `every_n=1`,
  `resample_crn`, `policy_expectation`) previously hit a `_pump_branches` settle
  timeout (`pump_until idle for 20.0s`) on ~9% of roots. **Root cause**
  (confirmed via a stall-state dump): when a branch reached eval=`wait` +
  opp=`forceswitch` (opp fainted during the root exchange), Showdown's
  `makeRequest` advanced both request serials together, but a `wait` side is
  never "answered", so the old `not other_advanced` guard left the opponent's
  follow-up forever unanswered -> host idle -> 20s timeout. The env's
  `_pump_settle` avoids this via the outer `_advance_lanes` loop the branch
  version lacked. **Fix** (`search_driver._pump_branches`'s `ready()`): answer
  opponent-only follow-ups whenever the eval side owes no decision (regardless
  of whether the eval `wait` serial advanced); re-answer single-side `|error|`
  re-prompts; never auto-answer a fresh eval move/force-switch (the park point).
  Regression tests: `tests/test_time_search/test_pump_branches.py` (2, GPU-gated).
  **Verification**: the §1C smoke command now completes 10 battles with
  `error_policy=raise`, 852 roots, 0 errors, 0 `base_fallback` lines; 73 tests
  pass on GPU. See `PROGRESS.md` "`_pump_branches` settle timeout — FIXED" and
  `HANDOFF_GPU.md` §1C for the full write-up.

The research-strategy risks 1–6 below have been addressed by the
implementation above; risk 6's "hide whether exhaustive search works" is
unblocked — exhaustive `all_legal` + `every_n=1` + `raise` now runs cleanly.

**Phase 1 (fixed-root estimator benchmark, §22) is IMPLEMENTED and the §22
go/no-go gate is PASSED on GPU** (ckpt 740; K_ref=256, derived K={4,16,64},
D={0,1}, 40 roots, ~28.7 min, verdict PASS 5/5). D=0 top-1 agreement with the
high-K reference rises monotonically 0.725→0.900→1.000 as K grows; regret falls
123→17→0; SE calibration `se_ratio`≈0.99 (block spread matches `std/sqrt(K)`,
confirming i.i.d. CRN reseeding); reference split-half stability = 1.000. D=1
adds variance not information at this scale (0.525→0.875 vs D=0's
0.725→1.000). See `PROGRESS.md` "Phase 1" for the full table + honest
limitations (early-phase-only corpus, self-play opponent, 40 roots) and the
next steps (span phases/mid-late + opponent-model matrix before any win-rate
sweep). Do not start a win-rate sweep (§23) until Phase 1 spans the §22
stratification space and the root-critic-only vs D=0/D=1 comparison is in.

**Phase 2 (paired+mirrored eval, §23) is IMPLEMENTED and ran to a screen.**
The headline result is **estimator-positive, game-negative** (skill §37): the
shaped-Q search changes actions in the estimator's preferred direction with
well-calibrated KL, but the paired win-rate delta is statistically
indistinguishable from baseline (sampling 500 pairs delta=-0.008 CI
[-0.066,+0.052]; argmax 300 pairs delta=+0.037 CI [-0.047,+0.117]; root-critic-
only 80 pairs delta=-0.038). See `PROGRESS.md` "Phase 2" for the full table +
the five candidate explanations.

**Phase A (terminal-win fixed-root benchmark — the §37 "Gate A: Is the critic
suitable for search?" go/no-go) is IMPLEMENTED.** The central unmeasured question
after the Phase 2 null was **objective alignment**: does the frozen shaped
critic's preference after an exact oracle transition predict which action
actually increases **terminal win probability** (not just shaped return)?
`search_driver.SearchEvalRunner.terminal_continuations()` plays one forced root
action to a terminal state with the frozen policy on both sides (reusing all
Phase 0 rollout infra; CRN-paired with the shaped-Q estimate on the same chance
stream `k`), and `terminal_win.py` runs the benchmark + analysis + gate + CLI.
A G=32 / 12-root preliminary pilot **PASSED Gate A (4/4)** with an encouraging
signal: D=0 K=ref Spearman vs terminal win = +0.246, and D=0 search reduces
terminal-win regret 0.126→0.089 (−29% vs the actor) — the first direct evidence
that the shaped objective IS partially aligned with winning (the Phase 2 null
was more about the KL update/sampling diluting the signal than objective
misalignment). The terminal-win estimator self-converges (term_G16 Spearman
0.679). **Honest caveat:** n=12, all early-phase, G=32 — a preliminary signal,
not a conclusion. The full G=128 / 80-root / phase-spanning run is the
go/no-go measurement; see `PROGRESS.md` "Phase A". Gate A PASS → proceed with
the existing shaped-critic evaluator (then Phase B/C/D); PARTIAL/FAIL → train a
terminal-outcome value head (§37 "Failure outcome").

### The main change in research strategy

Do **not** begin with a broad K × depth × beta win-rate sweep.

The next agent must first establish that the rollout estimator is legitimate and
convergent. The highest-priority risks are:

1. branches may inherit the trunk's exact hidden Showdown PRNG state, creating
   a future-chance oracle;
2. per-root z-score normalization can turn tiny noisy Q differences into large
   policy changes;
3. sampled-action leaf bootstrap adds avoidable variance;
4. the current policy-improvement operator is a single-anchor KL update, not the
   full magnetic operator used by Ataraxos;
5. 100-game unpaired evaluations are far too noisy for model selection;
6. policy pruning and `search_every_n=5` can hide whether exhaustive oracle
   search works at all.

These are not reasons to discard the implementation. They define the next
correctness-first phase.

---

## 2. Evidence labels: verified facts versus hypotheses

Every report and code comment should distinguish among the following:

### `VERIFIED`

Backed by a passing test, direct code inspection, or a completed reproducible
run.

Examples:

- JSON-string snapshots prevent the known `battle.log` aliasing bug.
- fork plus identical future actions reproduces the trunk state in the existing
  simulator tests.
- the frozen checkpoint loads with the documented architecture.
- the current K=4 search run produced a 0.47 win rate over 100 games.

### `LIKELY / MUST AUDIT`

Supported by architecture or documentation, but not yet demonstrated directly.

Primary example:

- Showdown serialization includes the battle PRNG state. If search forks that
  state without branch-only reseeding, the search is evaluating candidate
  actions under the actual hidden future RNG stream rather than an expectation
  over future chance. The current handoff does not document a reseeding step,
  so this is a high-probability issue, but the next agent must inspect the code
  and prove the actual behavior before claiming it is present.

### `HYPOTHESIS`

A research expectation to test, not an implementation fact.

Examples:

- exact policy-expectation leaf values will improve action-ranking stability;
- a uniform magnetic anchor will outperform a single policy anchor;
- deeper rollouts will help more at high-entropy roots;
- critic disagreement can identify turns where search is worth its latency.

Do not allow hypotheses to become undocumented defaults.

---

## 3. What the current search does

At every searched, settled decision for the evaluated player, the system
currently follows this high-level process:

1. Run the frozen actor and obtain a legal root-action distribution.
2. Prune low-probability legal actions according to the configured threshold.
3. Snapshot the trunk simulator lane as a JSON string.
4. Fork the evaluated player's and opponent's recurrent policy state into
   `A × K` branch lanes, where `A` is retained root actions and `K` is rollouts
   per action.
5. Force one evaluated-player root candidate in each branch.
6. Sample the opponent's simultaneous root action from a frozen rollout policy.
7. Settle the root interaction, including random outcomes, faints, forced
   switches, and re-prompts.
8. Optionally continue policy-guided play for `search_depth` additional settled
   evaluated-player decisions.
9. Accumulate discounted intermediate rewards.
10. Bootstrap nonterminal leaves with the frozen critic.
11. Average branch returns by root action.
12. Apply a policy-improvement operator anchored to the root actor policy.
13. Sample or greedily choose a live action from the improved policy.
14. Release snapshots and branch lanes, then continue the real trunk battle.

The current system is:

- **eval-only**;
- **oracle hidden-state** search;
- **root Monte Carlo**, not tree search;
- based on a **frozen actor and critic**;
- opt-in through `search_mode`;
- designed so `search_mode="none"` reproduces the frozen baseline path.

---

## 4. Hard research constraints

The next agent must preserve these constraints unless the user explicitly
changes the research direction:

- Do not integrate or depend on poke-engine.
- Do not implement MCTS.
- Do not build a full game-theoretic solver.
- Do not update, fine-tune, distill, or otherwise change actor or critic weights.
- Do not generate search-improved training targets in this phase.
- Do not train a belief model yet.
- Do not use inferred hidden opponent information yet.
- Do not replace official Showdown battle transitions with a damage-only or
  heuristic simulator for the primary search result.
- Do not silently change the validated snapshot/deepcopy/no-replay mechanism.
- Do not claim deployable hidden-information search from an oracle experiment.
- Do not claim a search gain from a noisy point estimate whose uncertainty
  includes zero.

Permitted work includes:

- branch-only RNG reseeding;
- exact critic expectation over legal leaf actions;
- policy-improvement operator changes;
- improved batching and static-shape inference;
- diagnostic opponent models;
- paired and mirrored evaluation;
- fixed-root estimator benchmarks;
- search gating and adaptive compute after exhaustive search works;
- belief-state search only after the oracle system passes its go/no-go gate.

---

## 5. Frozen checkpoint and return semantics

### Agent and checkpoint

- Agent: `MiniOnlinePsroV1_4`
- Registration: `metamon/rl/pretrained.py`
- Checkpoint epoch: **740**
- Expected checkpoint path:

```text
~/metamon_runs/mini_online_psro_v1.4/mini_online_psro_v1.4/ckpts/policy_weights/policy_epoch_740.pt
```

A corresponding `latest/policy.pt` may also exist.

The repository requires:

```bash
export METAMON_CACHE_DIR=/home/eddie/metamon_cache
```

### Architecture

The documented checkpoint uses:

- `V2AGroupedV2DataAblation`;
- approximately 14.6 million parameters;
- `TformerTrajEncoder`;
- 3 Transformer layers;
- `d_model=400`;
- 8 attention heads;
- FlashAttention-2;
- bfloat16 KV cache;
- `max_seq_len=128`;
- `MetamonMaskedResidualActor`;
- `NCriticsTwoHot`;
- 4 critics;
- 64 two-hot bins;
- `use_symlog=False`;
- `min_return=-100`;
- `max_return=2100`;
- PopArt enabled;
- `reward_multiplier=10.0`.

### Discount horizons

The policy exposes:

```python
policy.gammas = [0.1, 0.9, 0.95, 0.97, 0.99, 0.995, 0.999]
```

The primary search/evaluation horizon is currently the final entry,
`gamma=0.999`, unless `search_critic_horizon` selects another index.

### Reward

The documented environment reward is:

```text
AggressiveShapedReward =
    1.0 * (damage + hp)
  + 2.0 * (removed - lost)
  + 200.0 * victory
```

The critic return is trained in a scale that includes
`reward_multiplier=10.0`.

### Required audit: discount and reward units

Before changing the estimator, inspect the training trajectory construction and
verify all of the following:

1. whether gamma advances per vectorized environment transition, per settled
   player decision, per full turn, or according to another indexing rule;
2. whether forced-switch/re-prompt transitions consume discount steps;
3. whether the reward returned during root settlement already includes terminal
   victory reward;
4. whether terminal bootstrap must be exactly zero;
5. whether branch rewards are already multiplied by `reward_multiplier`;
6. whether PopArt output is being denormalized before it is combined with
   environment rewards;
7. whether the value is from the evaluated player's perspective at every leaf.

The search return must use the **same temporal and numerical convention as the
critic target**. A locally plausible discount formula is not enough.

Add tests that compare search return construction against the repository's
training/evaluation return builder on synthetic trajectories containing:

- a normal move turn;
- a unilateral forced switch;
- a double faint and two-sided switch;
- a terminal root action;
- a re-prompt cascade;
- a nonterminal rollout with two intermediate rewards.

---

## 6. Validated simulator snapshot and fork mechanism

This is the most valuable completed engineering work. Preserve it.

### Existing equivalence coverage

`tests/test_time_search/test_sim_fork.py` currently verifies that:

- fork plus identical future actions produces byte-equivalent relevant battle
  state;
- fork plus divergent actions produces independent divergence;
- branch simulation does not mutate the trunk;
- multiple phantom forks leave the live battle unchanged.

Relevant state includes:

- HP;
- status;
- PP;
- move lists;
- active requests;
- terminal outcome;
- battle-side and Pokémon state.

### Official Showdown serialization

The implementation uses `Battle.toJSON()` and `Battle.fromJSON()` from the
vendored Pokémon Showdown simulator. The serialized state includes the battle
object graph and, critically, the Showdown PRNG state.

### The log-aliasing bug and fix

A direct object snapshot can retain:

```javascript
state.log = battle.log
```

as a shared reference. A branch reconstructed from that object can therefore
mutate or append to the trunk's log.

The validated fix is:

```text
JSON.stringify(Battle.toJSON())
    -> store/pass a JSON string
    -> Battle.fromJSON(string)
    -> JSON.parse creates fresh arrays
```

Do not revert to object snapshots without a new equivalence and aliasing test.

### Request-regeneration quirk and fix

The deserializer regenerates `activeRequest` via `getRequests`; it does not
faithfully replay the exact previously emitted request object. Move PP and move
ordering can therefore differ from the live request at the snapshot point.

The validated search path is:

1. `copy.deepcopy` the Python `StreamBattleLane`;
2. fork the JS battle from the JSON-string snapshot;
3. use `replay_log=false`;
4. emit only new branch log entries.

This keeps the Python lane and JS battle aligned without regenerating and
replaying stale requests.

### Host synchronization

Fork and choose are separate stdin commands. Without synchronization, a branch
`choose` can race ahead of branch creation.

`ShowdownSimProcess._sync()` performs a `ping`/`pong` round trip after:

- `snapshot`;
- `fork`;
- `fork_batch`;
- `restore`.

`ShowdownSimProcess.drain()` flushes pending host chunks before snapshotting so
the Python deep copy and JS snapshot correspond to the same settled point.

### Host commands

The host supports:

```json
{"cmd": "snapshot", "lane": 3, "snapshot_id": 17}
{"cmd": "fork", "snapshot_id": 17, "to_lane": 42, "replay_log": false}
{"cmd": "fork_batch", "snapshot_id": 17, "to_lanes": [{"lane": 42, "epoch": 1}], "replay_log": false}
{"cmd": "restore", "snapshot_id": 17, "lane": 3, "replay_log": false}
{"cmd": "release_snapshot", "snapshot_id": 17}
```

`replay_log=false` is the validated search path.

### Python transport

`ShowdownSimProcess` and `ShardedShowdownSimProcess` contain:

- `snapshot()`;
- `fork()`;
- `fork_batch()`;
- `restore()`;
- `release_snapshot()`;
- `_sync()`;
- `drain()`.

Current search should use `n_workers=1`, because trunk and branch lanes must
share the same worker unless explicit cross-worker snapshot transfer is added
and tested.

---

## 7. Critical unresolved issue: future-chance RNG leakage

This is the first issue the next agent must audit.

### Why this matters

Showdown snapshots serialize the battle PRNG state. If every search branch is
restored from the exact trunk snapshot and no branch-only reseeding occurs,
then candidate actions are not being evaluated over sampled future chance.
They are being evaluated under the trunk's actual hidden future random stream.

That can expose information that a legal player does not possess, including the
future realization of:

- move accuracy;
- damage rolls;
- critical hits;
- speed ties;
- full paralysis;
- sleep duration;
- freeze and thaw events;
- secondary effects;
- partial-trapping continuation;
- any other simulator RNG consumed after the root.

The experiment is intentionally an oracle over unrevealed opponent state. It
should **not** silently become an oracle over future randomness.

Ataraxos-style search does not provide a direct precedent for this issue when
its evaluated environment is deterministic after hidden setup. Pokémon requires
an explicit chance-sampling design.

### Required code audit

Inspect:

- `battle_host.js` snapshot/fork handlers;
- the vendored Showdown `Battle` and PRNG implementation;
- `search_driver.py` branch creation;
- all uses of `search_seed`;
- opponent and evaluated-player action-sampling RNG;
- any branch `Battle` constructor or `fromJSON` hooks.

Answer, in code comments and the final report:

1. Does a branch begin with the trunk's exact PRNG state?
2. Is the PRNG ever replaced or advanced before the first branch transition?
3. Are all K rollouts for one action currently identical in environmental
   chance conditional on sampled policy actions?
4. Are candidate actions compared under the same or different chance samples?
5. Does search ever observe information derived from the trunk's future PRNG?

### Required primary semantics: resampled chance with common random numbers

Implement a research-safe chance mode with the following semantics:

- the live trunk battle is never reseeded or mutated;
- each rollout index `k` receives an independent branch-only environment seed;
- the same rollout-index seed is reused across every candidate root action;
- different rollout indices use independent seeds;
- the seed bank is deterministically derived from:
  - global search seed;
  - battle identifier;
  - side;
  - root decision index;
  - rollout index;
  - stream kind;
- branch seeds and any seed hashes are logged;
- no branch seed depends on candidate action identity.

Conceptually:

```text
env_seed[root, k] is shared across candidate actions a
env_seed[root, k] != env_seed[root, k'] for k != k'
trunk_seed is untouched
```

This is a **common-random-numbers** design. It reduces variance when comparing
candidate actions because action `a1` and action `a2` are exposed to the same
initial chance stream for rollout `k`.

Divergent action paths may consume random draws in different orders. Perfect
coupling is therefore not guaranteed, but sharing the initial branch seed is
still preferable to independent chance samples per action and is far preferable
to exposing the actual trunk future.

### Root opponent-action coupling

At simultaneous root decisions, the opponent cannot condition its action on the
evaluated player's hidden candidate action. Therefore:

- sample one opponent root action per rollout index `k` from the opponent's root
  policy;
- reuse that sampled opponent action across all candidate evaluated-player root
  actions for the same `k`;
- do not independently resample the opponent root action for every candidate.

Conceptually:

```text
opp_root_action[root, k] is shared across candidate actions a
```

This is both semantically correct and variance reducing.

For unilateral decisions, forced switches, or branches where only one player is
prompted, do not fabricate an opponent action.

### Deeper rollout policy randomness

After the root, observations diverge. Deeper rollout actions may be sampled
from different distributions. Use deterministic keyed policy RNG streams so
that the same uniform variate is used at the same logical rollout index and
ply where practical.

A reasonable design is:

```text
u[player, root, k, rollout_step]
```

and inverse-CDF sampling from each branch's policy. This creates a paired policy
sampling stream without forcing the same action when branch distributions
differ.

Keep environment RNG and policy-sampling RNG as separate streams.

### Preserve an explicit chance-oracle diagnostic

It can be useful to retain the existing inherited-PRNG behavior as a labeled
ablation, for example:

```text
search_chance_mode = inherited_trunk_rng
```

This mode must never be the primary result. Label it clearly as a
**future-chance oracle diagnostic**. Comparing it with resampled chance can
quantify how much apparent performance came from RNG leakage.

### Required RNG tests

Add integration tests proving:

1. same snapshot + same branch seed + same actions gives identical branch
   trajectories;
2. same snapshot + different branch seeds produces varied stochastic outcomes
   on a targeted stochastic position;
3. all candidate actions for rollout index `k` receive the same initial branch
   seed;
4. rollout indices receive different seeds;
5. opponent root action is identical across candidate actions for a fixed `k`;
6. reseeding branches does not alter the trunk PRNG state;
7. repeated search calls do not advance the trunk PRNG before the selected live
   action is submitted;
8. inherited-PRNG and resampled-PRNG modes are distinguishable in diagnostics;
9. fixed global search seed reproduces the branch seed table exactly;
10. branch cleanup still occurs if reseeding fails.

Use targeted Gen1 positions involving accuracy, crits, damage ranges, paralysis,
and sleep so that tests actually exercise chance.

---

## 8. Policy recurrent-state branching

### Current mechanism

`branch_state.py` provides:

```python
fork_hidden(trunk_hidden, trunk_lane, n_branches, device)
```

It broadcasts one trunk lane's Transformer KV cache and sequence length into a
batched `TformerHiddenState`, using one `expand` plus `clone` per cache rather
than a Python loop over layers.

```python
make_branch_state(trunk_driver, trunk_lane, n_branches, device)
```

also copies:

- RL² state;
- step counts;
- any policy-driver recurrent metadata needed by inference.

### Required additional tests

Simulator equivalence is necessary but not sufficient. Add policy-state tests
that prove:

1. trunk and forked policy states produce identical actor logits and critic
   outputs under identical future observations;
2. two branches receiving different observations diverge independently;
3. mutating or advancing a branch does not alter trunk KV cache, RL² state, or
   step count;
4. evaluated-player and opponent states remain assigned to the correct side;
5. branch batching and a scalar reference loop agree numerically;
6. repeated fork/cleanup cycles do not retain stale hidden state;
7. sequence lengths near the Transformer context boundary behave correctly.

Explicitly test roots at sequence lengths:

```text
127, 128, and 129
```

or the corresponding repository boundary semantics if the effective context
index differs. Verify that truncation, rollover, or cache reset behavior matches
the normal frozen-policy driver exactly.

### Base-only end-to-end equivalence

Add an end-to-end `base_only` test where search infrastructure is active but
the returned policy is the original actor policy.

Under a controlled policy RNG stream, prove that:

- legal-action probabilities match baseline;
- selected actions match baseline;
- recurrent states after the live action match baseline;
- battle outcomes match baseline in a deterministic CPU or controlled test
  configuration;
- no simulator snapshots or branch lanes remain allocated afterward.

GPU FlashAttention nondeterminism may prevent full bit identity in a large live
run. The unit/integration test should use the most deterministic supported path
without altering checkpoint semantics.

---

## 9. Root candidate actions: exhaustive first, pruning later

### Current behavior

The current default prunes legal actions whose probability is less than 5% of
the maximum legal actor probability. In the initial run this reduced an average
of roughly 6.9 legal actions to 2.2 retained actions.

That is a useful throughput optimization, but it is premature for the proof of
concept.

### Required proof configuration

For the first corrected oracle experiments:

- search every legal root action;
- do not use probability pruning;
- do not hard-cap root actions unless required to prevent a crash;
- search every non-forced evaluated-player decision;
- do not use `search_every_n=5`.

The purpose of the oracle phase is to answer:

> Can correct exhaustive root search improve this frozen policy at all?

A low-probability actor action can be exactly the tactical action search is
supposed to recover. Pruning it makes a negative result ambiguous.

### Logging required before pruning is restored

At every root log:

- number of legal actions;
- actor probability of each action;
- cumulative probability mass retained by a proposed pruning rule;
- whether the reference-best action would have been pruned;
- reference regret caused by pruning;
- latency saved.

### Later pruning policy

After exhaustive search shows a held-out gain, prefer cumulative-mass pruning
with a minimum number of actions, for example:

```text
retain actions until cumulative actor mass >= 0.99
retain at least 2 legal actions
always retain any action required by an explicit safety rule
```

Do not adopt `0.99` as a magic constant. Select it from the fixed-root benchmark
by measuring false exclusion of the reference-best action and wall-clock cost.

### Later adaptive allocation

Once K convergence is characterized, consider:

- successive halving;
- racing with confidence intervals;
- allocating more rollouts to close action pairs;
- stopping when the top action is statistically separated;
- actor-prior-guided but nonzero exploration allocation.

These are later efficiency features, not prerequisites for the first credible
result.

---

## 10. Leaf-value estimator

### Current behavior

The current `_leaf_values` path samples one leaf action per branch and evaluates
its critic value as an approximation to the state value.

This avoids dynamic-shape `torch.compile` recompilation storms, but adds
avoidable Monte Carlo variance exactly where the root action ranking is already
noisy.

### Required primary estimator

Compute the exact frozen-policy expectation at each nonterminal leaf:

```text
V_pi(h) = sum over legal actions a of pi(a | h) * Q(h, a)
```

Requirements:

- use the actor distribution from the same frozen policy state;
- score all legal actions with the frozen critic;
- respect the legal-action mask;
- average the four critics for the primary point estimate;
- log each critic head separately;
- log critic standard deviation, range, and optionally pairwise disagreement;
- use zero bootstrap at terminal leaves;
- preserve PopArt denormalization and selected gamma horizon;
- compute from the evaluated player's perspective.

### Static-shape batching

Do not return to one Python critic call per branch-action pair.

Use the repository's fixed action-space width. In a standard settled Gen1 OU
position, the maximum legal choices are typically four moves plus five bench
switches, but inspect the actual Metamon action representation and use its
constant rather than hard-coding assumptions.

A recommended implementation is:

1. build a `[num_branches, max_actions]` legal mask;
2. tile or broadcast leaf representations to a fixed
   `[num_branches, max_actions, ...]` action batch;
3. flatten to `[num_branches * max_actions, ...]`;
4. perform one or a small fixed number of critic forwards;
5. reshape to `[num_branches, max_actions, num_critics]`;
6. mask illegal actions;
7. multiply mean Q by actor probabilities and sum.

Add a test comparing this vectorized implementation with a simple scalar
brute-force loop on a small batch.

### Critic-only root baseline

Implement a no-simulator baseline:

```text
Q_root(a) = frozen critic Q(h_root, a)
```

Apply the same policy-improvement operator directly to all legal root actions.
Call this mode something explicit, such as:

```text
search_mode = root-critic-only
```

This ablation answers whether search gains come from:

- the critic already ranking root actions better than the actor;
- settling the real Showdown transition;
- deeper policy rollouts;
- or merely changing the policy distribution.

The minimum comparison set is:

1. frozen actor baseline;
2. root critic only;
3. depth-0 root settlement plus critic bootstrap;
4. depth-1 rollout;
5. depth-2 rollout if depth-1 is stable.

### Terminal and shaped-reward handling

For terminal branches:

- include rewards emitted by the environment exactly once;
- set bootstrap to zero;
- record terminal outcome and terminal step;
- verify that victory reward is not double counted.

For nonterminal branches, log the decomposition:

```text
Q_search = discounted_intermediate_reward + discounted_bootstrap
```

A search result is difficult to debug if only the final scalar is retained.

---

## 11. Value scale and normalization

### Disable per-root z-score normalization

The current default z-scores action values independently at each root before the
exponential policy update.

With only two retained actions, per-root z-scoring maps almost any nonzero value
difference to approximately `[-1, +1]`, regardless of whether the raw gap is
large and reliable or tiny and noisy. Consequences include:

- beta no longer has a stable global interpretation;
- small sign errors can cause large policy movement;
- action changes become insensitive to evidence magnitude;
- K scaling can fail to reduce behavioral volatility;
- a mean KL such as 0.18 can be driven by noisy rootwise rescaling.

Set the primary configuration to:

```text
search_value_normalization = false
```

Retain per-root z-score only as a labeled legacy ablation.

### Use a fixed global value scale

The policy-improvement temperature must be expressed in consistent value units.
The critic is PopArt-denormalized and trained with `reward_multiplier=10.0`, so
`beta=1.0` in raw critic units may be far too aggressive once root z-scoring is
removed.

Use one of these explicit modes:

1. **raw critic return units**;
2. **environment reward units**, dividing by the checkpoint's fixed reward
   multiplier;
3. **globally standardized advantage units**, dividing by a single robust scale
   estimated from the development root corpus and then frozen.

The recommended primary research mode is globally standardized units:

```text
A_scaled(root, action) = A_raw(root, action) / global_advantage_scale
```

where `global_advantage_scale` is computed once from the development root set,
for example using a robust standard deviation or median absolute deviation. It
must not vary by root, matchup, run, or candidate count after selection.

Log both raw and scaled values.

### Beta selection

Select one global beta on the development root corpus, not on held-out win rate.
A practical target is a modest distribution shift, such as a development-set
median KL in the neighborhood of 0.01 to 0.05, while preserving larger updates
when the evidence is genuinely strong.

This range is a tuning heuristic, not a performance claim. Report the full KL
distribution and action-change rate.

---

## 12. Policy-improvement operators

### Current single-anchor KL operator

The current operator is:

```text
A_search(a) = Q_search(a) - sum_a' pi_base(a') * Q_search(a')

pi_search(a) proportional to
    pi_base(a) * exp(A_search(a) / beta)
```

Because the baseline term is constant across actions, using `Q` or centered `A`
produces the same normalized distribution. Centering is useful numerically and
for logging.

Correct limiting behavior:

- `beta -> infinity`: `pi_search -> pi_base`;
- `beta -> 0+`: probability concentrates on the highest-Q action **within the
  support of `pi_base`**;
- equal Q values: `pi_search == pi_base`.

Do not describe the small-beta limit as an ordinary unanchored softmax. It is a
policy-prior-weighted concentration and cannot recover an action with exactly
zero prior probability unless a prior floor is explicitly applied.

Name this operator clearly:

```text
single_anchor_kl
```

It is a valid policy-improvement baseline and corresponds to the zero-magnet
special case of a two-anchor update.

### Ataraxos-style magnetic operator

Add an operator that solves:

```text
maximize over pi:
    E_pi[Q]
  - alpha * KL(pi || rho)
  - beta  * KL(pi || pi_base)
```

where:

- `pi_base` is the frozen actor policy at the root;
- `rho` is a fixed magnetic/reference distribution;
- `beta` controls policy anchoring;
- `alpha` controls the magnetic anchor.

The closed-form legal-action solution is:

```text
pi_search(a) proportional to
    rho(a) ** (alpha / (alpha + beta))
  * pi_base(a) ** (beta / (alpha + beta))
  * exp(Q(a) / (alpha + beta))
```

Use a uniform distribution over legal actions as the first magnetic reference:

```text
rho(a) = 1 / number_of_legal_actions
```

Implement in log space with legal masks and numerical floors.

Name it explicitly:

```text
magnetic_kl
```

Required ablations:

- `base_only`;
- `single_anchor_kl`;
- `magnetic_kl` with uniform legal magnet;
- `softmax_q` with no actor prior;
- `argmax_q`;
- optionally actor-prior sampling with no Q update as a plumbing control.

### Why the operator distinction matters

The current method should not be described as the full Ataraxos operator. The
Ataraxos ablations indicate that removing the policy KL anchor can be highly
destructive, while the additional magnetic term can provide a smaller but real
benefit. That motivates retaining the policy anchor as the default and adding,
not assuming, the magnet term.

### Root selection

Use sampling from `pi_search` as the primary policy-preserving selection rule.
Report greedy `argmax(pi_search)` and `argmax(Q_search)` as diagnostics and
ablations.

Sampling is especially important if the base policy is part of a mixed or
stochastic strategy in a zero-sum game. Greedy selection can make the policy
more exploitable even when it raises short-run value against the rollout model.

### Improvement tests

Expand `test_improvement.py` to cover:

- legal masking;
- exact anchor invariance for equal Q;
- beta limits;
- alpha limits;
- `alpha=0` equivalence between `magnetic_kl` and `single_anchor_kl`;
- uniform magnet behavior;
- zero and near-zero prior handling;
- numerical stability for large Q ranges;
- invariance to adding a constant to all Q values;
- global value scaling;
- no per-root z-score in primary mode;
- all ablation outputs summing to one;
- deterministic sampling under a fixed RNG stream.

---

## 13. Opponent rollout model

### Current model

The search rollout opponent is the frozen Metamon policy. In self-play this is
aligned with the live opponent. Against Tauros, Kakuna, earlier checkpoints, or
other policies, it becomes an opponent-model approximation.

### Root simultaneous-action requirement

At a simultaneous root:

- obtain the opponent policy from the opponent's own observation and recurrent
  state;
- sample one root opponent action per rollout index;
- reuse it across evaluated-player candidates;
- do not condition it on the candidate action;
- preserve the opponent's own side perspective.

### Required diagnostic opponent modes

Implement or expose the following evaluation-only rollout models:

1. **self model**
   - frozen `MiniOnlinePsroV1_4` models every opponent;
2. **live-opponent oracle model**
   - branch the actual evaluation opponent checkpoint and recurrent state;
   - valid only as a diagnostic because the evaluator knows the checkpoint;
3. **league-mixture model**
   - sample a rollout opponent from a defined policy mixture;
   - use only after single-policy estimator correctness is established;
4. **mismatched model ablation**
   - deliberately use the self model against a different live opponent.

The live-opponent oracle diagnostic separates two failure sources:

- the search estimator is bad even with a correct opponent model;
- the estimator is useful, but the rollout opponent model is mismatched.

Do not move to a sophisticated learned opponent belief until this distinction is
measured.

### Robustness opponents

At minimum evaluate against:

```text
TaurosV0, checkpoint 62
Kakuna, checkpoint 34
earlier MiniOnlinePsroV1_4 checkpoints such as 80 and 500
```

Confirm actual registered names and checkpoint availability in the repository
before running.

A method that only improves against its own rollout policy is not sufficient for
the broader superhuman objective.

---

## 14. Search depth semantics

### Current meaning

- `search_depth=0`: force and settle the root interaction, then bootstrap.
- `search_depth=1`: continue one additional policy-guided settled evaluated-player
  decision before bootstrap.
- `search_depth=2`: continue two additional settled evaluated-player decisions.

Verify this against the implementation; do not rely only on the intended
meaning.

### Required depth checks

For each branch log:

- number of raw simulator requests processed;
- number of environment transitions;
- number of evaluated-player decisions;
- number of opponent-only decisions;
- number of forced switches;
- discount exponent applied to each reward and bootstrap;
- terminal step if any.

Depth must not silently count forced-switch prompts differently from the frozen
policy's training sequence.

### Depth experiment interpretation

Deeper rollout can help by replacing critic bias with real transitions. It can
also hurt through:

- compounding opponent-model error;
- policy rollout variance;
- distribution shift in branch histories;
- incorrect recurrent-state advancement;
- incorrect discount accounting;
- rare re-prompt bugs;
- larger latency and fewer effective samples.

Compare depths at both:

- equal K;
- approximately equal wall-clock budget.

Do not conclude that depth is harmful merely because `D=2, K=4` loses to
`D=0, K=64` at radically different estimator variance.

---

## 15. Research-safe configuration defaults

The existing defaults reflect an early throughput prototype. The corrected
research defaults should be explicit and correctness oriented.

Recommended target configuration:

| setting | research default | reason |
|---|---|---|
| `search_mode` | `none` | search remains opt-in |
| `search_rollouts_per_action` | 16 | initial stable diagnostic budget |
| `search_depth` | 0 | isolate root settlement first |
| `search_every_n_decisions` | 1 | avoid arbitrary dilution |
| root candidate mode | `all_legal` | prove exhaustive oracle search |
| probability threshold | disabled | do not prune the winning tactic |
| root hard cap | disabled or action-space max | no silent support truncation |
| chance mode | `resample_crn` | no future-RNG oracle |
| root opponent coupling | enabled | same opponent action per k across candidates |
| leaf value mode | `policy_expectation` | remove sampled-action bootstrap variance |
| value normalization | `none` | preserve evidence magnitude |
| value scale | fixed global scale | make beta interpretable |
| improvement operator | `single_anchor_kl` | safe initial anchor baseline |
| magnet alpha | 0 initially | add as explicit ablation |
| root selection | `sample` | preserve mixed-policy behavior |
| critic horizon | primary 0.999 | match current frozen policy |
| intermediate rewards | enabled | use actual simulator outcomes |
| error policy | `raise` | research runs must not silently fall back |
| root logging | required | every corrected run must be auditable |

Do not silently alter old command behavior. Add explicit flags, migration notes,
and tests. Existing legacy configurations should remain runnable under labeled
modes where practical.

Suggested new config fields or equivalents:

```python
search_chance_mode: Literal[
    "resample_crn",
    "inherited_trunk_rng",
]
search_root_opponent_coupling: bool
search_leaf_value_mode: Literal[
    "policy_expectation",
    "sampled_action",
    "root_critic_only",
]
search_root_candidate_mode: Literal[
    "all_legal",
    "relative_threshold",
    "cumulative_mass",
]
search_value_scale_mode: Literal[
    "raw_return",
    "environment_units",
    "global_standardized",
]
search_global_advantage_scale: float | None
search_improvement_operator: Literal[
    "single_anchor_kl",
    "magnetic_kl",
    "softmax_q",
    "argmax_q",
    "base_only",
]
search_magnet_alpha: float
search_policy_anchor_beta: float
search_error_policy: Literal["raise", "base_fallback"]
search_log_branch_details: bool
```

Use repository naming conventions rather than copying these names mechanically
if a cleaner API already exists.

---

## 16. File map

Current search package:

```text
metamon/rl/experimental/test_time_search/
├── ARCHITECTURE.md
├── config.py
├── improvement.py
├── branch_state.py
├── search_driver.py          # search_root + _rollout_core + estimate_root (Phase 1) + terminal_continuations (Phase A)
├── eval_search.py
├── rng.py                    # Phase 0: deterministic keyed CRN seed bank
├── root_dataset.py           # Phase 1: fixed-root manifest + stratification features
├── benchmark_roots.py        # Phase 1: K/depth convergence benchmark + go/no-go
├── paired_eval.py            # Phase 2: mirrored paired evaluation helpers
├── terminal_win.py           # Phase A: terminal-win fixed-root benchmark + Gate A + CLI
├── analyze_roots.py          # root-result recovery / re-aggregation tool
├── PROGRESS.md
├── HANDOFF_GPU.md
└── __init__.py
```

Current tests:

```text
tests/test_time_search/
├── test_sim_fork.py
├── test_improvement.py
└── test_terminal_win.py      # Phase A: prefix-win-rate + Spearman/regret + Gate A (13 CPU + 1 GPU smoke)
```

Modified environment files:

```text
metamon/env/vectorized/battle_host.js
metamon/env/vectorized/sim_process.py
```

Recommended additional files, subject to repository style
(**all now implemented** — Phase 0 rng/leaf/return/improvement/cleanup tests
+ Phase 1 `root_dataset.py` / `benchmark_roots.py` / `test_root_benchmark.py`):

```text
tests/test_time_search/
├── test_branch_rng.py
├── test_policy_state_fork.py
├── test_leaf_values.py
├── test_return_accounting.py
├── test_search_equivalence.py
├── test_search_cleanup.py
└── test_root_benchmark.py

metamon/rl/experimental/test_time_search/
├── rng.py                  # deterministic keyed seed construction
├── root_dataset.py         # fixed-root capture/replay utilities
├── benchmark_roots.py      # estimator convergence benchmark
├── paired_eval.py          # mirrored paired evaluation helpers
└── schemas.py              # run/root JSONL schema if useful
```

Do not create files solely to match this list. Prefer coherent repository-local
organization.

---

## 17. Key code paths to inspect first

Read these before editing:

- `search_driver.py:SearchEvalRunner.search_root`
  - snapshot, branch layout, root-action forcing, rollout, return aggregation,
    improvement, selection, cleanup;
- `search_driver.py:_pump_branches`
  - settling logic, readiness predicates, forced switches, re-prompts;
- `search_driver.py:_leaf_values`
  - critic bootstrap, sampled action, PopArt, horizon selection;
- `branch_state.py:fork_hidden`
  - KV-cache broadcast and clone behavior;
- `improvement.py:improve_policy`
  - masks, normalization, beta semantics, ablations;
- `eval_search.py:run_search_eval`
  - policy loading, opponent setup, search cadence, error fallback, metrics;
- `metamon/env/vectorized/battle_host.js`
  - snapshot/fork/restore and Battle deserialization;
- `metamon/env/vectorized/sim_process.py`
  - sync, drain, lane epochs, sharding assumptions;
- the normal vectorized environment `_pump_settle`
  - reference semantics for branch settlement;
- the training return/trajectory builder
  - discount-step and reward-scale truth;
- the policy driver
  - actor/critic calls, recurrent-state advancement, action masks;
- `ARCHITECTURE.md`
  - original design intent and assumptions.

Also inspect the exact vendored Showdown version's PRNG implementation before
adding reseeding. Do not assume upstream current APIs match the vendored copy.

---

## 18. Tests and validation commands

### Existing test command

The repository may not include pytest in `pyproject.toml`:

```bash
cd /home/eddie/repos/metamon
uv pip install pytest
METAMON_CACHE_DIR=/home/eddie/metamon_cache uv run python -m pytest \
  tests/test_time_search/ -q -p no:cacheprovider
```

The original handoff expected 14 tests. The next handoff must report the new
exact count and duration.

### Formatting

The repository's existing CI primarily enforces black:

```bash
uv run black \
  metamon/rl/experimental/test_time_search/ \
  tests/test_time_search/ \
  metamon/env/vectorized/sim_process.py
```

Also run whatever lint, type, or compile checks are available in the current
checkout. At minimum:

```bash
uv run python -m py_compile \
  metamon/rl/experimental/test_time_search/*.py

git diff --check
```

### Required validation groups

#### A. Simulator fork

- existing identical/divergent/trunk-isolation tests;
- RNG reseeding tests;
- snapshot release and lane reuse;
- failure cleanup;
- stochastic Gen1 targeted positions.

#### B. Policy-state fork

- identical logits/Q after identical observations;
- branch independence;
- trunk isolation;
- side correctness;
- batch versus scalar agreement;
- context-boundary behavior.

#### C. Leaf values

- exact policy expectation versus brute-force loop;
- legal masking;
- terminal zero bootstrap;
- PopArt denormalization;
- multi-critic mean and disagreement;
- fixed-shape batching without compile proliferation.

#### D. Return accounting

- discount exponent agreement with training semantics;
- forced switches and re-prompts;
- intermediate reward decomposition;
- terminal reward counted once;
- evaluated-player sign.

#### E. Improvement math

- single-anchor and magnetic closed forms;
- alpha/beta limits;
- constant-shift invariance;
- support and prior-floor behavior;
- global scaling;
- numerical stability.

#### F. End-to-end equivalence

- `search_mode=none` baseline;
- `base_only` through search infrastructure;
- fixed RNG action identity;
- no branch leaks;
- no silent fallback.

#### G. Smoke evaluation

Before any long run, execute 2-10 battles with:

- exhaustive actions;
- resampled branch chance;
- exact leaf expectation;
- every-decision search;
- branch detail logging;
- `search_error_policy=raise`.

Inspect logs manually before scaling.

---

## 19. Error handling and cleanup

### Current risk

The evaluation path can fall back to the base action when search raises an
error. That is acceptable for a future production policy but dangerous in a
research run because it can make a broken search configuration look stable.

### Required behavior

Add or enforce:

```text
search_error_policy = raise
```

for all correctness and headline experiments.

If a fallback mode remains available, log:

- root identifier;
- exception type and message;
- traceback or structured error code;
- number of branch lanes created;
- number cleaned;
- whether the snapshot was released;
- selected fallback action;
- cumulative fallback count.

Any run with a nonzero search-error or fallback count must be labeled and cannot
serve as the primary result without a specific analysis.

### Cleanup invariants

Use `try/finally` around every root search. After completion or failure:

- all branch lanes are returned to the pool;
- all snapshots are released;
- no branch hidden state is retained;
- no pending host messages remain;
- trunk lane epoch is unchanged except for the live selected action;
- memory use remains bounded over repeated roots.

Add a stress test that performs hundreds of search-root create/cleanup cycles
and checks lane/snapshot counts and memory trends.

---

## 20. Required logging and run manifests

Every corrected search run must produce both:

1. a run-level manifest;
2. a root-level JSONL log.

### Run-level manifest

Record:

- timestamp;
- git branch;
- git SHA;
- whether the working tree is dirty;
- full diff or patch path;
- agent registration name;
- checkpoint path;
- checkpoint epoch;
- checkpoint SHA-256 or another strong hash;
- `METAMON_CACHE_DIR`;
- Python version;
- PyTorch version;
- CUDA version;
- GPU model;
- FlashAttention version;
- Pokémon Showdown vendored commit/version if identifiable;
- all search config fields;
- live opponent config;
- number of battles and mirrored pairs;
- battle/team/search seed ranges;
- exact command line;
- output paths;
- test results run before evaluation.

### Root-level JSONL schema

At minimum record:

#### Identity

- run ID;
- battle ID;
- pair ID;
- game within pair;
- evaluated-player side;
- team IDs or stable hashes;
- battle seed;
- search seed;
- turn number;
- evaluated-player decision index;
- policy sequence length;
- normal move, forced switch, or other request type.

#### Root policy

- legal action IDs;
- human-readable action labels;
- legal mask;
- base logits if practical;
- base probabilities;
- actor entropy;
- top-1/top-2 probability gap;
- candidate actions retained;
- retained probability mass;
- proposed-pruning diagnostics.

#### Branch sampling

- K;
- branch chance mode;
- branch environment seed or stable hash per rollout index;
- opponent root action per rollout index;
- policy RNG key per rollout index;
- branch lane IDs if useful for debugging;
- rollout depth reached;
- terminal indicator;
- timeout/re-prompt counts.

#### Values

For each candidate action:

- rollout returns by `k` when branch-detail logging is enabled;
- mean Q;
- standard deviation;
- standard error;
- confidence interval estimate;
- intermediate reward mean;
- bootstrap mean;
- terminal fraction;
- critic-head means;
- critic disagreement;
- raw advantage;
- scaled advantage.

#### Improvement

- operator;
- alpha;
- beta;
- value-scale mode;
- global scale;
- `pi_search`;
- KL(`pi_search || pi_base`);
- reverse KL if useful;
- total variation distance;
- whether actor argmax changed;
- whether sampled action changed relative to a coupled base draw;
- selected action;
- selection RNG key.

#### Performance

- snapshot latency;
- branch-fork latency;
- root settle latency;
- deeper rollout latency;
- leaf actor latency;
- leaf critic latency;
- improvement latency;
- total root latency;
- maximum allocated branch lanes;
- compile/recompile counters if available;
- search error/fallback status.

### Annotated example

Commit one small annotated JSONL example or a schema document. A future agent
should not need to infer field meaning from code.

---

## 21. Phase 0: correctness and estimator validity

This phase is mandatory before a large win-rate sweep.

### Phase 0A: RNG audit and correction

Deliver:

- documented answer to the five RNG audit questions;
- branch-only reseeding;
- common-random-number seed bank;
- root opponent-action coupling;
- separated environment and policy RNG streams;
- inherited-trunk-RNG diagnostic mode;
- all RNG tests passing.

### Phase 0B: recurrent-state correctness

Deliver:

- actor/critic branch-equivalence tests;
- branch/trunk isolation tests;
- side-perspective tests;
- sequence-boundary tests;
- batch/scalar agreement.

### Phase 0C: exact leaf expectation

Deliver:

- fixed-shape all-action critic scoring;
- exact `V_pi` leaf bootstrap;
- critic ensemble diagnostics;
- terminal handling;
- brute-force equivalence test;
- compile-shape instrumentation.

### Phase 0D: return-accounting audit

Deliver:

- documented training discount convention;
- reward scale and PopArt convention;
- terminal and sign tests;
- branch log decomposition.

### Phase 0E: policy improvement

Deliver:

- no per-root z-score in primary mode;
- fixed global scale support;
- corrected single-anchor semantics;
- magnetic operator;
- expanded unit tests.

### Phase 0F: plumbing equivalence

Deliver:

- `base_only` exact-equivalence test;
- `search_error_policy=raise`;
- cleanup stress test;
- smoke run with zero errors.

### Phase 0 go/no-go gate

**STATUS: PASSED (VERIFIED on GPU, ckpt epoch 740).** All boxes below are
met — 73 tests pass on GPU; the §18G smoke eval runs 10 battles with
`error_policy=raise`, 852 roots, 0 errors/fallbacks; RNG, leaf-expectation,
return-units, and root-logging checks are all VERIFIED (see `PROGRESS.md`
"Phase 0 go/no-go gate — PASSED"). Phase 1 (§22) may begin.

Do not run a broad search-strength experiment until:

- all required tests pass;
- no search error or fallback occurs in smoke evaluation;
- branch RNG is proven not to expose trunk future chance in primary mode;
- K rollouts produce actual stochastic diversity on stochastic roots;
- exact leaf expectation agrees with brute force;
- Q/reward units are documented;
- root logs contain enough information to reproduce one action decision.

---

## 22. Phase 1: fixed-root estimator benchmark

**STATUS: IMPLEMENTED; §22 go/no-go gate PASSED on GPU** (ckpt 740; K_ref=256,
derived K={4,16,64}, D={0,1}, 40 roots). `benchmark_roots.py` + `root_dataset.py`
+ `SearchEvalRunner.estimate_root` are built (99 tests pass). The benchmark
derives every lower-K from one high-K run per (root, depth) via prefix/block
averaging (the per-`k` branch seed is K-independent — `rng.py`), so the §22
K-sweep costs ~2 rollouts/root instead of ~10. The pilot's D=0 top-1 agreement
rises 0.725→0.900→1.000 and SE calibration `se_ratio`≈0.99 (i.i.d. confirmed).
**Remaining before a win-rate sweep (§23):** span the full §22 stratification
space (the pilot is early-phase-only / 40 roots / self-play opponent), add the
`root_critic_only` vs D=0/D=1 head-to-head to the gate table, and ideally scale
toward ~500 roots. See `PROGRESS.md` "Phase 1" for the full results + the
honest limitations. The spec below is the original plan (kept as the target).

The next scientific question is not yet “does search win more games?” It is:

> Does increasing rollout compute produce a more stable and more accurate root
> action ranking?

### Root corpus

Build a fixed oracle-root corpus of approximately 500 roots. The exact count can
change based on storage and replay cost, but it must be large and diverse enough
for subgroup analysis.

Stratify across:

- early, middle, and late battle;
- low, medium, and high actor entropy;
- large and small actor top-2 gaps;
- normal move decisions;
- voluntary switches;
- forced switches;
- one-sided and two-sided prompts;
- healthy and low-HP states;
- status-heavy positions;
- known and unrevealed opponent teams;
- likely stochastic positions;
- different team archetypes;
- both player sides;
- multiple opponent checkpoints.

Include targeted Gen1 tactical states involving:

- speed ties;
- paralysis;
- sleep;
- high-crit-rate moves;
- inaccurate moves;
- partial trapping;
- recovery;
- explosion/self-destruct;
- freeze risk;
- switching around predicted damage.

### Root reproducibility

Prefer reconstructing roots from:

- initial battle seed;
- exact teams;
- side assignment;
- complete public action history;
- opponent policy/checkpoint;
- evaluated policy/checkpoint;
- decision index.

If exact policy hidden-state serialization is robust, it may be stored as an
optimization, but replay-from-history should remain the truth source where
practical. Never rely on opaque pickles without version and checkpoint hashes.

### Reference estimates

Use two levels of reference:

#### Same-depth high-K reference

For every root and legal action, compute a large-K estimate at each tested depth,
for example `K_ref=256` or `512`, using a fixed common-random-number seed bank.
Choose the final K_ref based on convergence, not habit.

Check split-half stability of the reference itself.

#### Longer-horizon bias reference

On a smaller subset, continue substantially longer or to terminal when feasible.
This measures whether depth-0 or depth-1 estimates are stably biased even when K
is large.

### Candidate configurations

At minimum compare:

- root critic only;
- D=0 with K in `{4, 16, 64}`;
- D=1 with K in `{4, 16, 64}`;
- D=2 only after D=1 settlement and latency are stable;
- sampled leaf action versus exact policy expectation as an ablation;
- inherited trunk RNG versus resampled chance as a diagnostic;
- independent versus common-random-number candidate sampling if useful;
- single-anchor versus magnetic improvement after Q estimation is validated.

### Estimator metrics

Report by root and aggregate:

- top-action agreement with reference;
- top-2 pairwise ordering accuracy;
- Spearman and Kendall rank correlation;
- mean absolute Q error;
- simple regret under the reference estimate;
- probability that the chosen action is reference-best;
- stability of chosen action from K=4 to K=16 to K=64;
- standard error calibration;
- split-half top-action agreement;
- critic-head disagreement;
- terminal fraction;
- reward-versus-bootstrap contribution;
- latency;
- action-ranking accuracy per millisecond.

Stratify metrics by:

- actor entropy;
- actor top-2 gap;
- reference Q gap;
- critic disagreement;
- battle phase;
- action type;
- stochasticity category;
- opponent model.

### Expected convergence pattern

A viable Monte Carlo estimator should show, at least on roots with a meaningful
reference gap:

- lower standard error as K grows;
- higher top-action agreement as K grows;
- lower simple regret as K grows;
- less action-rank volatility as K grows.

If K=64 is not more stable than K=4, stop and debug. Do not proceed to a
win-rate sweep merely because one noisy K setting has a better point estimate.

### Phase 1 go/no-go gate

Proceed to end-to-end evaluation only if:

- the high-K reference is itself stable;
- increasing K improves estimator quality in the expected direction;
- exact leaf expectation is at least as stable as sampled-action bootstrap;
- resampled chance produces sensible uncertainty;
- the selected search operator does not cause large updates on near-tied noisy
  roots;
- root critic only, D=0, and D=1 have interpretable differences;
- there is a plausible wall-clock configuration for live evaluation.

---

## 23. Phase 2: paired and mirrored end-to-end evaluation

### Why the old protocol is insufficient

At a true win rate near 0.50:

- 100 independent games give a single-proportion 95% interval roughly
  `+/- 9.8 percentage points`;
- comparing two independent 100-game conditions gives an even wider interval on
  the difference;
- trying many K, depth, beta, and pruning settings on the same seed set will
  select noise.

As a rough independent-sample guide, obtaining a 95% confidence half-width of
about five percentage points on a difference near 0.50 takes on the order of
hundreds of games per condition; about three percentage points takes thousands.
A conventional 80%-power test for a true five-point difference can require
around 1,500 games per arm. Pairing can reduce the requirement substantially,
but the actual gain depends on the discordant-pair rate.

Do not treat these numbers as a substitute for a paired design.

### Paired battle unit

For each matchup unit:

1. fix exact team A, team B, battle seed, and policy/search seed schedule;
2. play search as player 1 against baseline as player 2;
3. swap sides with the same team pair and seed schedule;
4. record both outcomes as one mirrored pair;
5. compare search and baseline with paired statistics.

Where practical, also pair a search-enabled game with a baseline control using
the same initial battle setup. Actions will diverge, but initial conditions
remain controlled.

### Primary statistics

Report:

- search win rate;
- baseline win rate;
- paired win-rate delta;
- draws and the chosen draw scoring convention;
- number of mirrored pairs;
- number of discordant pairs;
- paired bootstrap confidence interval;
- exact or asymptotic McNemar test where appropriate;
- side-specific results;
- team-archetype subgroup results;
- latency and search-root count.

Do not report only a p-value. The effect size and confidence interval are the
primary result.

### Development and held-out split

Use:

- a development root corpus and development battle seed set for estimator and
  beta selection;
- a held-out confirmation seed set not used to choose K, depth, operator,
  global scale, pruning, or gating thresholds.

Pre-register the primary configuration before opening the held-out result.

### Suggested evaluation stages

#### Smoke

- 10-20 mirrored pairs;
- purpose: no crashes, no side asymmetry, logs complete;
- no strength conclusion.

#### Screen

- approximately 500 mirrored pairs for the small set of preselected modes;
- compare:
  1. baseline;
  2. root critic only;
  3. corrected D=0;
  4. best estimator-supported D=1;
- use confidence intervals and stop obviously harmful modes.

#### Confirm

- one primary search configuration;
- held-out seeds;
- approximately 1,500-2,500 mirrored pairs if needed, adjusted using observed
  discordance and a documented power calculation;
- no hyperparameter retuning after seeing the held-out result.

The exact counts should be based on observed paired variance and compute budget.
The key requirement is that the final claim be supported by a confidence
interval that resolves the practical effect.

### Phase 2 go/no-go gate

A credible positive result requires:

- a positive held-out paired effect;
- a confidence interval that excludes zero for the primary matchup or a clearly
  justified sequential criterion;
- zero unexplained search fallbacks;
- no dependence on inherited future RNG;
- no catastrophic side-specific failure;
- root diagnostics consistent with the estimator benchmark;
- a gain large enough to justify latency.

A 0.47 versus 0.50 result over 100 unpaired games is “no evidence,” not a
negative verdict and not a tuning target.

---

## 24. Phase 3: opponent-model diagnosis and robustness

After one corrected self-play configuration is selected, run a structured
opponent-model matrix.

### Matrix

For each live opponent:

- self-policy rollout model;
- exact live-opponent rollout model when technically available;
- later, league-mixture rollout model.

Live opponents should include:

- frozen baseline self-play;
- Tauros;
- Kakuna;
- earlier checkpoints of the same policy;
- the strongest available league members;
- eventually external agents or strong human replay positions where evaluation
  infrastructure exists.

### Interpretation

| self-model search | live-opponent-model search | likely conclusion |
|---|---|---|
| bad | bad | estimator/value/search operator problem |
| bad | good | opponent-model mismatch is primary |
| good | good | self model is adequate for this opponent |
| good | bad | implementation or recurrent-state mismatch; investigate |

### Robustness standard

A search method intended for superhuman play should not only exploit the exact
rollout policy. Prefer configurations that:

- improve against the frozen baseline;
- preserve or improve performance against diverse competent policies;
- do not create obvious deterministic exploitability through greedy selection;
- show sensible action changes on high-impact tactical turns.

---

## 25. Phase 4: compute scaling and efficiency

Only enter this phase after a positive corrected oracle result.

### Equal-compute comparisons

Compare configurations at:

- equal rollouts per action;
- equal branch transitions;
- equal wall-clock latency;
- equal GPU inference budget.

A deeper search with fewer rollouts and a shallow search with more rollouts are
both valid points on the compute frontier.

### Throughput targets

Profile separate stages:

- simulator snapshot;
- Python lane deepcopy;
- JS fork batch;
- root choose and settle;
- deeper policy rollout;
- actor inference;
- all-action critic inference;
- aggregation/improvement;
- cleanup.

Do not optimize from aggregate latency alone.

### Known current bottlenecks

- per-branch `copy.deepcopy(StreamBattleLane)` was measured around 1.6 ms in the
  original handoff;
- branch settling dominates simulator work;
- sampled leaf action avoided all-action compile storms but should be replaced
  by fixed-shape batching;
- `fork_batch` already reduces host synchronization overhead;
- search currently requires one simulator worker.

### Potential optimizations

After correctness:

- preallocate branch-lane objects;
- reduce Python deep-copy surface with a validated structural copy;
- reuse fixed branch buffers;
- use fixed action dimensions for actor/critic batching;
- compile only stable shapes;
- batch roots from multiple live battles when semantics permit;
- shard only if snapshot locality and lane routing are explicitly solved;
- use adaptive K/racing;
- use cumulative-mass pruning;
- search only high-value roots using a validated gate.

Every clone optimization must rerun the simulator and policy equivalence suite.

---

## 26. Phase 5: selective search and value of computation

Do not use arbitrary `search_every_n` scheduling as the final selective-search
method.

After exhaustive search works, train-free gates can use:

- actor entropy;
- actor top-1/top-2 probability gap;
- root critic action gap;
- critic ensemble disagreement;
- disagreement between actor argmax and root critic argmax;
- previous-turn tactical events such as faint, status, or revealed move;
- legal-action count;
- estimated search uncertainty from a small pilot batch;
- expected value of computation relative to latency budget.

A practical adaptive procedure is:

1. compute the root actor and all-action root critic;
2. skip search when both strongly agree and critic disagreement is low;
3. otherwise allocate a small pilot K to all legal actions;
4. estimate confidence intervals on action differences;
5. stop if the top action is separated;
6. allocate more rollouts only to unresolved candidates.

Evaluate the gate against exhaustive reference roots. Measure:

- fraction of roots searched;
- percentage of exhaustive-search value recovered;
- false-skip rate on roots where exhaustive search changes the best action;
- latency reduction;
- held-out win-rate impact.

---

## 27. Phase 6: hidden-information belief search

This phase is intentionally deferred.

The current oracle search branches from the exact hidden simulator state. It can
therefore use unrevealed opponent team members, moves, PP, and other hidden
state. This is useful as an upper-bound experiment, not as a deployable agent.

Proceed to belief-sampled search only if corrected exhaustive oracle search shows
a credible held-out gain.

A future belief phase would need:

- a posterior over hidden opponent states conditioned on public history;
- legal and internally consistent team/move samples;
- calibration and coverage metrics;
- branch allocation across both hidden-state samples and future chance samples;
- no leakage from evaluator internals;
- opponent-policy state consistent with each sampled hidden state;
- robust aggregation across beliefs, potentially including risk sensitivity.

The previous supervised belief-head experiment learned highly accurate labels
but did not produce a competitive policy when fed into the actor. That does not
prove belief sampling is useless for search, but it argues against adding belief
complexity before the oracle search estimator is validated.

### Oracle-to-belief go/no-go gate

Do not start belief implementation unless:

- corrected oracle search beats baseline on held-out paired evaluation;
- the gain is not caused by future RNG leakage;
- the estimator converges with K;
- the opponent-model failure mode is understood;
- latency leaves enough budget for hidden-state sampling;
- a clear belief-quality benchmark is defined.

---

## 28. CLI: current commands

Always set:

```bash
export METAMON_CACHE_DIR=/home/eddie/metamon_cache
```

### Frozen baseline

```bash
cd /home/eddie/repos/metamon
uv run python -m metamon.rl.experimental.test_time_search.eval_search \
  --agent MiniOnlinePsroV1_4 \
  --checkpoint 740 \
  --format gen1ou \
  --search_mode none \
  --total_battles 200 \
  --num_parallel 4 \
  --seed 42
```

### Legacy prototype search

```bash
uv run python -m metamon.rl.experimental.test_time_search.eval_search \
  --agent MiniOnlinePsroV1_4 \
  --checkpoint 740 \
  --format gen1ou \
  --search_mode oracle-root-mc \
  --rollouts_per_action 4 \
  --search_depth 0 \
  --search_beta 1.0 \
  --root_prob_threshold 0.05 \
  --search_every_n 5 \
  --total_battles 100 \
  --num_parallel 4 \
  --seed 42 \
  --search_log_roots /tmp/sr_k4_legacy.jsonl
```

This command is retained only to reproduce the existing result. It is not the
recommended next experiment.

---

## 29. CLI: target corrected commands

These match the **implemented** CLI (`eval_search --help`). The correctness
smoke run is the Phase 0 gate command — it now passes cleanly (the
`_pump_branches` settle-timeout blocker is FIXED; see `HANDOFF_GPU.md` §1C and
`PROGRESS.md`). The fixed-root benchmark and paired-eval commands reference
modules that are **not yet implemented** (Phase 1 §22, Phase 2 §23) and are
kept here as the target shape.

### Correctness smoke run (implemented; Phase 0 gate — PASSED)

```bash
uv run python -m metamon.rl.experimental.test_time_search.eval_search \
  --agent MiniOnlinePsroV1_4 \
  --checkpoint 740 \
  --format gen1ou \
  --search_mode oracle-root-mc \
  --rollouts_per_action 4 \
  --search_depth 0 \
  --root_candidate_mode all_legal \
  --search_every_n 1 \
  --search_chance_mode resample_crn \
  --root_opponent_coupling true \
  --leaf_value_mode policy_expectation \
  --search_value_normalization false \
  --search_ablation single_anchor_kl \
  --error_policy raise \
  --total_battles 10 \
  --num_parallel 4 \
  --seed 42 \
  --search_log_roots /tmp/tts_correctness_smoke.jsonl
```

Actual flag names: `--search_ablation` (not `--search_improvement_operator`),
`--error_policy` (not `--search_error_policy`), `--root_opponent_coupling`,
`--search_value_normalization` (bool: `false`/`true`), `--value_scale_mode`
(`raw`|`environment_units`|`global_standardized`|`legacy_zscore`),
`--magnet_alpha`, `--legacy_prototype` (restores the pre-correction prototype
defaults). Run `eval_search --help` for the full list.

### Fixed-root benchmark (IMPLEMENTED — Phase 1, §22)

The benchmark runs **in-battle** (no separate manifest required): it drives the
env with the baseline policy and, at each settled eval-side decision, runs the
estimator grid (`root_critic_only` + D=0/D=1 at `K_ref`) at the *same* trunk
state via `SearchEvalRunner.estimate_root`, derives every lower-K from the one
high-K run by prefix/block averaging (the per-`k` branch seed is K-independent —
`rng.py`), and writes `root_results.jsonl` + `root_manifest.jsonl` +
`summary.json` + `run_manifest.json` + `REPORT.md`. Actual flag names:

```bash
uv run python -m metamon.rl.experimental.test_time_search.benchmark_roots \
  --agent MiniOnlinePsroV1_4 --checkpoint 740 --format gen1ou --team_set competitive \
  --num_parallel 2 --seed 42 --search_seed 0 \
  --k_ref 256 --derived_ks 4 16 64 --depths 0 1 \
  --max_roots 40 --max_battles 40 --output_dir /tmp/tts_phase1_pilot
#   optional: --include_inherited_rng (future-chance oracle diagnostic, D=0)
#             --store_branch_matrices (keep per-branch R (A,K); verbose)
```

The high-K reference is the full-`K_ref` mean per (root, depth); convergence is
judged by whether K={4,16,64} prefix/block estimates move toward it (skill §22
gate). See `PROGRESS.md` "Phase 1" for the pilot result + go/no-go verdict.

### Terminal-win fixed-root benchmark (IMPLEMENTED — Phase A, the §37 Gate A)

The central go/no-go after the Phase 2 "estimator-positive, game-negative"
result: does the frozen shaped critic's preference after an exact oracle
transition predict which action increases **terminal win probability**? For each
legal action at each root it plays `G=K_ref` coupled continuations to a terminal
state (frozen policy both sides), records the actual win/loss, and correlates
the shaped-Q predictors against terminal win. CRN-paired with the shaped-Q
estimate on the same chance stream `k`. Writes `terminal_win_roots.jsonl` +
`root_manifest.jsonl` + `terminal_win_summary.json` + `terminal_win_REPORT.md` +
`run_manifest.json` + the Gate A verdict.

```bash
uv run python -m metamon.rl.experimental.test_time_search.terminal_win \
  --agent MiniOnlinePsroV1_4 --checkpoint 740 --format gen1ou --team_set competitive \
  --num_parallel 2 --seed 42 --search_seed 0 \
  --k_ref 128 --derived_ks 4 16 64 --depths 0 1 \
  --max_roots 80 --max_battles 240 --decision_stride 3 \
  --max_steps_to_terminal 250 --store_per_branch \
  --output_dir /tmp/tts_phaseA_run
#   --decision_stride spreads the corpus across early/mid/late (§22 phase bands)
#   --store_per_branch keeps wins (A,G) + R (A,K) for paired diagnostics
#   G=K_ref so terminal-win and shaped-Q share the CRN seed bank (§7)
```
Gate A (4 criteria): `correlated` (Spearman>0), `improves_over_actor`
(regret<actor), `not_catastrophic` (decrease freq<50%), `converges_with_k`.
PASS → proceed with the existing evaluator (Phase B/C/D); PARTIAL/FAIL → train
a terminal-outcome value head (§37). See `PROGRESS.md` "Phase A" for the
preliminary pilot (G=32/12-root PASS, Spearman +0.246, regret 0.126→0.089).

### Paired evaluation (NOT YET IMPLEMENTED — Phase 2, §23)

```bash
uv run python -m metamon.rl.experimental.test_time_search.eval_search \
  --agent MiniOnlinePsroV1_4 \
  --checkpoint 740 \
  --opponent_agent MiniOnlinePsroV1_4 \
  --opponent_checkpoint 740 \
  --format gen1ou \
  --search_mode oracle-root-mc \
  --rollouts_per_action 16 \
  --search_depth 0 \
  --root_candidate_mode all_legal \
  --search_chance_mode resample_crn \
  --leaf_value_mode policy_expectation \
  --search_value_normalization false \
  --search_ablation single_anchor_kl \
  --error_policy raise \
  --paired_mirrored_eval true \
  --num_pairs 500 \
  --seed_manifest /path/to/heldout_seed_manifest.jsonl \
  --search_log_roots /path/to/paired_search_roots.jsonl \
  --run_manifest /path/to/paired_search_run.json
```

Do not implement flags that imply unsupported guarantees. For example,
`paired_mirrored_eval` must truly preserve exact teams and mirrored side
assignment.

---

## 30. SearchConfig: current documented knobs

The original prototype documented:

| knob | old default | meaning |
|---|---:|---|
| `search_mode` | `none` | opt-in search |
| `search_rollouts_per_action` | 16 | K per retained root action |
| `search_depth` | 1 | additional settled evaluated-player decisions |
| `search_beta` | 1.0 | policy anchor temperature |
| `search_rollout_temperature` | 1.0 | rollout actor temperature |
| `search_root_selection` | `sample` | sample or argmax |
| `search_critic_horizon` | `None` | primary gamma when unset |
| `search_lane_batch_size` | 64 | simultaneous branch cap |
| `search_seed` | 0 | search RNG seed |
| `search_every_n_decisions` | 1 | search cadence |
| `search_log_roots` | `None` | JSONL path |
| `search_max_root_actions` | `None` | cap after pruning |
| `search_root_prob_threshold` | 0.05 | relative actor-probability pruning |
| `search_policy_prior_floor` | 0.0 | floor before anchoring |
| `search_include_intermediate_rewards` | `True` | MC return versus leaf only |
| `search_value_normalization` | `True` | per-root z-score |
| `search_ablation` | `kl_anchor` | improvement mode |

The next agent should preserve legacy reproducibility while introducing the
research-safe semantics described above.

---

## 31. Fixed-root benchmark implementation notes

### Capture format

A root manifest should contain enough information to recreate the exact public
trajectory and oracle simulator state, including:

- battle format;
- exact teams or team hashes with a secure local lookup;
- initial Showdown seed;
- side assignment;
- sequence of submitted actions;
- root decision index;
- evaluated and opponent policy checkpoint hashes;
- search-relevant observation hash;
- expected legal action set;
- optional simulator snapshot hash;
- optional policy-state hash.

Do not commit private or licensed team data to a public repository without
checking project policy.

### Root replay validation

When a root is reconstructed, verify:

- simulator-state hash;
- public observation hash;
- legal actions;
- actor logits within tolerance;
- critic outputs within tolerance;
- recurrent sequence length;
- active request type.

Reject a root that no longer reproduces after code changes. Do not silently use
an approximate reconstruction.

### Reference uncertainty

For each action, compute:

```text
mean, sample variance, standard error, confidence interval
```

using independent rollout-index seeds. Because candidate actions use common
random numbers, action differences should also be estimated directly:

```text
Delta_k(a, b) = return_k(a) - return_k(b)
```

Then report the paired standard error of `Delta`, which is often more useful
than separate action standard errors.

### Root-level regret

Given reference action `a*`:

```text
regret = Q_ref(a*) - Q_ref(a_selected)
```

Use this to compare pruning, K, depth, and gating decisions. Win-rate evaluation
alone does not reveal whether a configuration is making locally better choices.

---

## 32. Statistical analysis implementation notes

### Confidence intervals

Use:

- Wilson intervals for individual win rates;
- paired bootstrap intervals for mirrored-pair deltas;
- confidence intervals on root action-value differences;
- bootstrap over battles or pairs, not over individual search roots within one
  battle, when estimating game-level strength.

Search roots within a battle are correlated and must not be treated as
independent games.

### Multiple comparisons

A large K × D × beta × operator × opponent sweep can produce false winners.
Control this by:

- using root-estimator metrics for most configuration selection;
- limiting end-to-end screening to a small predefined set;
- maintaining a held-out confirmation set;
- reporting every attempted configuration;
- avoiding post-hoc seed selection;
- optionally using false-discovery or family-wise corrections for exploratory
  tables, while keeping one primary hypothesis.

### Sequential evaluation

Sequential stopping is acceptable if the rule is defined before the run. For
example:

- evaluate in blocks of 100 mirrored pairs;
- stop a harmful configuration if the upper confidence bound is below the
  minimum acceptable effect;
- stop for success only when a predeclared confidence or e-value criterion is
  reached;
- cap at a fixed maximum pair count.

Do not repeatedly inspect ordinary p-values and stop opportunistically.

---

## 33. Diagnostics for action changes

For every changed action, make it possible to answer:

- what was the actor's original ranking?
- what did the root critic rank?
- what did D=0 rank?
- what did deeper search rank?
- how large was the value gap?
- how uncertain was the gap?
- did all critic heads agree?
- was the branch result driven by immediate damage, a faint, status, switch
  value, or bootstrap?
- would the old pruning rule have excluded the selected action?
- did inherited future RNG choose a different action from resampled chance?
- did the live-opponent model disagree with the self model?
- did the action improve the final paired outcome?

Build a small human-readable root report generator. For a sampled set of roots,
render:

```text
turn / teams / visible state
legal actions and base probabilities
Q mean +/- SE
critic disagreement
pi_search
selected action
branch outcome summary
latency
```

This is essential for discovering systematic Gen1 errors that aggregate metrics
hide.

---

## 34. Gen1-specific stress cases

Create targeted regression scenarios for mechanics that are unusually important
in Gen1 OU:

- 1/256 accuracy misses where represented by the simulator version;
- high critical-hit rates tied to base speed;
- critical hits interacting with stat changes and screens;
- permanent freeze unless thaw mechanics apply in the chosen ruleset;
- sleep duration and wake-up turn behavior;
- full paralysis;
- partial trapping and switch restrictions;
- Hyper Beam recharge behavior after a KO in Gen1;
- Explosion/Self-Destruct defense-halving behavior;
- Counter and damage-history dependencies;
- Substitute interactions;
- speed ties;
- simultaneous faints;
- PP decrement/request-order quirks;
- forced-switch ordering;
- Transform and copied move state if present;
- recovery moves and the Gen1 failure edge cases;
- status and stat-overflow quirks supported by Showdown.

Do not reimplement these mechanics. Use official Showdown transitions and
construct tests that prove branch equivalence and chance diversity through them.

---

## 35. Known limitations and gotchas

### FlashAttention-2 nondeterminism

Same-seed GPU runs may not be bit-identical. Setting cuDNN deterministic mode
does not necessarily make FlashAttention deterministic. Disabling
FlashAttention may break checkpoint loading or alter inference semantics.

Mitigation:

- use controlled CPU/small deterministic paths for unit tests where possible;
- use paired large-sample evaluation for strength;
- record software and hardware versions;
- do not claim exact live-run reproducibility from seed alone.

### `num_parallel=1`

The single-battle environment may return unbatched observations. The current CLI
clamps `num_parallel >= 2`. Preserve or explicitly fix this behavior.

### `n_workers>1`

Search currently assumes branch lanes live on the trunk worker. Do not enable
sharded search until worker-local snapshot routing is implemented and tested.

### Depth settlement

`_pump_branches` mirrors `_pump_settle` + `_advance_lanes` (FIXED in Phase 0;
see `PROGRESS.md` "`_pump_branches` settle timeout — FIXED"). The old
`ready()` predicate stalled on the both-sides-advanced case where the eval side
is `wait` and the opponent has a `forceswitch` (e.g. the opponent fainted during
the root exchange): a `wait` side is never "answered", so the old
`not other_advanced` guard left the opponent forever unanswered and the host
went idle. The fix answers opponent-only follow-ups whenever the eval side owes
no decision and re-answers single-side `|error|` re-prompts. Treat any *new*
timeout as a correctness issue, not normal noise — re-enable the stall dump
described in `HANDOFF_GPU.md` §5 to capture the lane state.

### Host `lanes` Map growth (the 60s *total*-timeout class)

A **second, distinct** timeout class appeared in the Phase 1 phase-spanning
run: `pump_until timed out after 60.0s` (the *total* deadline, not the 20s
*idle* deadline) with **empty stderr** and a crash at ~root 130. This is NOT
the Phase 0 settle-logic stall (which is a 20s *idle* timeout — the host
produces no output). A 60s *total* timeout with non-idle host means the host is
producing output but a single pump round-trip for all active branches exceeds
60s.

**Root cause:** the host `lanes` Map grew unboundedly. Test-time search
allocates ever-incrementing fork-lane ids that are never recycled, so without
an explicit delete the Map accumulated ~A×K entries per searched root per
config (~2300/root here). By ~130 roots the Map held ~300k stale `Lane`
shells; every `handleCommand` / `getLane` lookup degraded, and a single pump
round-trip for ~1152 branches (K_ref=128 × A=9) eventually exceeded 60s.

**Fix** (`battle_host.js` `handleCommand("destroy")`): `lanes.delete(
msg.lane)` after `lane.destroy()`, so the Map stays bounded. `getLane()`
re-creates a fresh `Lane` on demand (fork ids are never reused, so a deleted
id is never referenced again). **Verified by elimination:** with the fix, a
K_ref=4 run sailed past root 130 to 200 roots (168s, no crash), a K_ref=128
measurement held pump times to ≤12.7s over 1024 branches, and the K_ref=128
200-root phase-spanning run held pump times to ≤21s with zero timeouts —
confirming the crash was host-side capacity (Map growth), not a settle-logic
bug (which would reproduce at any K, including K_ref=4).

**Diagnostics** (`search_driver._pump_branches`): `TTS_STALL_DUMP=1` dumps
per-lane state on timeout; `TTS_PUMP_TIMEOUT` (default `60.0`s) configures the
total deadline (set higher, e.g. `120`, for high-K runs as a safety margin);
`TTS_PUMP_VERBOSE=1` logs every pump's `dt` + `n_active`. Interpretation: a
timeout with **all** lanes in a consistent waiting state + empty stderr + high
`n_active` is a throughput/capacity issue (Map growth or branch count), NOT a
settle-logic bug; a timeout with **one** lane in an unhandled state (e.g. both
sides `needs` but `not decision_ready`, or an uncleared `error`) is a logic
bug — use the stall dump to tell them apart.

### Sim-fork equivalence flake (pre-existing, load-sensitive)

`test_sim_fork.py::test_fork_same_actions_equivalent` intermittently fails
(~15-25% of full-suite runs on GPU; higher under concurrent GPU load such as a
background benchmark run) with "fork diverged under identical actions" — a
sim-level PRNG/move-ordering drift in the vendored Showdown fork path under
GPU-test load. It does **not** use the search driver (it tests the raw
`sim_process` fork path); it passes stably in isolation. It is **not** a Phase 0
or Phase 1 blocker: the full suite is a clean **99 passed** on GPU with no
concurrent load (73 Phase 0 + 25 CPU Phase 1 + 1 GPU benchmark smoke); a flaked
run is re-run. The flake is load-sensitive and worsens with each additional
`frozen_env_bundle` consumer in the suite (`test_pump_branches.py`, then the
Phase 1 `test_root_benchmark.py` GPU smoke) — but it is the *same* pre-existing
sim-fork nondeterminism, not a regression from the search/benchmark code.
Investigating the sim-fork PRNG drift is a separate item.

### Request PP quirk

A freshly regenerated request can show stale or differently ordered PP. Preserve
Python deepcopy plus `replay_log=false`.

### Transport counters

`lastMoveLine`, `sentLogPos`, and `sentEnd` are transport/annotation counters,
not core battle state. They may be excluded from state equivalence if their
irrelevance is documented and tested.

### Actor support

A KL-anchored operator cannot recover an exactly zero-probability legal action
without a prior floor. Confirm whether the masked softmax ever produces exact
zeros for legal actions. Use a tiny numerical floor only if needed and log it.

### Search changes the policy distribution

Even a locally higher estimated action value can make a stochastic zero-sum
policy more exploitable. Preserve sampling and test against diverse opponents.

### Oracle hidden state

All current rollouts know the exact hidden opponent state. Never report the
oracle win rate as deployable performance.

---

## 36. Recommended implementation order for the next agent

Work in this order unless a discovered blocking defect requires adjustment.

### Step 1: establish repository state

- record branch, SHA, dirty status;
- locate current search commits and uncommitted changes;
- run existing tests;
- reproduce one baseline and one tiny legacy search run;
- hash the checkpoint;
- inspect current logs and CLI behavior.

### Step 2: audit branch RNG

- trace Showdown PRNG through snapshot/fork;
- write a failing targeted test if future-RNG inheritance is present;
- implement branch-only reseeding and common random numbers;
- couple root opponent actions across candidates;
- add inherited-RNG diagnostic mode;
- make all RNG tests pass.

### Step 3: audit policy-state branching

- add actor/critic equivalence tests;
- add sequence-boundary tests;
- fix any cache or side mismatch before changing estimator math.

### Step 4: implement exact leaf expectation

- add fixed-shape all-action critic evaluation;
- compare with brute-force scalar implementation;
- log critic heads and disagreement;
- remove sampled-action leaf mode from the primary path.

### Step 5: audit return semantics

- match training discount and reward units;
- add forced-switch and terminal tests;
- log reward/bootstrap decomposition.

### Step 6: improve policy update

- disable per-root normalization in primary mode;
- add fixed global scaling;
- rename current operator to `single_anchor_kl`;
- add `magnetic_kl`;
- expand unit tests.

### Step 7: harden research execution

- add error-policy flag with `raise` default for research;
- add cleanup stress test;
- add run manifest;
- expand root JSONL schema;
- verify no silent fallback.

### Step 8: build fixed-root benchmark

- capture/replay diverse roots;
- establish high-K references;
- run K/depth convergence;
- produce plots/tables and subgroup metrics;
- select a small number of end-to-end candidates.

### Step 9: implement paired mirrored evaluation

- exact team/seed pairing;
- side swap;
- paired statistics;
- development/held-out manifests;
- smoke, screen, then confirm.

### Step 10: opponent-model matrix

- self model;
- exact live-opponent diagnostic;
- robust opponents;
- explain failures by estimator versus model mismatch.

### Step 11: optimize only after a positive result

- equal-compute frontier;
- pruning;
- adaptive K;
- selective search;
- branch-copy and batching optimization.

### Step 12: consider belief search only after the oracle gate

Do not begin belief implementation merely because the systems work is
interesting.

---

## 37. Required outputs from the next agent

The next agent should not stop at code changes. The expected deliverables are:

### Code

- branch chance reseeding and seed coupling;
- exact leaf policy expectation;
- root critic-only mode;
- corrected and magnetic improvement operators;
- research-safe error behavior;
- fixed-root benchmark support;
- paired mirrored evaluation support;
- expanded structured logging.

### Tests

- full list of new tests;
- exact passing count;
- exact commands;
- runtime;
- any platform-specific skips;
- explanation of what each test proves.

### Reproducibility artifacts

- git SHA and patch/diff;
- checkpoint hash;
- run manifests;
- root corpus manifest;
- seed manifests;
- JSONL schema and sample;
- environment/software versions.

### Experiment report

At minimum include:

1. implementation summary;
2. verified RNG semantics;
3. policy-state validation;
4. return/value-scale validation;
5. fixed-root estimator results;
6. K convergence;
7. depth comparison;
8. critic-only comparison;
9. operator comparison;
10. paired end-to-end results;
11. opponent-model matrix;
12. latency breakdown;
13. failure/error counts;
14. limitations;
15. go/no-go recommendation.

### Honest conclusion categories

Use one of these conclusions:

- **Positive:** corrected oracle search shows a held-out paired gain and merits
  optimization/belief work.
- **Estimator-positive, game-negative:** root estimates converge, but the tested
  search does not improve game outcomes; investigate objective/value mismatch.
- **Opponent-model limited:** search helps with the exact live-opponent model but
  not the self model.
- **Estimator-invalid:** K does not stabilize rankings or correctness tests fail;
  stop strength evaluation.
- **Inconclusive:** uncertainty remains too large; state exactly what additional
  sample or test is required.

Do not describe an inconclusive result as “search does not work.”

---

## 38. Acceptance criteria for a completed successor handoff

A successor handoff is complete only when another clean-context agent can answer:

- What exact commit and checkpoint were used?
- Does primary search resample future chance?
- How are common random numbers assigned?
- Is the opponent root action coupled across candidates?
- How is the leaf state value computed?
- How are PopArt and reward scaling handled?
- What unit does beta use?
- Which policy-improvement operator is primary?
- Does K improve root action-ranking stability?
- What is the paired held-out effect and confidence interval?
- How often does search error or fall back?
- What is the latency breakdown?
- Does search remain robust against non-self opponents?
- What condition must be met before belief search begins?

If those answers require reading source code rather than the handoff, the
handoff is incomplete.

---

## 39. Immediate next experiment after implementation

The first meaningful post-fix experiment should not be a full win-rate sweep.
It should be a fixed-root convergence study with:

```text
root candidates: all legal
chance: branch-resampled with common random numbers
root opponent action: coupled across candidates by rollout index
leaf value: exact frozen-policy expectation
value normalization: none
value scale: one frozen global development scale
operator: single-anchor KL
root selection: diagnostic only; evaluate Q rankings first
K: 4, 16, 64, high-K reference
D: 0 and 1
errors: raise
```

Primary questions:

1. Does K reduce paired action-difference uncertainty?
2. Does top-action agreement with the high-K reference improve?
3. Is D=0 better than root critic only?
4. Does D=1 add information or only variance?
5. How much did the legacy inherited-RNG mode inflate apparent confidence?
6. How often would the old 5%-of-max pruning rule remove the reference-best
   action?
7. Are action changes concentrated at roots with high entropy or critic
   disagreement?

Only after those answers are positive should the agent launch the paired
end-to-end screen.

---

## 40. Research interpretation guide

### If root critic only beats the actor

The actor may be underusing information already present in the frozen critic.
A cheap critic-reranking layer could capture much of the benefit without
simulator rollouts. Compare robustness carefully because greedy critic policies
can be exploitable.

### If D=0 beats root critic only

The exact one-step Showdown transition is correcting critic action ranking.
This validates the core search idea and justifies testing deeper rollout.

### If D=1 beats D=0

Policy-guided continuation provides useful information beyond immediate
settlement. Optimize deeper branch throughput and opponent modeling.

### If D=1 is worse but high-K D=0 is stable

The rollout opponent, branch policy state, discounting, or compounding variance
may be the issue. Do not infer that all search depth is useless.

### If inherited RNG looks strong but resampled chance does not

The prototype was exploiting future chance. Retain the result only as a
diagnostic; do not use it as evidence for legal play strength.

### If exact live-opponent modeling helps but self modeling does not

Opponent-model mismatch is the bottleneck. A league mixture or online opponent
inference may be justified after the oracle estimator is established.

### If Q estimates converge but win rate does not improve

Possible explanations include:

- the shaped-reward/critic objective is not aligned with game win probability;
- local policy improvement increases exploitability;
- oracle action gains are too small relative to policy noise;
- root sampling rather than greedy choice dilutes the gain;
- search is used on too many low-impact turns;
- the critic is calibrated locally but not strategically;
- rollout model bias cancels estimator precision.

Use root reports and opponent robustness, not another blind sweep.

### If Q estimates do not converge

Stop. Investigate:

- branch chance independence;
- policy-state correctness;
- leaf-value variance;
- reward accounting;
- recurrent-state boundary bugs;
- branch settlement;
- insufficient K;
- extreme stochasticity;
- compiler or batching inconsistencies.

---

## 41. Original experiment status to preserve

Retain the original runs in the repository documentation as historical evidence:

| config | battles | WR | notes |
|---|---:|---:|---|
| baseline, no search | 200 | 0.50 | self-play reference; SE about 0.035 |
| K=4, D=0, beta=1, prune=0.05, every 5 | 100 | 0.47 | 1,026 roots; 23% changed argmax; mean KL 0.18 |
| same prototype | 40 | 0.60 | too few games; not evidence |

Also retain:

- mean legal actions around 6.9;
- mean retained actions around 2.2 under old pruning;
- approximately nine rollouts per searched root rather than 28;
- latency around 195 ms per searched root after CUDA warmup.

Label these results **legacy prototype** after the corrected RNG/value changes,
because they are not directly comparable to the new estimator.

---

## 42. Final principles

1. **Preserve the validated simulator fork.** It is the foundation.
2. **Separate hidden-state oracle from future-chance oracle.** Only the former is
   intentional.
3. **Prove convergence before win rate.** K must improve estimator quality.
4. **Remove avoidable variance.** Exact leaf policy expectation is the primary
   bootstrap.
5. **Use one global value scale.** Never normalize away evidence magnitude at
   each root.
6. **Anchor the improved policy.** Unanchored greedy updates are diagnostics,
   not the default.
7. **Evaluate all legal actions before pruning.** Search must first prove it can
   recover low-prior tactics.
8. **Pair and mirror evaluations.** One hundred unpaired games are not a model
   selection protocol.
9. **Fail loudly in research.** Silent baseline fallback invalidates analysis.
10. **Diagnose opponent-model mismatch explicitly.** Do not bury it in aggregate
    win rate.
11. **Optimize only after a positive corrected result.** Throughput work cannot
    rescue an invalid estimator.
12. **Delay belief complexity.** Oracle search must earn the right to become a
    hidden-information search project.

The target is not merely a search system that runs. It is a test-time decision
procedure whose simulator semantics, randomization, recurrent state, value
scale, policy update, statistical evaluation, and computational tradeoffs are
all strong enough to support a credible claim of improved Gen1 OU play.
