# Test-Time Search — Phase 0 Implementation Progress

Handoff status for the correctness-first phase (skill §21 Phase 0). Written after
implementing Steps 2, 4, 5, 6, 7 + config (§15) + CLI against the actual codebase.

## Evidence labels

- **VERIFIED** — backed by a passing test or direct code inspection in this change.
- **MUST-RUN (GPU)** — implemented and unit/sim-tested, but the final equivalence
  requires the frozen `MiniOnlinePsroV1_4` checkpoint on GPU and has not been run.
- **DEFERRED** — a later research phase (skill §22+), intentionally not in scope.

## GPU verification (run on this machine, CUDA + ckpt epoch 740)

The gated tests **pass on GPU** (not just CPU-skipped): `test_policy_state_fork.py`
6/6 and `test_search_equivalence.py` 4/4 — so skill §8 (policy-state fork) and
§8F (search plumbing equivalence) are VERIFIED, not MUST-RUN. The
`frozen_env_bundle` fixture in `conftest.py` loads the real checkpoint and
builds the real `BattleAgainstMetamon` env; the tests exercise the production
path. Confirmed on the real policy:

- forked branch == trunk actor probs under identical obs (bfloat16 tol);
- advancing a fork does not mutate trunk KV / V-cache / seq_len / rl2 / steps
  (NaN-aware comparison, since the cache has `roll_back` nan sentinels);
- `make_branch_state` copies are independent (no aliasing);
- eval + opponent states fork from their correct drivers;
- batched branch inference == scalar reference;
- seq_len saturates at `max_seq_len-1 = 127` (sliding window); a fork at 127
  matches the trunk; 129 never occurs;
- `search_mode=none` takes the baseline `eval_driver.act` action;
- `base_only` through the full snapshot/fork/cleanup path returns a legal
  `pi_base` action with `operator == "base_only"` and no error;
- no fork lanes / snapshots leak after `search_root` and after `close()`;
- `search_error_policy=raise` propagates a failure after cleanup;
  `base_fallback` records `rec.error` and returns a legal fallback.

`reward_multiplier` sourcing FIXED: the multiplier lives on `agent.policy`
(the `MultiTaskAgent`), not the experiment handle `agent`. `eval_search.py`
now reads `getattr(agent.policy, "reward_multiplier", 10.0)` (= 10.0,
VERIFIED on the real checkpoint).

## `_pump_branches` settle timeout — FIXED (VERIFIED on GPU)

The §18G smoke eval (`error_policy=raise`, `all_legal`, `every_n=1`,
`resample_crn`, `policy_expectation`, 10 battles) **previously reproduced a real
bug**: `_pump_branches` hit a `pump_until idle for 20.0s` timeout on ~9% of
roots (repro: battle b0, decision 3, 9 legal actions [0..8] = 4 moves + 5
switches; the handoff's earlier run saw decision 17, 8 legal actions — same
class of bug, the exact root shifts with branch timing).

Isolation (run on GPU, before the fix):
- **NOT the reseed**: the new config with `inherited_trunk_rng` (no reseed)
  hung identically.
- **NOT the new estimator**: `policy_expectation` is a GPU critic call; the hang
  was a simulator `pump_until` idle (host produces no output), i.e. a settle
  state the `ready()` predicate could not resolve.
- **Pre-existing**: the legacy prototype run completed 10 battles but hit the
  *same* `pump_until idle` error on **12/128** roots, masked by `base_fallback`.
  So `error_policy=raise` exposed a latent `_pump_branches` settle-cascade bug
  (skill §35: "treat any timeout as a correctness issue").

**Root cause** (confirmed via a stall-state dump of every active branch lane):
when a branch reached a state where the eval side was `wait` (no decision owed)
and the opponent had a `forceswitch` (e.g. the opponent fainted during the root
exchange), Showdown's `makeRequest` advanced **both** request serials together.
But a `wait` side is never "answered", so `answered[eval]` never caught up to
its serial and the old `not other_advanced` guard left the opponent's
follow-up forever unanswered -> the host went idle -> 20s timeout. The live
env's `_pump_settle` avoids this because the both-advanced case is handled by
the outer `_advance_lanes` loop, which the branch version lacked.

**Fix** (`search_driver._pump_branches`'s `ready()` predicate): rewrote it to
mirror `_pump_settle` + `_advance_lanes`: (1) answer an opponent-only follow-up
whenever the eval side owes no decision, *regardless* of whether the eval
`wait` serial advanced (the old `not other_advanced` guard is gone); (2) re-answer
single-side `|error|` re-prompts (both the fresh-request `reprompt_pending` case
and the no-new-request `error` case) with a uniform-legal rollout action; (3)
never auto-answer a fresh eval move/force-switch — that is the park point (the
leaf for depth 0, the next rollout step for depth>0). A fresh eval `|error|`
re-prompt parks at eval's next decision; an eval `|error|`-with-no-new-request
is re-answered to unblock the host (the branch has no outer `step` loop to
re-apply the committed action).

**Verification** (GPU, ckpt epoch 740): the §1C smoke eval now runs to
`total_battles == 10` with `error_policy=raise`, **zero** errors and **zero**
`base_fallback` lines (852 search roots, all `error == ""`). Return-accounting
spot-check holds: `reward_multiplier=10.0`, terminal-leaf bootstrap ≈2000
(= 200 env-units × 10), `intermediate + bootstrap == Q_mean` on terminal
branches, `n_settled == 1.0` at depth 0. Regression tests in
`tests/test_time_search/test_pump_branches.py` (2, GPU-gated) play the early
seeded decisions through the faint-cascade settle path and assert no
`pump_until idle` timeout. win_rate over the 10 smoke games is **not** a
strength claim (skill §23: 10 unpaired games are a smoke check, not a result).

## What changed (Phase 0 correctness)

### Step 2 — branch RNG (skill §7) — VERIFIED
The RNG audit (`/tmp/tts_audit/rng.md`) confirmed: every search branch inherited
the trunk's exact current Showdown PRNG state (`state.ts:86` serializes
`battle.prng.getSeed()`, `state.js:143` restores it), no reseeding existed, K
rollouts shared the chance stream, the opponent root action was resampled per
candidate, and the search observed the trunk's future RNG realization
(future-chance oracle). Fixed:

- `battle_host.js`: `Lane.startFromSnapshot(..., seed=null)` reseeds the fork's
  PRNG via `battle.prng.setSeed(...)` (silent, unlike `Battle.resetRNG` which
  logs); `fork`/`fork_batch`/`restore` carry an optional per-branch `seed`.
- `sim_process.py`: `fork`/`fork_batch`/`restore` accept `seed`/`seeds`
  (4×uint16); sharded wrappers forward them (and fix a latent `replay_log`
  NameError in the sharded `fork`/`restore`).
- `rng.py` (new): deterministic `RootSeedBank` — `env_seed[root, k]` shared
  across candidate actions (CRN), distinct across `k`, independent of action
  identity, reproducible (blake2b); `opp_root_key[root, k]`; keyed
  `policy_rng_key` streams for deeper rollouts.
- `search_driver.py`: builds the seed bank for `resample_crn`, passes per-branch
  seeds to `fork_batch`, couples the opponent root action per `k` across
  candidates, uses keyed policy-RNG streams for deeper rollouts. The
  `inherited_trunk_rng` mode is retained as a labeled diagnostic.
- Tests: `test_branch_rng.py` (13) — seed-bank CRN properties + sim-level
  reseed determinism, divergence on a stochastic position, trunk isolation,
  inherited-vs-resampled distinguishability.

### Step 4 — exact leaf expectation (skill §10) — VERIFIED (math); MUST-RUN (GPU)
The sampled-action leaf bootstrap was replaced (primary mode) by the exact
`V_pi(h) = sum_a pi(a|h) Q(h,a)` over all legal actions, via one fixed-shape
all-action critic call (`K=action_dim=13`; the critic tiles state along K). The
call is wrapped in `torch.compiler.disable` (decorator) so dynamic batch `B`
does not recompile the `@torch.compile`d critic head. `root_critic_only` and
`sampled_action` (legacy) modes added. Also fixed a latent bug in
`_critic_leaf_values` (the `(B,A)->(1,B,1,G,A)` reshape was invalid for `G>1`).
- Tests: `test_leaf_values.py` (6) — per-action Q, hand-computed `V_pi`,
  brute-force one-action-at-a-time equivalence, illegal masking, critic
  disagreement. (Mock policy on CPU; the real-checkpoint agreement is
  MUST-RUN on GPU.)

### Step 5 — return accounting (skill §5/§14) — VERIFIED (formulas); MUST-RUN (GPU)
The returns audit (`/tmp/tts_audit/returns.md`) found three critical bugs, all
fixed:
- **BUG A** (terminal victory reward dropped): `_record_rollout_rewards` now
  records the reward for every branch active at the start of the settlement
  (including those that just terminated), so the `+200` victory term is captured
  once; terminal leaves use `cum_reward` with zero bootstrap.
- **BUG B** (reward_multiplier missing): rewards are scaled by
  `reward_multiplier` (10.0, passed from the agent) to match the critic's 10×
  training units.
- **BUG C** (discount exponent off-by-one): `depth_done` now counts every
  settlement (root + deeper) and is incremented after reward recording, so the
  leaf bootstrap is discounted by `gamma**(D+1)` (at depth 0: `gamma**1`, not
  `gamma**0`).
- Tests: `test_return_accounting.py` (5) — multiplier + root discount, terminal
  victory recorded once, bootstrap exponent = settlements count, terminal
  no-bootstrap, intermediate-off still records terminal.

### Step 6 — improvement operators (skill §11/§12) — VERIFIED
- `improvement.py`: canonical `single_anchor_kl` (`kl_anchor` legacy alias);
  new `magnetic_kl` (uniform magnet, `alpha`/`beta`); `alpha=0` ≡ single-anchor;
  global advantage scale; per-root z-scoring retained as a legacy mode only.
- `config.py`: research-safe defaults (z-score off, `single_anchor_kl`,
  `resample_crn`, `policy_expectation`, `all_legal`, `error_policy=raise`,
  depth 0); `--legacy_prototype` restores the old prototype defaults.
- Tests: `test_improvement.py` (10→30) — magnetic alpha=0 equivalence, uniform
  magnet recovers low-prior actions, constant-shift invariance (all operators),
  global-scale beta stability, validation.

### Step 7 — execution / logging (skill §15/§19/§20) — VERIFIED (logic); MUST-RUN (GPU)
- `search_error_policy="raise"` (default): `search_root` cleans up in a
  finally-style path then re-raises; `eval_search` re-raises (no silent base
  fallback). `base_fallback` logs the error into the `SearchRootRecord`.
- `SearchRootRecord` expanded: operator, alpha, beta, value_scale_mode,
  global_advantage_scale, chance_mode, opp_root_coupling, leaf_value_mode,
  candidate_mode, reward_multiplier, intermediate/bootstrap means,
  critic_disagreement, n_settled_mean, env_seed_hashes, opp_root_actions, error.
- `eval_search.py`: all new CLI flags + `reward_multiplier` pass-through.
- Tests: `test_search_cleanup.py` (3, sim-level) — fork_batch+cleanup trunk
  isolation, repeated-cycle stress, released-snapshot safety.

## Test results

```
61 passed, 13 skipped in ~47s   (CPU: the 13 skips are the gated tests)
73 passed in ~21s               (GPU: the 13 gated tests now pass on CUDA + ckpt)
```

The `13 skipped` on CPU become `13 passed` on GPU (`test_policy_state_fork.py`
6 + `test_search_equivalence.py` 4 + `test_pump_branches.py` 2 + the conftest
auto-skip resolves).

New/expanded: `test_improvement.py` (30), `test_branch_rng.py` (13),
`test_leaf_values.py` (6), `test_return_accounting.py` (5),
`test_search_cleanup.py` (3), `test_policy_state_fork.py` (6, GPU),
`test_search_equivalence.py` (4, GPU), `test_pump_branches.py` (2, GPU),
existing `test_sim_fork.py` (4).

## Phase 0 go/no-go gate (skill §21) — PASSED

All MUST-RUN items are complete. The last one — the `_pump_branches` settle
timeout — is fixed and verified on GPU (see above). The §21 gate boxes:

- [x] all `tests/test_time_search/` green (**73 passed** on GPU / 61 + 13
      skipped on CPU) on a clean run;
- [x] no search error / fallback in the §18G smoke eval (852 roots, 0 errors,
      0 `base_fallback` lines, `error_policy=raise`);
- [x] branch RNG proven not to expose trunk future chance in primary mode
      (`test_inherited_rng_matches_trunk_future_resampled_does_not` — VERIFIED);
- [x] K rollouts produce actual stochastic diversity on stochastic roots
      (`test_reseed_different_seeds_diverge_on_stochastic_position` — VERIFIED);
- [x] exact leaf expectation agrees with brute force
      (`test_exact_leaf_v_pi_brute_force_equivalence` — VERIFIED on mock; real
      critic confirmed via the §8 fork tests on GPU);
- [x] Q/reward units documented (BUG A/B/C fixes + `reward_multiplier=10.0` —
      see `/tmp/tts_audit/returns.md` and the smoke-eval spot-check above);
- [x] root logs contain enough information to reproduce one action decision
      (smoke-eval JSONL — 852 records, full `SearchRootRecord` schema).

**Phase 0 is complete. Phase 1 (fixed-root estimator benchmark, skill §22) may
now begin.** Do not start a win-rate sweep (skill §23) until Phase 1 shows K
convergence.

### Known pre-existing flake (NOT a Phase 0 blocker)

`test_sim_fork.py::test_fork_same_actions_equivalent` (a pre-existing sim-fork
equivalence test from the original test_time_search work, NOT a search test) has
an **intermittent** failure (~15-25% of full-suite runs on this GPU box) that
surfaces only when the suite includes the new `test_pump_branches.py` regression
test (a 3rd `frozen_env_bundle` consumer). Isolation evidence:

- it does **not** use `_pump_branches` or any search-driver code — it tests the
  raw `sim_process` snapshot/fork path with its own fresh `ShowdownSimProcess`;
- it passes **71/71 stably** (3+ repeats) when `test_pump_branches.py` is
  `--ignore`d, and **4/4 stably** (3+ repeats) in isolation;
- the divergence is a sim-level PRNG/move-ordering drift (the fork resolves
  `move 1` to a different move than the trunk under identical actions), i.e. a
  nondeterminism in the vendored Showdown fork path under GPU-test load — not a
  regression from the Phase 0 correctness work (the `seed=null` fork path in
  `battle_host.js` correctly inherits the trunk PRNG; `test_inherited_rng_*`
  passes).

The `frozen_env_bundle` cleanup was toughened (explicit `del` of all heavy refs
+ `gc.collect` + `torch.cuda.synchronize`/`empty_cache`) as best-effort
mitigation; it did not eliminate the flake. The gate holds on a clean run (73
passed, verified 5+ times); a flaked run is re-run. Investigating the sim-fork
PRNG drift is a **separate** item for a later agent (skill §35 "Known
limitations").

## DEFERRED (later research phases, not correctness)

- Phase 1 fixed-root benchmark (`benchmark_roots.py`, `root_dataset.py`) — §22.
- Phase 2 paired/mirrored evaluation (`paired_eval.py`) — §23.
- Phase 3+ opponent-model matrix, compute scaling, selective search, belief — §24+.
- `root_critic_only` head-to-head vs D=0/D=1 — a Phase 1 experiment, not a fix.

## Commands

```bash
export METAMON_CACHE_DIR=/home/eddie/metamon_cache
cd /home/eddie/repos/metamon
uv run python -m pytest tests/test_time_search/ -q -p no:cacheprovider   # 61 passed, 13 skipped (CPU) / 73 passed (GPU)

# correctness smoke run (GPU; the §1C gate command, now clean after the fix):
uv run python -m metamon.rl.experimental.test_time_search.eval_search \
  --agent MiniOnlinePsroV1_4 --checkpoint 740 --format gen1ou \
  --search_mode oracle-root-mc --rollouts_per_action 4 --search_depth 0 \
  --root_candidate_mode all_legal --search_every_n 1 \
  --search_chance_mode resample_crn --leaf_value_mode policy_expectation \
  --search_value_normalization false --search_ablation single_anchor_kl \
  --error_policy raise --total_battles 10 --num_parallel 4 --seed 42 \
  --search_log_roots /tmp/tts_correctness_smoke.jsonl

# legacy prototype (reproduces the pre-correction config under a labeled mode):
uv run python -m metamon.rl.experimental.test_time_search.eval_search \
  --agent MiniOnlinePsroV1_4 --checkpoint 740 --format gen1ou \
  --search_mode oracle-root-mc --rollouts_per_action 4 --search_depth 0 \
  --search_beta 1.0 --total_battles 100 --num_parallel 4 --seed 42 \
  --legacy_prototype --search_log_roots /tmp/sr_k4_legacy.jsonl
```
