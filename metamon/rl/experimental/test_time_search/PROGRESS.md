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

## The ONE remaining blocker: `_pump_branches` settle timeout

The §18G smoke eval (`error_policy=raise`, `all_legal`, `every_n=1`,
`resample_crn`, `policy_expectation`, 10 battles) **reproduces a real bug**:
`_pump_branches` hits a `pump_until idle for 20.0s` timeout on some roots
(repro: battle b0, decision 17, 8 legal actions [0..7] = 4 moves + 4 switches).
65/66 roots succeed cleanly before the hang.

Isolation (run on GPU):
- **NOT the reseed**: the new config with `inherited_trunk_rng` (no reseed)
  hangs identically (66 roots, then timeout).
- **NOT the new estimator**: `policy_expectation` is a GPU critic call; the hang
  is a simulator `pump_until` idle (host produces no output), i.e. a settle
  state the `ready()` predicate cannot resolve.
- **Pre-existing**: the legacy prototype run completed 10 battles but hit the
  *same* `pump_until idle` error on **12/128** roots, masked by `base_fallback`.
  So `error_policy=raise` exposes a latent `_pump_branches` settle-cascade bug
  (skill §35: "treat any timeout as a correctness issue"; §14: "rare
  faint/re-prompt cascades can timeout").

This is the single MUST-RUN item left. See `HANDOFF_GPU.md` §1 for the repro
and the likely root cause (a follow-up decision type or both-sides-advanced
state the `ready()` predicate misses).

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
61 passed, 11 skipped in ~47s   (CPU: the 11 skips are the gated tests)
71 passed in ~19s               (GPU: the 11 gated tests now pass on CUDA + ckpt)
```

The `11 skipped` on CPU become `11 passed` on GPU (`test_policy_state_fork.py`
6 + `test_search_equivalence.py` 4 + the conftest auto-skip resolves).

New/expanded: `test_improvement.py` (30), `test_branch_rng.py` (13),
`test_leaf_values.py` (6), `test_return_accounting.py` (5),
`test_search_cleanup.py` (3), `test_policy_state_fork.py` (6, GPU),
`test_search_equivalence.py` (4, GPU), existing `test_sim_fork.py` (4).

## MUST-RUN before the Phase 0 go/no-go gate (skill §21)

Only ONE item remains (the gated tests + reward_multiplier are VERIFIED on GPU):

1. **Fix the `_pump_branches` settle timeout** (reproduced; see `HANDOFF_GPU.md`
   §1). Until `error_policy=raise` completes a smoke eval with zero errors,
   the gate is not passed. The fix is in `_pump_branches`'s `ready()` predicate
   (`search_driver.py`), not in the RNG/estimator/operator code.
2. After the fix: re-run the smoke eval and confirm zero errors/fallbacks, then
   tick the §21 gate boxes.

The GPU-gated tests are NO LONGER MUST-RUN — they pass on GPU
(`test_policy_state_fork.py` + `test_search_equivalence.py`).

## DEFERRED (later research phases, not correctness)

- Phase 1 fixed-root benchmark (`benchmark_roots.py`, `root_dataset.py`) — §22.
- Phase 2 paired/mirrored evaluation (`paired_eval.py`) — §23.
- Phase 3+ opponent-model matrix, compute scaling, selective search, belief — §24+.
- `root_critic_only` head-to-head vs D=0/D=1 — a Phase 1 experiment, not a fix.

## Commands

```bash
export METAMON_CACHE_DIR=/home/eddie/metamon_cache
cd /home/eddie/repos/metamon
uv run python -m pytest tests/test_time_search/ -q -p no:cacheprovider   # 61 passed, 11 skipped

# correctness smoke run (GPU, after enabling the gated tests):
uv run python -m metamon.rl.experimental.test_time_search.eval_search \
  --agent MiniOnlinePsroV1_4 --checkpoint 740 --format gen1ou \
  --search_mode oracle-root-mc --rollouts_per_action 4 --search_depth 0 \
  --root_candidate_mode all_legal --search_every_n 1 \
  --search_chance_mode resample_crn --leaf_value_mode policy_expectation \
  --search_improvement_operator single_anchor_kl --search_error_policy raise \
  --total_battles 10 --num_parallel 4 --seed 42 \
  --search_log_roots /tmp/tts_correctness_smoke.jsonl

# legacy prototype (reproduces the pre-correction config under a labeled mode):
uv run python -m metamon.rl.experimental.test_time_search.eval_search \
  --agent MiniOnlinePsroV1_4 --checkpoint 740 --format gen1ou \
  --search_mode oracle-root-mc --rollouts_per_action 4 --search_depth 0 \
  --search_beta 1.0 --total_battles 100 --num_parallel 4 --seed 42 \
  --legacy_prototype --search_log_roots /tmp/sr_k4_legacy.jsonl
```
