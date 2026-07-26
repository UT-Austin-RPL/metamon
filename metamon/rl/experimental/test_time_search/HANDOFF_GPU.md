# GPU MUST-RUN Handoff — Test-Time Search Phase 0

This runbook is for the agent that has a CUDA GPU and the frozen
`MiniOnlinePsroV1_4` checkpoint (epoch 740). **Most of the GPU MUST-RUN work is
already done and VERIFIED on GPU** (see `PROGRESS.md` "GPU verification"): the
policy-state fork tests (§8) and search-equivalence tests (§8F) pass on GPU,
and `reward_multiplier` sourcing is fixed. **The single remaining blocker is a
`_pump_branches` settle timeout reproduced by the §18G smoke eval** (§1C below).

## 0. Environment

```bash
cd /home/eddie/repos/metamon
export METAMON_CACHE_DIR=/home/eddie/metamon_cache
# checkpoint expected at:
#   ~/metamon_runs/mini_online_psro_v1.4/mini_online_psro_v1.4/ckpts/policy_weights/policy_epoch_740.pt
# (sha256 d3ee307ac597103c69f598057760a3fb29aff12824962c1451ec6fa8f7c5b2c6 as of this handoff)
```

Confirm CUDA + checkpoint, then run the gated tests (they auto-skip without
GPU; on GPU they pass — 10/10 as of this handoff):

```bash
uv run python -m pytest tests/test_time_search/ -q -p no:cacheprovider
# CPU: 61 passed, 11 skipped   |   GPU: 71 passed
```

If a gated test regresses after touching `branch_state.py` / `search_driver.py`,
the `# handoff:` notes in the two gated test files flag the spots to check.

## 1. The remaining MUST-RUN item

### 1C. Fix the `_pump_branches` settle timeout (skill §18G / §35)

**Repro** (correctness config, `error_policy=raise`):

```bash
uv run python -m metamon.rl.experimental.test_time_search.eval_search \
  --agent MiniOnlinePsroV1_4 --checkpoint 740 --format gen1ou \
  --search_mode oracle-root-mc --rollouts_per_action 4 --search_depth 0 \
  --root_candidate_mode all_legal --search_every_n 1 \
  --search_chance_mode resample_crn --leaf_value_mode policy_expectation \
  --search_value_normalization false --search_ablation single_anchor_kl \
  --error_policy raise --total_battles 10 --num_parallel 4 --seed 42 \
  --search_log_roots /tmp/tts_correctness_smoke.jsonl
```

**Failure**: `ShowdownSimProcessError: pump_until idle for 20.0s (host produced
no output)` raised from `search_driver._pump_branches` during `_rollout_root`.
65/66 roots succeed cleanly; the failing root (battle b0, decision 17) has 8
legal actions [0..7] (4 moves + 4 switches). The failing root writes a
`SearchRootRecord` with `error` set, then `error_policy=raise` re-raises.

**Isolation (already run on GPU — do not re-derive):**
- NOT the reseed: `--search_chance_mode inherited_trunk_rng` hangs identically
  (66 roots, then timeout).
- NOT the new estimator: `policy_expectation` is a GPU critic call; the hang is
  a simulator `pump_until` idle (host produces no output) — a settle state the
  `ready()` predicate cannot resolve.
- PRE-EXISTING: the legacy prototype (`--legacy_prototype`, `base_fallback`)
  completes 10 battles but hits the *same* `pump_until idle` error on **12/128**
  roots, masked by `base_fallback`. So `error_policy=raise` exposes a latent
  `_pump_branches` settle-cascade bug, not a regression from the Phase 0 work.

**Likely root cause** (in `search_driver._pump_branches`'s `ready()` predicate,
`search_driver.py` ~`_pump_branches`): the follow-up handler only answers a
side when `advanced and not other_advanced and request_kind(s) in
("move", "forceswitch", "teampreview")`. A settle state it misses — e.g. a
both-sides-advanced simultaneous re-prompt, a `wait`/`switchdrag`/`pass` kind,
or a forceswitch that arrives for the eval side while the opp side is also
waiting — leaves the branch neither parked nor answered, so the host goes idle.
Compare the predicate against the env's `_pump_settle`
(`metamon/env/vectorized/vector_env.py`) which is the battle-tested reference
for the same settling semantics; the branch version mirrors it but has less
exposure (skill §35).

**Fix approach**: make `ready()` answer any answerable follow-up the env's
`_pump_settle` would answer, and park only when the eval side truly owes no
decision. Add a regression test in `test_search_cleanup.py` or a new
`test_pump_branches.py` that reconstructs the failing position (8 legal actions
incl. switches, seed 42, b0 decision 17) and asserts `_pump_branches` resolves
within a tight timeout. After the fix, the smoke command above must complete 10
battles with **zero** `[search] ... falling back to base` lines and every JSONL
record `error == ""`.

**Pass criterion for the gate**: the smoke eval runs to `total_battles == 10`
with `error_policy=raise` and zero errors; spot-check one JSONL record's
`search_q_mean` vs `bootstrap_mean + intermediate_reward_mean` (values in ~10×
env-reward units, victory ≈ 2000, confirming the BUG A/B/C return-accounting
fixes hold on the real critic).

## 2. Already VERIFIED on GPU (no action needed)

- **§8 policy-state fork** — `test_policy_state_fork.py` 6/6.
- **§8F search equivalence** — `test_search_equivalence.py` 4/4 (search_mode=none
  baseline; base_only through-infra; no branch leaks; error-policy raise vs
  base_fallback).
- **reward_multiplier** — `agent.policy.reward_multiplier == 10.0` (VERIFIED);
  `eval_search.py` sources it correctly now.

## 3. After the blocker is fixed — the Phase 0 go/no-go gate (skill §21)

Tick every box before any win-rate sweep:

- [ ] all `tests/test_time_search/` green (71 passed on GPU);
- [ ] no search error / fallback in the smoke eval (§1C);
- [ ] branch RNG proven not to expose trunk future chance in primary mode
      (`test_inherited_rng_matches_trunk_future_resampled_does_not` — VERIFIED);
- [ ] K rollouts produce actual stochastic diversity on stochastic roots
      (`test_reseed_different_seeds_diverge_on_stochastic_position` — VERIFIED);
- [ ] exact leaf expectation agrees with brute force
      (`test_exact_leaf_v_pi_brute_force_equivalence` — VERIFIED on mock; real
      critic confirmed via the §8 fork tests on GPU);
- [ ] Q/reward units documented (BUG A/B/C fixes + `reward_multiplier` — see
      `PROGRESS.md` and `/tmp/tts_audit/returns.md`);
- [ ] root logs contain enough information to reproduce one action decision
      (smoke eval JSONL — confirm in §1C).

## 4. What is explicitly NOT in this handoff (later phases)

- Phase 1 fixed-root benchmark (`benchmark_roots.py`, `root_dataset.py`) — skill §22.
- Phase 2 paired/mirrored evaluation (`paired_eval.py`) — skill §23.
- Opponent-model matrix, compute scaling, selective search, belief search — §24+.

Do not start these until the gate above is passed.

## 5. If a gated test regresses

- **Policy-state fork failure** → the sim-level fork is VERIFIED
  (`test_sim_fork.py`); a policy-state failure points at `branch_state.fork_hidden`
  / the KV-cache broadcast or the `_index_hidden`/`_scatter_hidden` view logic in
  `search_driver.py`. The NaN-aware comparison in
  `test_branch_advance_does_not_mutate_trunk_hidden_state` is deliberate (the
  cache has `roll_back` nan sentinels).
- **Equivalence `base_only` failure** → check `runner._active_fork_lanes` and the
  `SearchRootRecord.error` field; the `frozen_env_bundle` fixture in
  `conftest.py` is the setup path.
- **Smoke eval still hangs after the §1C fix** → the settle state was only
  partially fixed; capture the new failing root from the JSONL (the failing
  root writes its record before re-raising) and compare its `legal_actions` /
  `decision` against the env's `_pump_settle` handling.
