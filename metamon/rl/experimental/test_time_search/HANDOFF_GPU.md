# GPU MUST-RUN Handoff — Test-Time Search Phase 0

This runbook is for the agent that has a CUDA GPU and the frozen
`MiniOnlinePsroV1_4` checkpoint (epoch 740). **All GPU MUST-RUN work is now
done and VERIFIED on GPU** (see `PROGRESS.md` "GPU verification" and
"`_pump_branches` settle timeout — FIXED"): the policy-state fork tests (§8),
search-equivalence tests (§8F), `reward_multiplier` sourcing, and the
`_pump_branches` settle-timeout fix (§1C) all pass on GPU. **Phase 0 is
complete; Phase 1 (skill §22) may begin.** This document is retained as the
runbook/audit trail for the §1C fix.

## 0. Environment

```bash
cd /home/eddie/repos/metamon
export METAMON_CACHE_DIR=/home/eddie/metamon_cache
# checkpoint expected at:
#   ~/metamon_runs/mini_online_psro_v1.4/mini_online_psro_v1.4/ckpts/policy_weights/policy_epoch_740.pt
# (sha256 d3ee307ac597103c69f598057760a3fb29aff12824962c1451ec6fa8f7c5b2c6 as of this handoff)
```

Confirm CUDA + checkpoint, then run the gated tests (they auto-skip without
GPU; on GPU they pass — 13/13 as of this update):

```bash
uv run python -m pytest tests/test_time_search/ -q -p no:cacheprovider
# CPU: 61 passed, 13 skipped   |   GPU: 73 passed
```

If a gated test regresses after touching `branch_state.py` / `search_driver.py`,
the `# handoff:` notes in the gated test files flag the spots to check.

## 1. The (former) MUST-RUN item — RESOLVED

### 1C. `_pump_branches` settle timeout — FIXED (skill §18G / §35)

**Status**: FIXED and VERIFIED on GPU. The §1C repro command below now completes
10 battles with `error_policy=raise`, **zero** errors and **zero**
`base_fallback` lines (852 search roots, all `error == ""`). Regression tests in
`tests/test_time_search/test_pump_branches.py` (2, GPU-gated) guard the fix.

**Root cause** (confirmed via a stall-state dump): when a branch reached a
state where the eval side was `wait` and the opponent had a `forceswitch`
(opponent fainted during the root exchange), Showdown's `makeRequest` advanced
**both** request serials together, but a `wait` side is never "answered", so
`answered[eval]` never caught up and the old `not other_advanced` guard left the
opponent's follow-up forever unanswered -> host idle -> 20s timeout. The live
env's `_pump_settle` avoids this via the outer `_advance_lanes` loop the branch
version lacked.

**Fix** (`search_driver._pump_branches`'s `ready()`): answer opponent-only
follow-ups whenever the eval side owes no decision (regardless of whether the
eval `wait` serial advanced); re-answer single-side `|error|` re-prompts; never
auto-answer a fresh eval move/force-switch (that is the park point). See
`PROGRESS.md` "`_pump_branches` settle timeout — FIXED" for the full write-up.

**Repro** (correctness config, `error_policy=raise` — now passes):

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

**Result (VERIFIED on GPU)**: the smoke eval above now runs to
`total_battles == 10` with `error_policy=raise`, **852 search roots, 0 errors,
0 `base_fallback` lines**. The pre-fix failure was
`ShowdownSimProcessError: pump_until idle for 20.0s (host produced no output)`
raised from `search_driver._pump_branches` during `_rollout_root`; ~9% of roots
hung (e.g. battle b0, decision 3, 9 legal actions [0..8] = 4 moves + 5
switches; the earlier handoff run saw decision 17, 8 legal actions — same class).

**Isolation (run on GPU before the fix — preserved so the next agent doesn't
re-derive it):**
- NOT the reseed: `--search_chance_mode inherited_trunk_rng` hung identically.
- NOT the new estimator: `policy_expectation` is a GPU critic call; the hang was
  a simulator `pump_until` idle (host produces no output) — a settle state the
  `ready()` predicate could not resolve.
- PRE-EXISTING: the legacy prototype (`--legacy_prototype`, `base_fallback`)
  completes 10 battles but hits the *same* `pump_until idle` error on **12/128**
  roots, masked by `base_fallback`. So `error_policy=raise` exposed a latent
  `_pump_branches` settle-cascade bug, not a regression from the Phase 0 work.

**Confirmed root cause** (via a stall-state dump of every active branch lane):
the old `ready()` only answered a side when
`advanced and not other_advanced and request_kind(s) in ("move", ...)`. When a
branch reached eval=`wait` + opp=`forceswitch` (opp fainted during the root
exchange), Showdown's `makeRequest` advanced **both** serials together, but a
`wait` side is never "answered", so `answered[eval]` never caught up and
`not other_advanced` stayed False forever — the opponent's follow-up was never
answered and the host went idle. The env's `_pump_settle`
(`metamon/env/vectorized/vector_env.py`) avoids this because the both-advanced
case is handled by the outer `_advance_lanes` loop the branch version lacked.

**Fix applied** (`search_driver._pump_branches`'s `ready()`): rewrote the
predicate to mirror `_pump_settle` + `_advance_lanes` — (1) answer an
opponent-only follow-up whenever the eval side owes no decision, *regardless*
of whether the eval `wait` serial advanced (the old `not other_advanced` guard
is gone); (2) re-answer single-side `|error|` re-prompts (both the
fresh-request `reprompt_pending` case and the no-new-request `error` case) with
a uniform-legal rollout action; (3) never auto-answer a fresh eval
move/force-switch (that is the park point). Regression tests in
`tests/test_time_search/test_pump_branches.py` (2, GPU-gated) play the early
seeded decisions through the faint-cascade settle path and assert no
`pump_until idle` timeout.

**Pass criterion for the gate (MET)**: the smoke eval runs to
`total_battles == 10` with `error_policy=raise` and zero errors; the JSONL
spot-check holds — `search_q_mean ≈ intermediate_reward_mean + bootstrap_mean`,
values in ~10× env-reward units, terminal victory bootstrap ≈ 2000
(= 200 env-units × `reward_multiplier=10.0`), `n_settled == 1.0` at depth 0
(confirming the BUG A/B/C return-accounting fixes hold on the real critic).

## 2. Already VERIFIED on GPU (no action needed)

- **§8 policy-state fork** — `test_policy_state_fork.py` 6/6.
- **§8F search equivalence** — `test_search_equivalence.py` 4/4 (search_mode=none
  baseline; base_only through-infra; no branch leaks; error-policy raise vs
  base_fallback).
- **reward_multiplier** — `agent.policy.reward_multiplier == 10.0` (VERIFIED);
  `eval_search.py` sources it correctly now.

## 3. Phase 0 go/no-go gate (skill §21) — PASSED

Every box is ticked (verified on GPU, ckpt epoch 740). Phase 1 (skill §22)
may begin; do not start a win-rate sweep (skill §23) until Phase 1 shows K
convergence.

- [x] all `tests/test_time_search/` green (**73 passed** on GPU / 61 + 13
      skipped on CPU);
- [x] no search error / fallback in the smoke eval (§1C) — 852 roots, 0 errors,
      0 `base_fallback` lines;
- [x] branch RNG proven not to expose trunk future chance in primary mode
      (`test_inherited_rng_matches_trunk_future_resampled_does_not` — VERIFIED);
- [x] K rollouts produce actual stochastic diversity on stochastic roots
      (`test_reseed_different_seeds_diverge_on_stochastic_position` — VERIFIED);
- [x] exact leaf expectation agrees with brute force
      (`test_exact_leaf_v_pi_brute_force_equivalence` — VERIFIED on mock; real
      critic confirmed via the §8 fork tests on GPU);
- [x] Q/reward units documented (BUG A/B/C fixes + `reward_multiplier=10.0` —
      see `PROGRESS.md` and `/tmp/tts_audit/returns.md`);
- [x] root logs contain enough information to reproduce one action decision
      (smoke-eval JSONL — 852 records, full `SearchRootRecord` schema).

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
  `decision` against the env's `_pump_settle` handling. Re-enable the stall
  dump by temporarily wrapping the `proc.pump_until(ready, ...)` call in
  `_pump_branches` with a try/except that prints, per active non-ended lane,
  `{side, request_serial, settled_serial, answered, request_kind,
  needs_agent_decision, _side_ready, error, reprompt_pending, decision_ready}`.
  (The §1C fix already covers the confirmed both-advanced `wait`/`forceswitch`
  case and both `|error|` re-prompt variants; a new hang would be a third
  settle state `_pump_settle` handles that the branch predicate still misses.)
