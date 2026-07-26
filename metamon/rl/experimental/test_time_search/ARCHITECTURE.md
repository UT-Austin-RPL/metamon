# Test-Time Search for a Frozen Gen1 OU Metamon Policy — Architecture Note

## Research goal

Determine whether shallow, batched, policy-guided Monte Carlo rollouts over
**exact** Pokémon Showdown continuations improve a frozen strong Gen1 OU
Metamon policy (`MiniOnlinePsroV1_4`) at inference time, as test-time rollout
compute increases.

This is an **oracle** experiment: rollouts branch from the simulator's true
hidden state. No belief model, no MCTS, no training, no policy/critic weight
changes.

## Files inspected (actual control flow)

### Vectorized Showdown host + Python transport
- `metamon/env/vectorized/battle_host.js` — one Node process hosting N
  `BattleStream`s. JSON-lines commands in (`start`, `choose`,
  `choose_batch`, `reset`, `ping`, `close`); binary frames out
  (`READY/CHUNK/HOST_ERROR/LANE_ERROR/PONG`). Each `Lane` owns one
  `BattleStream` + `getPlayerStreams`; player channels are pumped to Python as
  chunks tagged with `(lane, epoch, stream)`.
- `metamon/env/vectorized/sim_process.py` — `ShowdownSimProcess` /
  `ShardedShowdownSimProcess`. Spawns the host, background thread reads stdout
  into an inbox, `pump_until(predicate)` dispatches chunks to per-lane
  `LaneHandler`s. Per-lane epochs drop stale chunks from destroyed battles.
- `metamon/env/vectorized/lane.py` — `StreamBattleLane`: two-POV (p1/p2) parsed
  battle state via two `MetamonBackendBattle`. `handle_chunk` parses `|request|`
  + protocol lines. Decision cycles synchronized on per-side request serials.
  `request_kind` ∈ {move, forceswitch, teampreview, wait, done};
  `needs_agent_decision` / `decision_ready` / `mark_settled` /
  `universal_state` / `legal_action_indices`.
- `metamon/env/vectorized/vector_env.py` — `VectorizedShowdownEnv`: a `step` is
  **request-driven**. It consumes exactly one eval-side agent decision per lane
  (a normal move OR a KO-induced force-switch), answers the opponent
  simultaneously when it owes a decision in the same cycle, then
  `_pump_settle` + `_advance_lanes` auto-resolves opponent-only follow-ups and
  re-prompts until every live lane parks at the next eval-side decision (or
  ends). `BattleAgainstMetamon` / `BattleAgainstOpponentPool` factories.

### Frozen policy inference path
- `metamon/env/vectorized/amago_policy.py` — `AmagoLadderPolicyDriver`: batched
  forward mirroring AMAGO `interact`. Tracks per-lane `rl2s`
  (`concat(reward, prev_action_one_hot)`, dim `action_dim+1`) and `step_counts`
  (the AMAGO `time_idx`). `act(active, obs_list)` snapshots/restores KV cache
  for **inactive** lanes so only active lanes advance recurrent state.
  `observe(lane, reward, action)` updates rl2/time_idx after a committed
  decision. `_snapshot_hidden`/`_restore_hidden` clone
  `hidden_state.{seq_lens,key_cache.data,val_cache.data}` for inactive indices.
- `metamon/env/vectorized/opponent.py` — `BatchedOpponent` ABC;
  `AmagoBatchedOpponent` wraps a driver; `ConfigBatchedOpponent` samples from a
  pool on env `reset()`.

### AMAGO policy / critic outputs (the `MultiTaskAgent`)
- `policy.get_actions(obs, rl2s, time_idxs, hidden_state, sample)` →
  `(actions, hidden_state)`. Actions are for the **primary gamma**
  (`self.gammas[-1]`, index -1).
- `policy.get_state_embedding(obs, rl2s, time_idxs, hidden_state)` →
  `(traj_emb_t [B,1,state_dim], hidden_state)` — reusable to get both actor
  distribution and critic values from one embedding.
- `policy.actor(traj_emb_t, straight_from_obs={"illegal_actions": mask})` →
  `amago.nets.policy_dists._Categorical` (logits; illegal actions masked to
  `-inf` by `MetamonMaskedResidualActor`). `.probs` / `.logits` /
  `.sample()` / `.entropy()`.
- `policy.critics(state, action_onehot.unsqueeze(0))` → for `NCriticsTwoHot`: a
  `pyd.Categorical` over 64 bins, shape
  `(K=1, B, L=1, num_critics=4, num_gammas=7, output_bins=64)`.
- `critic.bin_dist_to_raw_vals(bin_dist)` → expected scalar
  `(bin_dist.probs * bin_vals).sum(-1)` then `invert_bins` (symexp if
  `use_symlog`). **This model: `use_symlog=False`, `min_return=-100`,
  `max_return=2100`** → values are in raw (reward-multiplier-scaled) return
  units.
- `policy.popart(q, normalized=False)` → denormalized value;
  `policy.popart(q, normalized=True)` → PopArt-normalized. **PopArt is enabled.**
- `policy.gammas` = `[0.1, 0.9, 0.95, 0.97, 0.99, 0.995, 0.999]`. Primary
  rollout/eval gamma = `0.999` (index 6 / -1). Search defaults to this horizon.

### Hidden state structure (TformerTrajEncoder)
- `TformerHiddenState(key_cache, val_cache, seq_lens)`.
- `key_cache.data` / `val_cache.data`: shape
  `(n_layers=3, batch, max_seq_len=128, n_heads=8, head_dim=50)`, **bfloat16**.
- `seq_lens`: `int32` shape `(batch,)`.
- `init_hidden_state(B, device)` / `reset_hidden_state(hs, dones)`.

### Reward + return scale
- `AggressiveShapedReward`: `1.0*(damage_done+hp_gain) +
  2.0*(removed_pokemon-lost_pokemon) + 200.0*victory`. Agent
  `reward_multiplier=10.0` scales returns the critic predicts (critic
  `max_return=2100` is in scaled space). Search uses this same reward +
  bootstrap with the denormalized critic value at the same gamma.

## Chosen snapshot/fork implementation

**Official Showdown serialization (`Battle.toJSON()` / `Battle.fromJSON()`)** for
the JS simulator state **+** `copy.deepcopy` of the trunk `StreamBattleLane` for
the Python parsed state.

`State.serializeBattle` / `State.deserializeBattle` (in `sim/state.ts`)
serializes **every** state component: PRNG seed, turn, requestState, the action
`queue`, `log`, `inputLog`, `sentLogPos`, `sentEnd`, `hints` Set, all
`Side`/`Pokemon`/`Field` state (HP, status, volatiles, stat boosts, move PP,
side conditions, field weather/effects, `choice` + `switchIns`, activeRequest),
and reconstitutes object-graph references (Battle/Field/Side/Pokemon/Ability/
Item/Move/Species). This is the same mechanism Showdown uses for reconnects.

### Critical correctness details discovered during validation
1. **`state.log = battle.log` is a *reference*, not a copy.** A forked battle
   deserialized from an *object* snapshot shares the live battle's `log` array,
   so appending to a fork's log corrupts the trunk. **Fix: snapshots are stored
   and passed as JSON strings** (`JSON.stringify(toJSON())` →
   `Battle.fromJSON(string)`), so `JSON.parse` creates fresh arrays. The host
   stores snapshots as strings.
2. **Request regeneration quirk.** `deserializeBattle` regenerates each side's
   `activeRequest` via `getRequests` (the cached request is not serialized).
   The regenerated request can differ from the live-emitted one in move-PP
   (Showdown emits a turn's request *before* decrementing PP in some orderings)
   and in the move list it exposes. Replaying the full log to a *fresh* Python
   lane therefore produces an observation that differs from the trunk's
   incrementally-built one.
3. **Chosen fix for (2): deepcopy the Python lane + skip log replay.** The fork's
   Python `StreamBattleLane` is `copy.deepcopy(trunk)` (validated byte-exact,
   including revealed moves and request-PP), and the JS fork is created with
   `replay_log=False` so it keeps `sentLogPos = log.length` and emits **only new**
   log entries (no replay, no request re-emit). This makes the fork's
   observation bit-exact with the trunk's at the snapshot point.
4. **Sync race.** `fork` and a subsequent `choose` are separate stdin lines;
   the host could process `choose` first (creating an empty lane that drops the
   choice). `ShowdownSimProcess` round-trips a `ping`/`pong` after every
   `snapshot`/`fork`/`restore` (`_sync`) so the fork battle is live before any
   `choose` is sent. A `drain()` helper flushes trailing chunks so the Python
   `deepcopy` and JS `snapshot` capture the same settled point.

### Validated properties (Node POC, `/tmp` scripts, since cleaned up)
- Fork + same future actions → **canonical game-state identical** (sorted-key
  deep equality, `|t:|` timestamps normalized, `sentLogPos`/`sentEnd` excluded
  as transport counters). PRNG seeds identical.
- Fork + divergent actions → states diverge; PRNG diverges.
- **Trunk battle unaffected** by any number of phantom forks (after the
  deep-copy fix).

## Policy-state branching

A branch = simulator snapshot **+** a fork of the policy's recurrent state.
`AmagoLadderPolicyDriver._snapshot_hidden` already clones
`seq_lens`/`key_cache.data`/`val_cache.data` for a set of lane indices; we
generalize it into a reusable `fork_hidden(hidden_state, lane_idx)` that returns
an independent `TformerHiddenState` for one trunk lane, and a
`scatter_into(hidden_state, lane_idxs, src_state)` that copies one trunk lane's
cached state into many fork lanes. Branches also carry independent `rl2s` rows
and `step_counts` (cheap numpy copies).

## Deviations from the proposed design
- **Snapshot storage is host-side (Node), keyed by an integer id**, and forks
  are created host-side from the stored string. Python only exchanges ids
  (plus the fork's starting lane ids). This avoids shipping large JSON across
  IPC for every fork and keeps the Python layer decoupled from the JS object
  representation, as required.
- A fork replays its full log to a **fresh `StreamBattleLane`** rather than the
  Python layer deep-copying `MetamonBackendBattle` (which the task warns
  against without proof of correctness). Log-replay is the normal lane
  ingestion path, so it is well-tested.
- "Depth" = number of **settled evaluated-player decisions** in the rollout
  (our action + opponent simultaneous action + RNG + faint cascade + forced
  switches/re-prompts → next eval-side request or terminal). This matches the
  env's `step` semantics.
- Search is an **eval-only wrapper** (`SearchPolicyDriver`) around the frozen
  policy; `AmagoLadderPolicyDriver` and the env are untouched when search is
  disabled (`search_mode=none` reproduces baseline exactly).

## Important limitations (initial version)
- Oracle: rollouts branch from the simulator's **true** hidden state. Both
  rollout policies still receive only their own POV observation (no peeking),
  but the *branching point* uses ground truth, which a deployed agent would
  not have. This is intentional for the first experiment.
- `lastMoveLine` (a log-line index used for move-log annotation) and
  `sentLogPos`/`sentEnd` are transport/annotation counters, excluded from
  equivalence checks; they do not affect future simulation outcomes (verified:
  identical HP/status/PRNG/requests/log content across forks).
- Snapshot/fork of a lane that is mid-`|error|` re-prompt is not supported in
  v1; search branches only from **settled** eval-side decision points (the env
  already guarantees this is where `step` is called).
