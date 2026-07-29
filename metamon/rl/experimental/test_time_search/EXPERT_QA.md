# Test-Time Search — Expert Q&A (Answers)

This document answers the expert's 13 question sets using the actual code
(`search_driver.py`, `config.py`, `interface.py`, `branch_state.py`, `rng.py`),
the Phase 1 corpus (`/tmp/tts_phase1_v2/root_results.jsonl`, 120 roots, K_ref=128),
and a dedicated behavioral run (`/tmp/tts_behavioral_roots.jsonl`, 40 battles,
812 searched roots). All claims are marked VERIFIED (from code/data) or
NOT-YET-MEASURED (requires a new experiment).

Evidence labels follow the skill: VERIFIED = backed by code/data; NOT-YET-MEASURED
= requires a new experiment; HYPOTHESIS = a research expectation.

---

## 1. Clarify the exact search algorithm

**What precisely does D=0 evaluate for each candidate action?**

VERIFIED (`search_driver._rollout_core` + `_rollout_root`): For each candidate
legal action `a` and rollout index `k`, D=0:

1. Snapshots the trunk Showdown battle lane as a JSON string
   (`Battle.toJSON()` → `Battle.fromJSON()`, `replay_log=false`).
2. Forks `A×K` branch lanes from the snapshot, each reseeded with a
   branch-only 4×uint16 Showdown PRNG seed (`RootSeedBank`, CRN: same seed
   across candidates per `k`, distinct across `k`).
3. Forks the eval and opponent Transformer recurrent state (KV-cache, RL²,
   step counts) into the `A×K` branches.
4. **Forces** the eval candidate action `a` in every branch `a*K + k`.
5. **Samples one opponent root action per `k`** from the frozen opponent
   policy, **coupled across all candidate actions** (same opponent action for
   all `a` at a given `k`).
6. Settles the simultaneous root turn via `_pump_branches` (the full Showdown
   turn: move resolution, speed order, accuracy, damage, crits, secondary
   effects, faints, forced switches, re-prompts).
7. Records the immediate reward (AggressiveShapedReward × 10.0, discounted by
   γ^depth_done).
8. **At D=0, stops** (no further rollout step). Boots the leaf with the exact
   V(s') policy expectation (see below).

**Does it execute the full simultaneous Showdown turn including the sampled
opponent action, stochastic mechanics, switching order, and newly revealed
information?**

VERIFIED: Yes. `_pump_branches` drives the real Showdown `choose` + `pump_until`
path — the branch settles the same simultaneous turn the live env settles,
including all stochastic mechanics. Newly revealed information (e.g. an
opponent move revealed by the root turn) IS reflected in the branch observation
used for the leaf bootstrap — this is correct, since that information would be
publicly revealed by the time the real next decision occurs.

**What value is averaged after the transition: V(s'), immediate reward +
discounted V(s'), or another critic quantity?**

VERIFIED (`_leaf_values`): `Q(a) = cum_reward + γ^(D+1) · V_pi(s')`, where
`V_pi(s') = Σ_a' π(a'|s') · Q_critic(s', a')` (exact policy expectation over
all legal actions, averaged over 4 critic heads, PopArt-denormalized). At D=0,
`cum_reward` is the single root-turn reward. **Empirically (120 roots): the
bootstrap V(s') is ~100% of the Q-value** (intermediate_reward_mean=-1.7 vs
bootstrap_mean=13112). So D=0 is essentially "force the action, settle one
real turn, then read off the frozen critic's value of the resulting state."

**Which critic head and discount factor are used? Is this the same γ=.999 head
used by the deployed actor?**

VERIFIED: `search_critic_horizon=None` → `critic_horizon_index = -1` → the
**primary gamma = 0.999** (the 7th/last of `[0.1, 0.9, 0.95, 0.97, 0.99,
0.995, 0.999]`). This is the same head the deployed actor uses. The critic is
the 4-head ensemble, averaged; 64 two-hot bins; PopArt-denormalized;
`use_symlog=False`; min_return=-100, max_return=2100.

**Does the search value use the exact reward definition on which the
checkpoint's critic was trained? Is it really AggressiveShapedReward with a
200× victory term, and how does it differ from the published Metamon reward?**

VERIFIED (`interface.py:AggressiveShapedReward`): The reward is exactly
`1.0*(damage_done + hp_gain) + 2.0*(removed_pokemon - lost_pokemon) +
200.0*victory`. The search uses `model.reward_function` (=
`AggressiveShapedReward`) and scales by `reward_multiplier=10.0` (matching the
critic's training units). This differs from the *published* Metamon default
`DefaultShapedReward`, which is `1.0*(damage+hp) + 1.0*(removed-lost) +
100.0*victory` plus status shaping. The "Aggressive" variant doubles the
KO-penalty weight (2.0 vs 1.0), doubles the victory term (200 vs 100), and
drops status shaping — explicitly to discourage clinging to lost positions.

**What information makes this search "oracle"? Does it know the opponent's
full team, moves, chosen action, RNG outcomes, or some subset?**

VERIFIED: The forked **Showdown simulator state** contains the *full* hidden
state — both complete teams, all moves, all PP, all HP, the opponent's
unrevealed bench. The branch settles from this full state. So:
- **Opponent's full team & moves & PP: KNOWN** (oracle). The forked battle
  has all 6 opponent Pokémon with full move lists.
- **Opponent's chosen root action: NOT known in advance** — it is *sampled*
  from the frozen opponent policy (coupled per `k`). The search does not see
  the opponent's actual future choice; it integrates over the opponent policy.
- **Future RNG outcomes: NOT known** (VERIFIED, the §7 fix) — branches are
  reseeded with branch-only seeds (CRN), so the search does not see the trunk's
  hidden future PRNG stream. This was the Phase 0 fix.

**Is oracle information used only to initialize simulator states, or can it
leak into the policy or critic observation?**

VERIFIED (important): The **simulator** is oracle (full hidden state), but the
**observation fed to the policy/critic** is NOT oracle. `state_to_obs`
(`DefaultObservationSpace`) exposes only: player active Pokémon + moves +
available switches, **opponent ACTIVE Pokémon only** (not the hidden bench),
`opponents_remaining` count, weather, conditions, previous moves. So the
policy/critic see only public information; the oracle advantage comes solely
from the simulator being able to settle the real transition (including
unrevealed opponent switches that the simulator can legally force). The
opponent rollout policy also sees only its own public observation.

**When comparing actions, are hidden-state samples and opponent root actions
shared across candidates?**

VERIFIED: Yes, both (the §7 design):
- **Opponent root action**: one sample per `k`, reused across all candidate
  actions (coupling).
- **Branch PRNG seed**: one seed per `k`, shared across candidates (CRN) — so
  the initial chance stream is the same for `a1` and `a2` at rollout `k`
  (divergent action paths may consume draws in different orders, so coupling
  is imperfect but variance-reducing).
- **Recurrent policy state**: forked identically from the trunk for all
  branches (one `make_branch_state` call, broadcast).

---

## 2. Establish what the estimator validation proves

**What exactly is the K=128 "reference" estimating?**

VERIFIED: The K=128 reference is the **mean shaped-return estimate** of
`Q(a) = reward + γ·V_pi(s')` at D=0, averaged over 128 branch rollouts per
action (with CRN reseeding). It is an estimate of the frozen critic's
shaped-return objective under the real Showdown one-step transition.

**Why should agreement with the K=128 shaped-return estimator imply better
gameplay? Has K=128 itself been compared against terminal win probability?**

VERIFIED concern / NOT-YET-MEASURED: It should NOT be assumed to imply better
gameplay. K=128 agreement measures **estimator convergence** (low-K approaches
high-K), not objective alignment. **K=128 has NOT been compared against
terminal win probability.** This is the central open question (see Q5). The
shaped reward includes `200*victory` but is dominated by damage/HP/KO shaping;
a converged estimator of the wrong objective is precisely the risk the skill
(§40) flags.

**When the report says the actor is "wrong" 63% of the time, does that mean
only that its top action differs from the D=0 reference?**

VERIFIED: Yes — "wrong" means `base_argmax != D=0_argmax`. It does NOT mean
the D=0 action actually wins more. This is a reference-disagreement rate, not
a ground-truth error rate.

**How large are the estimated value differences when the actor and reference
disagree?**

VERIFIED (120 roots): mean regret = 955, median = 434, p25/p75/p90 =
171/1040/2631 (in critic units, where the global scale ≈ 459 and a KO ≈
~2000-6000 depending on the KO-shaping term × multiplier). So the typical
disagreement is ~1-2 "global scales" — large in shaped-return units.

**What fraction of disagreements are statistically distinguishable from
estimator noise?**

VERIFIED (computed against K=128 SE): 59/76 (78%) of disagreements have
regret > 2×SE at K=128. At K=16 (SE ~2.83× larger), only 43/76 (57%) are
distinguishable at 2σ. **So at K=16, ~43% of the action changes the search
makes are based on advantages that are NOT statistically distinguishable from
estimator noise.** This is a primary suspect for the null game result.

**What is the distribution of action regret rather than only top-1 agreement?**

VERIFIED: see above (median 434, p90 2631). Only 7/76 (9%) of disagreements
are "near-tied" (regret < 20). So most disagreements are large in shaped
units — but "large shaped regret" ≠ "win-probability regret."

**Are convergence and SE calibration computed independently by action, root,
team, and battle? Are multiple roots from the same battle treated as
correlated?**

VERIFIED concern: Convergence/SE-calibration are computed **per root** and
aggregated by mean — NOT independently by action/team/battle. **Roots from the
same battle ARE treated as independent in the aggregate metrics** (the 120
roots come from only 10 battles). This inflates the apparent sample size. The
Phase 1 gate is about the *direction* of convergence (monotone in K), which is
robust to this, but the point estimates (e.g., "0.925 top-1 agreement at K=64")
have more uncertainty than n=120 suggests because of within-battle correlation.

**The distinction: a converged estimator may precisely optimize the wrong
objective.** VERIFIED — this is exactly the situation. The estimator converges
(§22 PASS), but the Phase 2 result (delta ≈ 0) is consistent with "converged
estimator of a shaped objective that is misaligned with win probability."

---

## 3. Understand the root corpus

**Why 111 early / 9 mid / 0 late?**

VERIFIED: The corpus was captured greedily as 4 concurrent battles progressed
with `decision_stride=3`. The 120-root cap was hit before battles reached late
game. Gen1 OU self-play battles on the `competitive` team set are often short
(many end by decision ~30-40), so few roots reach mid (decision ≥40) and none
reach late (≥80). The phase_band uses `typical_battle_len=120` (early<40,
mid 40-80, late>80).

**Were any late-game or endgame roots included?**

VERIFIED: No. 0 late roots.

**How were roots sampled? How many unique battles, teams, seeds?**

VERIFIED: 10 unique battles (`b0_0`..`b3_2`), 1 seed (42), the `competitive`
team set (28 teams, drawn coupled per battle). Roots captured at every 3rd
eval-decision per battle until the 120 cap. Not stratified by entropy/team —
greedy capture.

**Are high-entropy roots overrepresented?**

VERIFIED: No — the opposite. entropy_band: low=93, medium=26, high=1. The
frozen policy is mostly confident (low entropy), so the corpus is
low-entropy-heavy. top2_gap: large=90, medium=18, small=12.

**Does the advantage scale differ between early/mid/late? Does search
disagreement differ by phase?**

VERIFIED (partial — no late data): Early mean_adv (when disagree) = 555; mid
mean_adv = 1223 (2× larger, but n=9). Disagreement rate: early 63%, mid 67%
(similar). **The advantage scale (459) was frozen globally from a corpus that
is 92% early-game** — so beta may be miscalibrated for mid/late where
advantages are larger. This is a real concern for Q9.

**Does search performance differ between tactical and strategic turns? Are
forced switches / sleep / paralysis / Explosion / recovery / imminent-KO
represented?**

NOT-YET-MEASURED: The corpus does not stratify by request type or tactical
category. n_legal ranges 2-9 (mean 7.2), so forced-switch turns (n_legal ≈ 5
switches only) are likely present but not tagged. No analysis of
sleep/paralysis/Explosion/recovery representation has been done. **This is a
gap.**

**Should β and normalization be conditioned on phase or value scale?**

HYPOTHESIS: The 2× advantage-scale difference between early and mid suggests
phase- or scale-conditioned β could help. NOT-YET-MEASURED. The skill (§11)
deliberately chose a single frozen global scale to keep β interpretable;
conditioning it is a permitted later optimization but not yet tested.

---

## 4. Measure whether search actually changes behavior

(VERIFIED from `/tmp/tts_behavioral_roots.jsonl`, 40 battles, 812 roots,
K=16/D=0/beta=5.0/every_n=3/sampling — the exact main config.)

**On what % of searched turns does the updated distribution materially differ?**

VERIFIED: mean KL = 0.031, but **median KL = 0.0013** — the distribution is
heavily right-skewed. p10=0.0001, p90=0.022. Most searched turns barely move
the policy; a few move it a lot (max=4.64). "Materially differs" depends on
threshold, but >90% of roots have KL < 0.02.

**How much probability moves onto the search-best action?**

VERIFIED: improved prob on the base-argmax action = 0.797 mean (i.e. the
improved policy still puts ~80% on the base's top choice). Entropy barely
changes: 0.513 → 0.509.

**How often does sampling from the updated policy produce the same action the
base actor would have sampled?**

VERIFIED (approx): The *actually played* action differs from base_argmax on
**19.5%** of searched roots. (This equals `changed_argmax` because with
beta=5.0 the improved argmax rarely differs from what sampling picks — the
policy is concentrated.)

**How often does the actual selected action differ over a complete game?**

VERIFIED: mean 4.6 action-diffs per game (median 4, max 13) across ~21
searched roots/game. **13% of games have ZERO action differences.**

**Are action changes concentrated in openings/mid/endgame?**

NOT-YET-MEASURED precisely (the behavioral run didn't tag phase per root, but
since every_n=3 and battles are short, most searched roots are early-game).

**Could every_n=3 + KL≈0.02 + sampling make the treatment too weak to detect?**

VERIFIED concern: Yes, this is plausible. every_n=3 means **2/3 of decisions
are never searched**. Of the 1/3 searched, 80% of the probability stays on the
base action, and only ~20% of those change the played action. Net: the played
action differs from baseline on roughly `0.33 × 0.20 ≈ 6.6%` of all decisions.
The argmax result (+0.037 vs sampling's -0.008) supports that sampling dilutes
further. **The treatment may genuinely be too weak.**

---

## 5. Test objective alignment directly

**Can we estimate terminal win probability after forcing each legal action at
held-out roots? How well does root critic Q / D=0 / D=1 correlate with
terminal win probability?**

NOT-YET-MEASURED — **this is the single most important missing experiment.** It
is feasible with the existing infrastructure: `estimate_root` already forks
branches and can force actions; continuing each branch to terminal (with the
frozen policy playing both sides) gives a terminal win indicator per branch.
The plumbing exists; no one has run it. This directly answers "is the shaped
critic aligned with winning?"

**Does D=1 improve that correlation?**

NOT-YET-MEASURED.

**How frequently does the search-best shaped-return action reduce terminal
win probability? Mean terminal-win regret of actor / K=16 / K=128?**

NOT-YET-MEASURED — requires the terminal-win experiment above.

**Is there an outcome-classification or terminal-value head? Could a
win-probability head be trained without retraining the actor?**

VERIFIED: No win-probability head exists. The critic is a 4-head two-hot
shaped-return regressor. A frozen-policy win-probability head could be trained
on existing self-play trajectories (the `battle_won` label is already logged)
without touching the actor — this is the Ataraxos-style categorical
win/loss/draw value the expert references. **This is a concrete permitted
direction** (skill §4 allows "belief-state search only after the oracle system
passes its go/no-go gate" — but training a *value head*, not the actor, is
allowed and is the natural fix if objective misalignment is confirmed).

---

## 6. Validate the counterfactual continuation protocol

**Can the simulator branch from exactly the same partially-observed root state?
Can each candidate be forced while keeping the opponent's root action
identical? Can hidden-team samples and RNG be shared across candidates?**

VERIFIED: Yes to all three. The snapshot/fork (§6) is validated
(`test_sim_fork.py`); opponent root action is coupled per `k`; CRN seeds are
shared across candidates per `k`. This is the core validated engineering.

**Are continuation policies identical after the forced first action? Does the
continuation policy receive only genuinely-revealed information?**

VERIFIED: At D=0 there is no continuation (only the leaf bootstrap, which uses
the public observation). At D≥1, both sides sample from the frozen policy; the
branch observation is built from `universal_state(side)` → `state_to_obs`,
which exposes only public info (opponent active Pokémon, not hidden bench). So
the rollout policy does NOT see oracle hidden info — only the *simulator*
settles from the full state.

**How are impossible/inconsistent oracle determinizations handled? Does
forcing a different action create out-of-distribution observation histories?**

VERIFIED concern / NOT-YET-MEASURED: Forcing a different action does produce a
different observation-history for the Transformer (the branch sees the outcome
of the counterfactual action). The branch recurrent state is forked from the
trunk and advanced with the branch's own observations — so it is *in-distribution
by construction* (the Transformer was trained on trajectories that include all
legal actions). However, **no manual audit of branched trajectories for
information leakage or mechanics correctness has been done beyond the
simulator-fork equivalence tests.** This is a gap the expert flags.

**Has the model's sequence history been verified to update correctly with the
counterfactual action? Have branched trajectories been manually audited?**

VERIFIED (partial): The policy-state fork tests (`test_policy_state_fork.py`,
GPU) verify the forked KV-cache/RL² matches the trunk and advances
independently. **No manual replay audit of full branched games has been done.**

---

## 7. Determine whether K=16 is the bottleneck

**Among roots where K=16 and K=128 disagree, which action has higher terminal
win probability?**

NOT-YET-MEASURED — requires the terminal-win experiment (Q5).

**What fraction of K=16 errors occur when action values are close?**

VERIFIED (partial): At K=128, 43/76 disagreements are 2σ-distinguishable; at
K=16 only 43/76 survive the 2.83× larger SE. So **~33/76 (43%) of the changes
K=16 makes would revert at K=128** — K=16 is making a substantial fraction of
noisy/wrong changes. This directly supports "K=16 is too noisy" as a
contributor to the null result.

**Does K=64 materially improve terminal-win decisions (not just K=128
agreement)?**

NOT-YET-MEASURED for terminal win. VERIFIED for K=128 agreement: K=64 top-1
agreement = 0.925 vs K=16's 0.883 — modest improvement. K=64 at every_n=1 was
infeasible (~30s/root × 85 roots/battle); K=64 at every_n=3 is ~3× slower
than K=16 (~3s/root) and may be feasible.

**Could CRN reduce variance enough to improve K=16? Adaptive K? Early
stopping? Early elimination?**

VERIFIED: CRN is already used (§7). Adaptive K / racing / successive halving
are permitted later optimizations (skill §9/§26) but NOT implemented. The
benchmark infrastructure (`estimate_root` + per-branch matrices) could support
them.

**Computational bottleneck?**

VERIFIED: at K=16, the bottleneck is **simulator settling** (the pump times),
not transformer inference or batching. At K=64, pump times jump to ~12-30s.
Environment reset is not the bottleneck (fork_batch is batched).

---

## 8. Determine whether depth is the bottleneck

**What strategic information does D=1 add beyond D=0?**

VERIFIED (Phase 1): D=1 adds one more policy-guided settled turn for both
sides before the critic bootstrap. Phase 1 showed D=1 is **noisier than D=0 at
every K** (top-1 0.525 vs 0.725 at K=4; 0.875 vs 0.925 at K=64) — it adds
variance (rollout-opponent-model + recurrent-state) faster than it removes
critic bias. **D=1 did NOT help at this scale.**

**Does deeper search improve action ranking vs terminal win probability? Does
depth help only at particular root types?**

NOT-YET-MEASURED for terminal win. The Phase 1 stratification hinted D=1 might
help on specific tactical roots but it was not the default win.

**Are subsequent actions sampled from the frozen actor for both players? Does
the rollout preserve each player's separate information state? Does the
opponent policy receive info revealed by the candidate first action?**

VERIFIED: Yes, yes, yes. `_rollout_step` samples both sides from their frozen
policies; each side has its own forked recurrent state; each side's
observation is built from its own `universal_state(side)` (public info only),
so the opponent does see information revealed by the candidate action (correct).

**Are rollouts evaluating a continuation equilibrium or merely performance
against the frozen opponent policy?**

VERIFIED: The latter — both sides play the frozen policy. This is not an
equilibrium search; it's policy-improvement-against-self-rollout. Against a
different live opponent, the rollout opponent is a *model* of that opponent
(here, the self model).

**Would D=2-4 with fewer rollouts outperform D=0 with many at equal compute?
Is there a horizon where critic/model error dominates?**

NOT-YET-MEASURED. The skill (§14) warns deeper rollout can hurt through
compounding opponent-model error and distribution shift. Phase 1 suggests D=1
already adds variance. Ataraxos used 40-ply — radically deeper — but with a
win/loss value head, not a shaped critic. **Depth with the current shaped
critic is unlikely to help until objective alignment is fixed.**

---

## 9. Revisit the policy update

**Why median KL ≈ 0.02? Was it selected using the eval phase distribution? Was
β tuned against terminal win probability?**

VERIFIED: KL≈0.02 was chosen from the skill's heuristic range (0.01-0.05) and
the global-scale formula `beta ≈ sqrt(1/(2·KL))` → beta=5.0 for KL=0.02. It
was derived from the **Phase 1 dev corpus (92% early-game)**, NOT the eval
distribution, and NOT tuned against terminal win probability. **The actual
observed median KL is 0.0013, not 0.02** — the *mean* is 0.03 but the
distribution is extremely right-skewed. So most roots get an update far
smaller than the target.

**What happens at KL targets 0.05/0.10/0.20? Does a stronger update help when
the advantage is large and well-estimated?**

NOT-YET-MEASURED. Lower beta (stronger update) is the obvious next knob. The
skill (§12) warns too-aggressive updates can increase exploitability, but the
current update is likely too *weak* (median KL 0.0013).

**Should update strength depend on advantage-to-SE ratio? Should search retain
the base policy when no candidate is confidently superior? Would
confidence-gated argmax capture the argmax benefit without changing uncertain
decisions?**

HYPOTHESIS / NOT-YET-MEASURED: A confidence-gated update (only move when
advantage/SE > threshold, else keep base) is very promising — it would combine
the argmax benefit (only change confident decisions) with sampling safety. The
skill (§26) permits this as a later adaptive feature. **This is a concrete
next experiment.** The data to design it exists: 57% of K=16 changes are
<2σ-distinguishable — gating those out could remove the noise that cancels the
signal.

**Does the current update preserve strategic randomization in mixed positions?**

VERIFIED: Sampling from the improved policy is the default; the improved
policy is a softened KL-anchored update, so mixed positions stay mixed (the
entropy drops only 0.004 on average). The skill (§12) explicitly chose sampling
to preserve mixed-strategy behavior in zero-sum games.

---

## 10. Replace periodic search with meaningful triggers

**Why every_n=3?**

VERIFIED: every_n=3 was a **throughput compromise** — K=64 at every_n=1 was
infeasible (~30s/root), and K=16 at every_n=1 was ~3× slower than every_n=3.
It was NOT chosen for a strategic reason. It searches 1/3 of decisions.

**Does the schedule search the same turn numbers after forced switches? How
many important turns are skipped?**

VERIFIED concern: every_n=3 counts *steps*, not *decisions* — after forced
switches/irregular sequences the searched turn numbers shift. **Tactically
critical turns (a faint, a revealed trap, an imminent KO) are frequently
skipped because they don't fall on the periodic schedule.** This is a real
weakness. The skill (§26) explicitly says "Do not use arbitrary every_n as the
final method."

**Would actor entropy / top-1/top-2 margin / leverage be a better trigger?
Cheap K=4 probe + escalate? Equal-compute comparison?**

HYPOTHESIS / NOT-YET-MEASURED: All are permitted (skill §26) and none
implemented. A cheap K=4 probe that escalates only on disagreement is the most
promising. The Phase 1 data shows search matters most at *medium top-2-gap*
roots (agreement 0.43 at K=4 → 1.0 at K=64) — exactly the roots a top-2-margin
trigger would catch. **No equal-compute comparison of trigger strategies has
been done.**

---

## 11. Clarify the paired evaluation

**What exactly constitutes one pair? Are team/RNG/side/opponent randomness
mirrored? Does each pair differ only in search on/off?**

VERIFIED: One pair = (search-ON battle, search-OFF battle) at the same
**(seed, side)**. `run_search_eval(seed=s)` fixes the Showdown battle PRNG,
team draws (coupled per battle via `coupled_player_specs`), and frozen-policy
sampling stream. So search-ON and search-OFF at the same (seed, side) start
from **identical initial conditions** and take identical actions at
non-searched decisions; they differ only in the search-selected action at
searched decisions. Mirroring = running side 0 and side 1 with the same seed
(swaps which team is on which physical side).

**Are all paired outcomes independent across seeds and team selections? Are
same-team/same-seed matches clustered in the CI?**

VERIFIED concern: The bootstrap resamples *pairs* with replacement, treating
pairs as independent. Pairs from different seeds are independent. Pairs from
the same seed but different sides share the initial team draw (mirrored) —
mildly correlated. **The 500 pairs span only 3 seeds × 2 sides**, so the
effective sample size for between-seed variance is small (the per-seed-side
delta ranges -0.08 to +0.11). The CI may be slightly too narrow due to this
clustering; a cluster-robust bootstrap (resampling seeds) would be more honest.

**How are draws/forfeits/simulator errors/timeouts handled?**

VERIFIED: "both-lose" pairs (127/500 = 25.4%) are battles where both search-ON
and search-OFF recorded a loss (win=0). This happens when the *opponent* wins
both battles (natural self-play loss rate ≈ 0.45² ≈ 0.20, close to observed
25%). These are not draws/forfeits — they are genuine opponent-wins. No
simulator errors/timeouts occurred (`error_policy=raise` would have crashed).
The "WARNING" verdict flag on both-lose > 10% is a conservative heuristic; the
25% is expected self-play noise, not a bug.

**Was the argmax configuration selected after viewing the sampling result? Is
there a pre-registered primary configuration?**

VERIFIED: **No pre-registration.** The argmax config was run *after* seeing
sampling's null result, as a diagnostic (the skill §40 hypothesis "sampling
dilutes"). This means the argmax +0.037 is a *post-hoc* observation and should
not be treated as a confirmed result — it is hypothesis-generating. **The next
evaluation must pre-register its primary config.**

**What minimum effect size is practically meaningful? How many pairs for
80%/90% power?**

VERIFIED (power calc, observed discordant fraction 0.456):
- 80% power, delta=0.05: ~1430 pairs
- 80% power, delta=0.08: ~559 pairs
- 90% power, delta=0.05: ~1915 pairs
- 80% power, delta=0.03: ~3972 pairs

The current 500 pairs can only reliably detect a ~8% effect (80% power). A
3-5% effect (the plausible true size) needs 1500-4000 pairs. **The next
evaluation should be powered for delta=0.05 (~1500 pairs) at minimum.**

**Will sequential monitoring inflate false-positive rates? Should the endpoint
be paired WR / Elo / expected score?**

VERIFIED concern: No sequential monitoring was used (fixed n). The primary
endpoint is paired win-rate delta with bootstrap CI + McNemar — appropriate
for a binary zero-sum game. Elo would add little over paired WR for a single
matchup.

---

## 12. Expand the opponent evaluation carefully

**Why expect improvement mainly against weaker opponents? Should a valid
improvement operator first beat the frozen base more reliably?**

VERIFIED concern: The expert is right — a policy-improvement operator should
*first* beat the base policy (self-play), which is the hardest test (the
opponent is identical). Failing self-play and then testing weaker opponents
risks "search helps only by exploiting weaker opponents," which does not
support the superhuman objective. The self-play null is the correct primary
result. Opponent expansion is secondary.

**Which frozen opponents are strategically distinct? Can the matrix include
historical league members/checkpoints? Does search help vs deterministic vs
stochastic? Does oracle search overfit to the frozen opponent continuation?**

NOT-YET-MEASURED: The opponent-model matrix (§24) is not implemented. The
repo has `TaurosV0` (ckpt 62), `Kakuna` (ckpt 34), and earlier
`MiniOnlinePsroV1_4` checkpoints (80, 500) registered. **Oracle search overfit
to the self-rollout opponent is a real risk** — the search optimizes against
the frozen self model, which IS the live opponent in self-play, so overfitting
should be *minimal* in self-play (the model is correct). Against a different
live opponent, self-model rollout would be mismatched. The skill (§13)
provides diagnostic modes (live-opponent oracle model) to separate estimator
vs. model mismatch.

**Should Foul Play / external baselines be included? What justifies ladder
testing?**

NOT-YET-MEASURED / out of scope for this phase. Ladder testing requires the
oracle gate to pass first (skill §27).

---

## 13. Define the next experimental gates

**What result would convince the team the search evaluator has genuine
win-rate signal?**

VERIFIED criterion (proposed): The terminal-win-probability experiment (Q5)
must show that the D=0 K=128 reference action has **higher terminal win
probability than the actor action** on a held-out root set, with a
statistically significant margin. Without this, no game-result experiment is
worth running.

**What result would demonstrate K=16 is adequate? That deeper search is
justified? That a new value head is needed?**

VERIFIED criteria (proposed):
- K=16 adequate: terminal-win regret of K=16 ≈ K=128 (the K=16 errors don't
  cost win probability).
- Deeper search justified: D≥1 terminal-win regret < D=0 regret on tactical
  roots.
- New value head needed: shaped-Q correlates poorly with terminal win
  (Spearman < 0.5 or the search-best shaped action reduces win prob > 25% of
  the time).

**What result would cause the team to stop pursuing this design?**

VERIFIED criterion (proposed): If the terminal-win experiment shows the
K=128 shaped-Q reference does NOT predict terminal win probability better than
the actor (i.e., the shaped critic is misaligned), then **stop tuning this
search design** and pivot to training a win-probability value head (the
Ataraxos approach) before any further search.

**What is the smallest experiment that distinguishes objective misalignment
from estimator noise?**

VERIFIED (proposed): The **terminal-win-probability fixed-root benchmark**:
take ~100 held-out roots, force each legal action, continue each branch to
terminal with the frozen policy playing both sides (D=large or to-terminal),
record the win indicator. Compute (a) Spearman(shaped-Q, terminal-win-rate)
and (b) terminal-win regret of the actor vs K=128-shaped-best vs K=128-terminal-best.
This is ~100 roots × ~7 actions × ~K-to-terminal rollouts — feasible in a few
hours. It directly decomposes objective alignment from estimator noise.

**Can the next experiment decompose reference quality, finite-K error, update
dilution, and scheduling dilution?**

VERIFIED (proposed design): Yes — a single "terminal-win fixed-root" run gives:
- reference quality: K=128-shaped-best vs K=128-terminal-best (objective
  alignment)
- finite-K error: K=16-terminal-best vs K=128-terminal-best
- update dilution: (search-selected action win-rate) vs (K=128-terminal-best
  win-rate) after the KL update
- scheduling dilution: compare every_n=1 vs every_n=3 on the *same* roots

**What artifacts are saved? Compute budget for diagnosis vs confirmation?**

VERIFIED: All Phase 1/2 artifacts are in `/tmp/tts_phase1_v2/`,
`/tmp/tts_phase2_combined/`, `/tmp/tts_phase2_argmax_combined/`,
`/tmp/tts_behavioral_roots.jsonl`. PROGRESS.md records the run history. The
diagnosis experiment (terminal-win benchmark) is a few hours; a confirmatory
paired eval at 1500 pairs is ~10-15 hours.

---

## The most important subset — answers

**Does the K=128 search-selected action improve terminal win probability from
held-out roots?**
NOT-YET-MEASURED. This is the #1 experiment to run.

**How much of that gain survives at K=16?**
NOT-YET-MEASURED (depends on the above; Phase 1 suggests K=16 loses ~43% of
distinguishable changes to noise).

**How much survives the KL update and sampling?**
VERIFIED (behavioral): median KL is 0.0013 (much smaller than the 0.02 target);
only 19.5% of searched roots change the played action; 13% of games have zero
changes. Most of the estimator signal is *not* surviving the update+sampling.
The argmax result (+0.037) vs sampling (-0.008) confirms sampling dilutes.

**How often does search actually change the played action?**
VERIFIED: 19.5% of searched roots, ~4.6 changes/game, 13% of games unchanged.

**Are the reward, critic head, and γ used during search exactly correct?**
VERIFIED: Yes — AggressiveShapedReward × 10.0, γ=0.999 (primary head, same as
actor), 4-critic two-hot ensemble, PopArt-denormalized. Matches training.

**Are candidate actions evaluated with shared hidden states and opponent root
actions?**
VERIFIED: Yes — opponent root action coupled per `k`; CRN branch seeds shared
across candidates; recurrent state forked identically.

**Does search performance differ across early/mid/late game?**
NOT-YET-MEASURED (no late-game roots; mid n=9). The 2× advantage-scale
difference (early 555 vs mid 1223) suggests it might.

**Is every_n=3 skipping the turns where search has value?**
VERIFIED concern: Yes, likely. every_n=3 skips 2/3 of decisions including
tactical critical turns. No entropy/leverage trigger is used.

**What effect size is the next paired evaluation powered to detect?**
VERIFIED: 500 pairs detects ~8% (80% power). For a 5% effect, ~1500 pairs
needed. For 3%, ~4000.

**What explicit gate determines tune / change evaluator / deeper search?**
VERIFIED (proposed): The terminal-win-probability fixed-root benchmark is the
gate. If shaped-Q does not predict terminal win → change the evaluator (train
a win-prob head). If shaped-Q predicts win but K=16 is too noisy → K=64 /
adaptive K. If K is fine but the update is too weak → confidence-gated argmax
/ lower beta. If the update is fine but scheduling skips key turns →
entropy/leverage triggers. **Run the diagnosis before scaling battles.**
