# Test-Time Search Research Plan — Ataraxos-Inspired (arXiv:2511.07312)

Branch: `kimi-search` (from `ec/test-time-search`)
Target agent: **squirtle** (`mini_online_smogon_v0`, ~35M GroupedV2, from-scratch
smogon-only online RL, 7-gamma NCriticsTwoHot critic).

## 0. Where we are (state of `ec/test-time-search`)

The repo already has a complete root-MCTS-style search harness
(`metamon/rl/experimental/test_time_search/`):

* `search_driver.py` — fork-the-simulator root search: force each legal action,
  K rollouts/action (D=0 leaf = critic value; D>0 = policy rollouts), CRN seeds
  per (root, k), adaptive-K with z-score early stopping.
* `improvement.py` — update operators: `single_anchor_kl`
  (pi_search ∝ pi_base · exp(A/β)), `magnetic_kl` (Ataraxos eq. 8:
  pi ∝ exp(q̂)·ρ^α·π_θ^β / (α+β) — already implemented!), `confidence_gated_kl`,
  `argmax_q`, `softmax_q`.
* `paired_eval.py` — paired A/B eval (search vs baseline, same seeds/teams,
  both sides), bootstrap CI + McNemar.
* `terminal_win.py` — Phase A gate: terminal-win ground truth on fixed roots.

**Empirical status (see PROGRESS.md):**
* Phase A: estimator is *sound* (Gate A PASS 4/4): search Q̂ ranking matches
  terminal-win ranking on fixed roots.
* Phase 2 / Phase B+C: **estimator-positive, game-negative.** Search changes
  decisions in the direction the estimator prefers, but win rate does not
  improve (delta −0.0375, CI [−0.14, +0.07]). Diagnosis: the *shaped* critic's
  objective is only moderately aligned with winning (Spearman 0.30), so
  confident Q̂ advantages do not reliably correspond to game wins. Gating +
  adaptive beta made the (partially misaligned) updates *stronger*, not better.

## 1. What Ataraxos does differently (and what we should copy)

Ataraxos (superhuman Stratego, imperfect information, ~$few-k budget):

1. **Search = one tabular step of magnetic mirror descent** (their §2.5, App.
   D.7): q̂ from ~1000 depth-40 rollouts over ~1000/A belief-sampled world
   states, then π_search ∝ exp(q̂)·ρ^α·π_θ^β with α=0.002 (magnet = uniform),
   β=0.02 (anchor to policy net). Our `magnetic_kl` is exactly this operator.
2. **The value target is the game outcome.** Their move network's value head is
   trained on self-play win/loss/draw — not a shaped reward. The search
   objective *is* winning. This is the single biggest gap in our setup: our
   leaf value is the shaped-reward critic (AggressiveShapedReward), which Phase
   A showed is only ~0.3-Spearman-aligned with winning.
3. **Belief network**: sample opponent-hidden world states from a learned
   posterior, rollout each candidate action from each sampled state. In
   Pokémon the hidden state is the opponent's full team/sets; our ladder/cache
   infra (375 squirtle battles) plus the observation already reveals sets over
   time. *Deferred* — the shaped→terminal fix is prerequisite; but the CRN
   seed-bank design already anticipates "resample chance" worlds.
4. **Update can be aggressive at test time** because it is tabular (no
   interference at other states) and based on lower-variance estimates. Our
   adaptive-K + z-gate are budgeted versions of the same idea.
5. **Search is *not* used to generate training data** (they tried; data
   quantity > data quality for them). So: search is a pure test-time lift, and
   we evaluate it as such (paired eval), not as a distillation source (yet).

## 2. Research hypotheses (in priority order)

**H1 (objective alignment is the blocker).** Switching the search leaf value /
advantage signal from the shaped-reward critic to a *terminal-win-aligned*
value turns the estimator-positive/game-negative result positive. Confidence:
high — Phase A measured the misalignment directly; Ataraxos's entire search
works because V predicts game outcome.

**H2 (squirtle's γ→1 head is already terminal-aligned).** The NCriticsTwoHot
critic is trained at 7 gammas with γ_main=0.999 over a horizon where shaped
reward ≈ 0 late-game, so the high-γ head is close to a win-probability head.
Using the max-γ (or a win-calibrated monotone map of it) as the search value
may recover most of H1 *with no new training*. Confidence: medium-high, and
it's cheap to test — do this first.

**H3 (damping the update toward the right fixed point).** With an aligned
value, the magnetic-KL operator with Ataraxos-style coefficients
(α ≈ 0.002 magnet, β ≈ 0.02 anchor — note: *weak* regularization, i.e.
aggressive updates) beats both the conservative gated update and raw argmax.
Confidence: medium. Note our `beta` semantics are inverted vs theirs
(ours divides A by β); needs care in calibration.

**H4 (search-aware fine-tuning / distillation).** Once search reliably beats
the base policy, distilling π_search back into squirtle (or fine-tuning on
search-improved advantages) lifts the *base* policy. Confidence: speculative;
Ataraxos explicitly did *not* need this, but our base is much weaker
(from-scratch 35M), so the base policy has more headroom than the search lift.

**H5 (belief/state uncertainty matters for teams preview).** Modeling
opponent-team uncertainty (samples over unrevealed sets) improves search on
early-game roots. Confidence: low-medium; expensive. Defer until H1–H3 land.

## 3. Milestones

### M0 — Branch + calibrated baseline (½ day)
* [x] Branch `kimi-search` off `ec/test-time-search`.
* [ ] Freeze a squirtle checkpoint as the search base (latest ≈ epoch 975).
* [ ] Re-run the Phase B+C paired-eval config against **squirtle** (not
  MiniOnlinePsroV1_4) at every_n=3 to get the squirtle baseline delta. This is
  the "does shaped-critic search hurt squirtle too" control — expect ≈ 0 or
  slightly negative.

### M1 — Terminal-aligned leaf value, no training (H2; 1–2 days)
* [ ] Add `leaf_value_mode=terminal_gamma` (value = critic's max-γ head) and
  `win_calibrated` (fit an isotonic/logistic map V_γmax → P(win) on the
  existing `terminal_continuations` data + the 375 ladder battles already
  cached by tools/traj_analysis).
* [ ] Phase A gate on squirtle: does terminal-win ranking correlation
  (Spearman) of the search estimator *rise* vs shaped critic? Gate: Spearman
  ≥ 0.5 on the fixed-root benchmark before spending paired-eval compute.
* [ ] If pass → M2 paired eval.

### M2 — Ataraxos-operator paired screen (H1+H3; 2–3 days)
* [ ] Paired eval on squirtle, `magnetic_kl` with α/β calibrated to
  Ataraxos-equivalent strength, terminal-aligned leaf value, adaptive-K
  (pilot 4, max 32, z-stop 2.0), every_n ∈ {3, 1}, n ≥ 320 pairs
  (adaptive-K savings make this affordable).
* [ ] Success gate: paired delta > 0 with 95% CI excluding 0, and no per-side
  collapse (|per-side delta| asymmetry < 0.10 — watch the recurring side-1
  exploitability).
* [ ] Ablations: (a) gating on/off, (b) adaptive beta on/off, (c) α magnet
  on/off. 2–3 arms max per screen; keep CRN seeds fixed across arms.

### M3 — Trained terminal-outcome value head (H1 hard version; 3–5 days, GPU)
* [ ] If M1's γ→1 head is insufficient (Spearman < 0.5 or M2 negative): train a
  small win-probability head on the frozen squirtle encoder using (a) Phase A
  `terminal_continuations` fixed-root data and (b) the online run's replay
  buffer (terminal battle outcomes are already logged).
* [ ] Swap it in as `leaf_value_mode=win_head`; re-run M1 gate + M2 screen.

### M4 — Distillation / search-aware fine-tune (H4; exploratory)
* [ ] Log (state, π_search) pairs during evals; fine-tune squirtle with a KL
  distillation loss toward π_search on logged roots. Measure base-policy lift
  vs the ladder validator.

### M5 (deferred) — Belief/team-set posterior (H5)
* [ ] Sample opponent unrevealed sets from a team-conditioned posterior during
  rollouts. Only if M1–M3 show search works and early-game roots remain the
  weakest (check `analyze_roots.py` by game phase).

## 4. Evaluation protocol (fixed across milestones)

* Agent: `squirtle` (frozen checkpoint per milestone; record epoch).
* Format gen1ou, team_set competitive, opponent = same-checkpoint baseline
  (self-play paired eval), both sides, seeds held out from any tuning.
* Every experiment logs: paired delta + bootstrap CI, McNemar, per-side split,
  % roots gated / argmax-changed, median & mean KL, mean K_eff.
* No arm counts as "benefit" unless delta > 0 with CI excluding 0 at n ≥ 320
  pairs. Smokes (n=20–40) are for crash-testing only.
* All runs: `error_policy=raise`, artifacts under `/tmp/tts_kimi_<milestone>/`,
  PROGRESS.md updated per milestone.

## 5. Risks / open questions

* **Value-scale calibration**: `global_advantage_scale=458.7` was tuned for the
  shaped critic; terminal/win-prob values live in [0,1] or symlog bins — β must
  be re-calibrated (the constant-shift-invariance tests cover the operator, not
  the scale).
* **Side-1 exploitability** recurred in Phase 2 and B+C — if it persists with
  aligned values, it's a real exploitability effect (greedy-ish updates), not
  noise. Mitigation: raise β anchor, lower z_gate, or sample (not argmax) the
  root action (`search_root_selection=sample`).
* **Compute**: D=0 search on a 5090 is fast; every_n=1 at K_max=32 is the
  expensive arm. Adaptive-K measured ~46% savings — budget accordingly.
