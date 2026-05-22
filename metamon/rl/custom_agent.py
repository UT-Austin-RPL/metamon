"""Finetuning agent with a slow-EMA tortoise shadow and optional IS correction.

Implements ``MetamonFinetuneAgent``, a ``MultiTaskAgent`` subclass that
maintains a slow exponential-moving-average (tortoise) copy of the full
online (hare) actor-critic.  Inference can be switched between hare and
tortoise via a flag, and iterative finetuning is supported by initialising
each new round's hare from the previous round's tortoise weights.

Optionally, an ``ISAdvantageFilter`` applies log-space clipped importance-
sampling corrections using a frozen base-policy snapshot and a trainable
behavioral-cloning head that estimates the data distribution.
"""

from __future__ import annotations

import bisect
import copy
import itertools
import math
from collections import deque
from typing import Any, Optional, Tuple

import gin
import torch
import torch.nn.functional as F
from einops import repeat

import amago
from amago.agent import MultiTaskAgent
from amago.loading import Batch
from amago.nets.policy_dists import DiscreteLikeContinuous


@gin.configurable
class ISAdvantageFilter:
    """Batch-normalized exponential weighting with optional IS and sequence-level filtering.

    Per-timestep: ``w(s,a) = exp(beta * A_norm + delta_log_clipped)``

    Optionally multiplied by a per-sequence sigmoid weight based on the
    online percentile rank of each sequence's mean advantage.  Enabled when
    ``seq_p_low`` is not None.  The sequence sigmoid maps:

    * percentile ≈ ``seq_p_low``  →  weight ≈ ``seq_floor``
    * percentile ≈ ``seq_p_full`` →  weight ≈ 1.0

    Steepness is derived: ``k = 2·ln(99) / (seq_p_full − seq_p_low)``.
    The floor ramps from 1.0 (off) to ``seq_floor`` over
    ``seq_floor_warmup_steps`` calls.

    Injected tensors (``delta_log``, ``seq_mask``) are set externally
    before each ``__call__`` and cleared after use.

    Args:
        beta: Temperature for normalized advantages.
        clip_delta: Symmetric log-space clip bound for the IS ratio.
        eps: Numerical stability constant for std normalization.
        clip_weights_low: Floor for final per-timestep weights.
        clip_weights_high: Ceiling for final per-timestep weights.
        seq_p_low: Percentile where sequence weight ≈ floor (None = disabled).
        seq_p_full: Percentile where sequence weight ≈ 1.0.
        seq_floor: Minimum sequence weight.
        seq_floor_warmup_steps: Ramp floor from 1.0 to ``seq_floor`` over
            this many calls.  Match to LR warmup.
        seq_buffer_size: Circular buffer capacity for percentile estimation.
        seq_warmup: Min buffer entries before non-uniform sequence weights.
    """

    def __init__(
        self,
        beta: float = 2.0,
        clip_delta: float = 2.0,
        eps: float = 1e-8,
        clip_weights_low: Optional[float] = 1e-7,
        clip_weights_high: Optional[float] = 100.0,
        seq_p_low: Optional[float] = None,
        seq_p_full: Optional[float] = None,
        seq_floor: float = 0.1,
        seq_floor_warmup_steps: int = 2000,
        seq_buffer_size: int = 10_000,
        seq_warmup: int = 200,
    ):
        self.beta = beta
        self.clip_delta = clip_delta
        self.eps = eps
        self.clip_weights_low = clip_weights_low
        self.clip_weights_high = clip_weights_high
        self._mask: Optional[torch.Tensor] = None
        self._delta_log: Optional[torch.Tensor] = None

        # Sequence-level filter state
        self.seq_enabled = seq_p_low is not None
        self._seq_mask: Optional[torch.Tensor] = None
        self._seq_last_weights: Optional[torch.Tensor] = None
        self._seq_last_percentiles: Optional[torch.Tensor] = None
        self._seq_last_eff_floor: float = 1.0
        if self.seq_enabled:
            assert seq_p_full is not None, "seq_p_full required when seq_p_low is set"
            assert seq_p_full > seq_p_low, "seq_p_full must be greater than seq_p_low"
            self._seq_p_low = seq_p_low
            self._seq_p_full = seq_p_full
            self._seq_floor = seq_floor
            self._seq_warmup_steps = seq_floor_warmup_steps
            self._seq_warmup = seq_warmup
            self._seq_center = (seq_p_low + seq_p_full) / 2.0
            self._seq_k = 2.0 * math.log(99.0) / max(seq_p_full - seq_p_low, 1e-6)
            self._seq_buffer: deque[float] = deque(maxlen=seq_buffer_size)
            self._seq_sorted_cache: Optional[list[float]] = None
            self._seq_cache_dirty = True
            self._seq_step = 0

    def set_mask(self, mask: Optional[torch.Tensor]):
        """Inject boolean mask for BN statistics; cleared after use."""
        self._mask = mask

    def set_delta_log(self, delta_log: Optional[torch.Tensor]):
        """Inject IS correction tensor; cleared after use."""
        self._delta_log = delta_log

    def set_seq_mask(self, mask: Optional[torch.Tensor]):
        """Inject state-validity mask for sequence-level mean; cleared after use."""
        self._seq_mask = mask

    def _seq_sorted_buf(self) -> list[float]:
        if self._seq_cache_dirty:
            self._seq_sorted_cache = sorted(self._seq_buffer)
            self._seq_cache_dirty = False
        return self._seq_sorted_cache  # type: ignore[return-value]

    def _compute_seq_weights(self, adv: torch.Tensor) -> torch.Tensor:
        """Per-sequence sigmoid weights from per-timestep advantages."""
        self._seq_step += 1
        seq_mask = self._seq_mask
        self._seq_mask = None
        B, L, G, _ = adv.shape

        adv_f = adv.detach().float()
        if seq_mask is not None:
            m = seq_mask[:, :L, :].expand(B, L, G).unsqueeze(-1).bool()
            counts = m.float().sum(dim=(1, 2, 3)).clamp(min=1)
            mean_adv = (adv_f * m.float()).sum(dim=(1, 2, 3)) / counts
        else:
            mean_adv = adv_f.mean(dim=(1, 2, 3))

        mean_adv_list = mean_adv.cpu().tolist()
        self._seq_buffer.extend(mean_adv_list)
        self._seq_cache_dirty = True

        if len(self._seq_buffer) < self._seq_warmup:
            self._seq_last_weights = None
            self._seq_last_percentiles = None
            self._seq_last_eff_floor = 1.0
            return torch.ones(B, 1, 1, 1, device=adv.device)

        ramp = min(self._seq_step / max(self._seq_warmup_steps, 1), 1.0)
        eff_floor = 1.0 - (1.0 - self._seq_floor) * ramp
        self._seq_last_eff_floor = eff_floor

        sorted_buf = self._seq_sorted_buf()
        n = len(sorted_buf)
        pcts = [bisect.bisect_left(sorted_buf, v) / n for v in mean_adv_list]
        percentiles = torch.tensor(pcts, device=adv.device, dtype=adv.dtype)

        sigmoid_in = self._seq_k * (percentiles - self._seq_center)
        weights = eff_floor + (1.0 - eff_floor) * torch.sigmoid(sigmoid_in)

        self._seq_last_weights = weights
        self._seq_last_percentiles = percentiles
        return weights.view(B, 1, 1, 1)

    def __call__(self, adv: torch.Tensor) -> torch.Tensor:
        mask = self._mask
        self._mask = None
        delta_log = self._delta_log
        self._delta_log = None

        if mask is not None:
            mask = mask[:, : adv.shape[1], ...]
            while mask.ndim < adv.ndim:
                mask = mask.unsqueeze(-1)
            mask = mask.expand_as(adv)
            valid = adv[mask]
            mu = valid.mean()
            sigma = valid.std() + self.eps
        else:
            mu = adv.mean()
            sigma = adv.std() + self.eps

        adv_norm = (adv - mu) / sigma
        exponent = self.beta * adv_norm

        if delta_log is not None:
            delta_log = delta_log[:, : adv.shape[1], ...]
            exponent = exponent + torch.clamp(
                delta_log, -self.clip_delta, self.clip_delta
            )

        weights = torch.exp(exponent)
        if self.clip_weights_low is not None or self.clip_weights_high is not None:
            weights = torch.clamp(
                weights, min=self.clip_weights_low, max=self.clip_weights_high
            )

        if self.seq_enabled:
            weights = weights * self._compute_seq_weights(adv)

        return weights


_TORTOISE_PREFIX = "_tortoise_"
_BASE_PREFIX = "_base_"

_TORTOISE_MODULES = {
    "_tortoise_tstep_encoder": "tstep_encoder",
    "_tortoise_traj_encoder": "traj_encoder",
    "_tortoise_actor": "actor",
    "_tortoise_critics": "critics",
}
_BASE_MODULES = {
    "_base_tstep_encoder": "tstep_encoder",
    "_base_traj_encoder": "traj_encoder",
    "_base_actor": "actor",
}


@gin.configurable
class MetamonFinetuneAgent(MultiTaskAgent):
    """MultiTaskAgent with a slow-EMA tortoise shadow and optional IS correction.

    On top of the standard hare (online) and target networks, this agent
    maintains:

    * **Tortoise** — slow EMA of the full hare (encoder + actor + critics),
      usable for inference and as the starting point for iterative finetuning.
    * **Base model** — static frozen snapshot of the encoder + actor at
      training start, used for computing ``log pi_base(a|s)``.
    * **BC actor** — trainable actor head that estimates the data distribution
      ``pi_data`` on the frozen base representation.

    When ``fbc_filter_func`` is an :class:`ISAdvantageFilter`, the filter
    receives a per-sample IS correction
    ``delta_log = log pi_base - log pi_data`` before each training step.
    Sequence-level filtering is controlled via ``ISAdvantageFilter.seq_p_low``
    (set in gin); when enabled, the filter also multiplies per-timestep
    weights by a per-sequence sigmoid weight.

    Args:
        bc_coeff: Weight for the auxiliary BC loss on ``_bc_actor``.
        tortoise_tau: EMA rate for the tortoise update (smaller = slower).
        use_tortoise_for_inference: If True, ``get_actions`` runs the
            tortoise encoder + actor instead of the hare.
        use_is_correction: If False, skips the base-model / BC-actor IS
            correction and trains with plain batch-normalized advantages
            (while still maintaining the tortoise EMA shadow).
    """

    def __init__(
        self,
        *args,
        bc_coeff: float = 1.0,
        tortoise_tau: float = 0.001,
        use_tortoise_for_inference: bool = False,
        use_is_correction: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.bc_coeff = bc_coeff
        self.tortoise_tau = tortoise_tau
        self.use_tortoise_for_inference = use_tortoise_for_inference
        self.use_is_correction = use_is_correction
        self._checkpoint_loaded = False
        self._loaded_from_tortoise_agent = False

        # Tortoise (slow EMA of entire hare, frozen during training)
        self._tortoise_tstep_encoder = copy.deepcopy(self.tstep_encoder)
        self._tortoise_traj_encoder = copy.deepcopy(self.traj_encoder)
        self._tortoise_actor = copy.deepcopy(self.actor)
        self._tortoise_critics = copy.deepcopy(self.critics)
        for m in (
            self._tortoise_tstep_encoder,
            self._tortoise_traj_encoder,
            self._tortoise_actor,
            self._tortoise_critics,
        ):
            m.requires_grad_(False)

        # Base model (static frozen snapshot for IS correction)
        self._base_tstep_encoder = copy.deepcopy(self.tstep_encoder)
        self._base_traj_encoder = copy.deepcopy(self.traj_encoder)
        self._base_actor = copy.deepcopy(self.actor)
        for m in (
            self._base_tstep_encoder,
            self._base_traj_encoder,
            self._base_actor,
        ):
            m.requires_grad_(False)

        # BC actor (trainable, estimates pi_data on frozen base representation)
        self._bc_actor = copy.deepcopy(self.actor)

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        has_tortoise = any(k.startswith(_TORTOISE_PREFIX) for k in state_dict)
        self._loaded_from_tortoise_agent = has_tortoise

        if not has_tortoise:
            # Loading a standard (non-MetamonFinetuneAgent) checkpoint.
            # Fill tortoise, base, and bc_actor keys from the hare weights.
            extra = {}
            for tort_attr, hare_attr in _TORTOISE_MODULES.items():
                prefix = hare_attr + "."
                for k, v in state_dict.items():
                    if k.startswith(prefix):
                        extra[tort_attr + k[len(hare_attr) :]] = v.clone()
            for base_attr, hare_attr in _BASE_MODULES.items():
                prefix = hare_attr + "."
                for k, v in state_dict.items():
                    if k.startswith(prefix):
                        extra[base_attr + k[len(hare_attr) :]] = v.clone()
            # BC actor gets actor weights
            actor_prefix = "actor."
            for k, v in state_dict.items():
                if k.startswith(actor_prefix):
                    extra["_bc_actor" + k[len("actor") :]] = v.clone()
            state_dict = {**state_dict, **extra}

        super().load_state_dict(state_dict, strict=strict, **kwargs)

    def on_checkpoint_loaded(self, is_resume: bool = False):
        if is_resume:
            self._checkpoint_loaded = True
            return

        if self._loaded_from_tortoise_agent:
            # Iterative finetuning: initialise hare from the previous
            # round's tortoise (the stable, generalised weights).
            for tort_attr, hare_attr in _TORTOISE_MODULES.items():
                self._full_copy(getattr(self, hare_attr), getattr(self, tort_attr))
            # Base model = snapshot of the new hare (== previous tortoise)
            for base_attr, hare_attr in _BASE_MODULES.items():
                self._full_copy(getattr(self, base_attr), getattr(self, hare_attr))
            # BC actor restarts from the new hare's actor
            self._full_copy(self._bc_actor, self.actor)
            # Target networks must match the new hare
            self.hard_sync_targets()
            # Reset tortoise to the new hare (fresh EMA accumulation)
            for tort_attr, hare_attr in _TORTOISE_MODULES.items():
                self._full_copy(getattr(self, tort_attr), getattr(self, hare_attr))

        # Freeze tortoise and base
        for tort_attr in _TORTOISE_MODULES:
            getattr(self, tort_attr).requires_grad_(False)
        for base_attr in _BASE_MODULES:
            getattr(self, base_attr).requires_grad_(False)

        self._checkpoint_loaded = True

    def soft_sync_targets(self):
        super().soft_sync_targets()
        if self._checkpoint_loaded:
            for tort_attr, hare_attr in _TORTOISE_MODULES.items():
                self._ema_copy(
                    getattr(self, tort_attr),
                    getattr(self, hare_attr),
                    tau=self.tortoise_tau,
                )

    def hard_sync_targets(self):
        super().hard_sync_targets()
        if hasattr(self, "_tortoise_tstep_encoder"):
            for tort_attr, hare_attr in _TORTOISE_MODULES.items():
                self._full_copy(getattr(self, tort_attr), getattr(self, hare_attr))

    @property
    def trainable_params(self):
        return itertools.chain(super().trainable_params, self._bc_actor.parameters())

    def get_grad_norms(self) -> dict[str, float]:
        norms = super().get_grad_norms()
        norms["BC Actor Grad Norm"] = amago.utils.get_grad_norm(self._bc_actor)
        return norms

    def get_actions(
        self,
        obs: dict[str, torch.Tensor],
        rl2s: torch.Tensor,
        time_idxs: torch.Tensor,
        hidden_state: Optional[Any] = None,
        sample: bool = True,
    ) -> Tuple[torch.Tensor, Any]:
        if not self.use_tortoise_for_inference:
            return super().get_actions(
                obs, rl2s, time_idxs, hidden_state=hidden_state, sample=sample
            )

        with torch.no_grad():
            o = self._tortoise_tstep_encoder(obs=obs, rl2s=rl2s)
            s_rep, hidden_state = self._tortoise_traj_encoder(
                o, time_idxs=time_idxs, hidden_state=hidden_state
            )
            action_dists = self._tortoise_actor(
                s_rep,
                straight_from_obs={k: obs[k] for k in self.pass_obs_keys_to_actor},
            )
            if sample:
                actions = action_dists.sample()
            else:
                if self.discrete:
                    actions = torch.argmax(action_dists.probs, dim=-1, keepdim=True)
                else:
                    actions = action_dists.mean
            actions = actions[..., -1, :]
            dtype = (
                torch.uint8 if (self.discrete or self.multibinary) else torch.float32
            )
            return actions.to(dtype=dtype), hidden_state

    def _compute_log_probs(self, actor_head, s_rep, a_buffer, obs=None):
        """Run an actor head and return log pi(a|s) with shape (B, L, G, 1)."""
        straight_from_obs = (
            {k: obs[k] for k in self.pass_obs_keys_to_actor}
            if obs is not None
            else None
        )
        a_dist = actor_head(s_rep, straight_from_obs=straight_from_obs)
        if self.discrete:
            a_dist = DiscreteLikeContinuous(a_dist)
        if self.discrete:
            logp = a_dist.log_prob(a_buffer).unsqueeze(-1)
        elif self.multibinary:
            logp = a_dist.log_prob(a_buffer).mean(-1, keepdim=True)
        else:
            logp = a_dist.log_prob(a_buffer).sum(-1, keepdim=True)
        return logp

    def forward(self, batch: Batch, log_step: bool) -> torch.Tensor:
        if not self._checkpoint_loaded:
            return super().forward(batch, log_step)

        # --- Prepare action buffer (mirrors parent) ---
        a = batch.actions
        a = a.clamp(0, 1.0) if self.discrete else a.clamp(-1.0, 1.0)
        G = len(self.gammas)
        a_buffer = F.pad(a, (0, 0, 0, 1), "replicate")
        a_buffer = repeat(a_buffer, f"b l a -> b l {G} a")

        # --- Frozen base model forward (pi_base) ---
        with torch.no_grad():
            o_base = self._base_tstep_encoder(obs=batch.obs, rl2s=batch.rl2s)
            s_rep_base, _ = self._base_traj_encoder(
                seq=o_base, time_idxs=batch.time_idxs, hidden_state=None
            )
            logp_base = self._compute_log_probs(
                self._base_actor, s_rep_base, a_buffer, obs=batch.obs
            )

        # --- BC actor forward (pi_data, trainable head on frozen repr) ---
        logp_data = self._compute_log_probs(
            self._bc_actor, s_rep_base.detach(), a_buffer, obs=batch.obs
        )

        # --- IS correction (optional) ---
        delta_log = None
        if self.use_is_correction:
            delta_log = logp_base - logp_data.detach()
            self.fbc_filter_func.set_delta_log(delta_log)

        # --- Sequence-level filter mask ---
        if self.fbc_filter_func.seq_enabled:
            seq_mask = (~(batch.rl2s == self.pad_val).all(-1, keepdim=True)).bool()
            self.fbc_filter_func.set_seq_mask(seq_mask)

        # --- Standard MultiTaskAgent forward ---
        total_loss = super().forward(batch, log_step)

        # --- Auxiliary BC loss for _bc_actor (1:1 with parent masking) ---
        bc_loss_elems = -logp_data[:, :-1, ...]
        state_mask = (~(batch.rl2s == self.pad_val).all(-1, keepdim=True)).bool()[
            :, 1:, ...
        ]
        bc_mask = repeat(state_mask, f"b l 1 -> b l {G} 1")
        bc_mask = self.edit_actor_mask(batch, bc_loss_elems, bc_mask)
        bc_loss = amago.utils.masked_avg(bc_loss_elems, bc_mask)

        total_loss = total_loss + self.bc_coeff * bc_loss

        if log_step:
            self.update_info["BC Loss"] = bc_loss.detach()
            self.update_info["Log Pi Base (mean)"] = logp_base.mean().detach()
            self.update_info["Log Pi Data (mean)"] = logp_data.mean().detach()
            f = self.fbc_filter_func
            if f.seq_enabled and f._seq_last_weights is not None:
                sw = f._seq_last_weights
                sp = f._seq_last_percentiles
                self.update_info["Seq Filter Weight (mean)"] = sw.mean()
                self.update_info["Seq Filter Weight (min)"] = sw.min()
                self.update_info["Seq Filter Weight (std)"] = sw.std()
                self.update_info["Seq Filter Percentile (mean)"] = sp.mean()
                self.update_info["Seq Filter Effective Floor"] = f._seq_last_eff_floor
                self.update_info["Seq Filter Buffer Size"] = float(len(f._seq_buffer))
                buf = f._seq_sorted_buf()
                n = len(buf)
                if n > 1:
                    self.update_info["Seq Filter Adv @ p_low"] = buf[
                        min(int(n * f._seq_p_low), n - 1)
                    ]
                    self.update_info["Seq Filter Adv @ p_full"] = buf[
                        min(int(n * f._seq_p_full), n - 1)
                    ]
            if delta_log is not None:
                dl = delta_log.detach()
                dl_clipped = torch.clamp(
                    dl,
                    -self.fbc_filter_func.clip_delta,
                    self.fbc_filter_func.clip_delta,
                )
                self.update_info["IS Delta Log (mean)"] = dl.mean()
                self.update_info["IS Delta Log (std)"] = dl.std()
                self.update_info["IS Delta Log Clipped (mean)"] = dl_clipped.mean()
                self.update_info["IS Delta Log Clipped (std)"] = dl_clipped.std()
                pct_clipped = (
                    (dl.abs() > self.fbc_filter_func.clip_delta).float().mean()
                )
                self.update_info["IS Pct Clipped"] = pct_clipped
                self.update_info["IS Ratio (mean)"] = torch.exp(dl_clipped).mean()
                self.update_info["IS Ratio (std)"] = torch.exp(dl_clipped).std()

        return total_loss
