"""kimi-search M4: distill the terminal-win signal into the squirtle policy.

The fixed-root gates (M1/M3) showed no *leaf value* (shaped critic, gamma
heads, or a trained win head) is action-aligned enough to beat the actor --
the frozen representation is the ceiling. But the **terminal-win selector**
(term_G) is an excellent action ranker on fixed roots (regret 0.014-0.019 vs
the actor's 0.072-0.091). M4 moves that signal into the **weights**: fine-tune
the policy so its own action distribution puts more mass on the
terminal-win-best action at each captured root.

Data: ``distill_labels.jsonl`` from ``terminal_win --save_distill`` -- one row
per root with the raw observation, legal actions, and the terminal-win-best
action label (from G to-terminal continuations per action).

Loss (per root): cross-entropy of the actor's legal-action distribution
against the one-hot terminal-win-best label, plus a KL anchor to the frozen
base policy (so distillation doesn't collapse the policy off the captured
roots). The backbone is fine-tunable (that is the point -- the representation
must adapt), with a low LR and the anchor to keep it close to the base.

Usage::

    export METAMON_CACHE_DIR=/home/eddie/metamon_cache
    uv run python -m metamon.rl.experimental.test_time_search.distill_win \
        --checkpoint 975 --labels /tmp/tts_kimi_m4_data/distill_labels.jsonl \
        --epochs 5 --lr 1e-5 --kl_anchor 1.0 \
        --out /home/eddie/metamon_runs/kimi_search_base/squirtle_distilled.pt
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from metamon.rl.pretrained import get_pretrained_model


def _load_labels(path: str) -> List[Dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _obs_to_torch(obs: Dict[str, list], device) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in obs.items():
        t = torch.from_numpy(np.asarray(v))
        if t.is_floating_point():
            t = t.float()
        if t.ndim == 1:
            t = t.unsqueeze(0)  # (1, feat)
        out[k] = t.unsqueeze(0).to(device)  # (B=1, L=1, ...)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="squirtle")
    ap.add_argument("--checkpoint", type=int, default=975)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--kl_anchor", type=float, default=1.0)
    ap.add_argument("--val_fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"

    maker = get_pretrained_model(args.agent)
    experiment = maker.initialize_agent(checkpoint=args.checkpoint, log=False)
    agent = experiment.policy.to(device)

    # frozen reference for the KL anchor
    experiment_ref = maker.initialize_agent(checkpoint=args.checkpoint, log=False)
    ref = experiment_ref.policy.to(device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    rows = _load_labels(args.labels)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(rows)
    n_val = max(int(len(rows) * args.val_fraction), 1)
    val_rows, train_rows = rows[:n_val], rows[n_val:]
    print(f"distill roots: {len(train_rows)} train / {len(val_rows)} val", flush=True)

    opt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, agent.parameters()), lr=args.lr
    )
    agent.train()

    def root_loss(row, policy, train: bool):
        obs = _obs_to_torch(row["obs"], device)
        legal = torch.tensor(row["legal_actions"], device=device).long()
        label = int(row["label"])
        # rl2 / time_idxs: single-step decision context (no history) -- the
        # benchmark captured roots with their full history, but for
        # distillation we condition on the current observation only (the
        # grouped obs space carries the battle state). rl2 zeros.
        rl2 = torch.zeros((1, 1, policy.rl2_space.shape[-1]), device=device)
        tidx = torch.zeros((1, 1, 1), dtype=torch.long, device=device)
        emb, _ = policy.get_state_embedding(
            obs=obs, rl2s=rl2, time_idxs=tidx, hidden_state=None
        )
        illegal = obs["illegal_actions"].bool()
        dist = policy.actor(emb, straight_from_obs={"illegal_actions": illegal})
        # dist.probs: (B, L=1, G, A); take primary gamma (-1) and squeeze to (A,)
        logp = torch.log(
            dist.probs[..., -1, :].squeeze(1)[0].clamp_min(1e-12)
        )  # (A,)
        y = torch.zeros_like(logp)
        y[label] = 1.0
        ce = -(y * logp).sum()
        if train and args.kl_anchor > 0:
            with torch.no_grad():
                emb_r, _ = ref.get_state_embedding(
                    obs=obs, rl2s=rl2, time_idxs=tidx, hidden_state=None
                )
                dist_r = ref.actor(
                    emb_r, straight_from_obs={"illegal_actions": illegal}
                )
                logp_r = torch.log(
                    dist_r.probs[..., -1, :].squeeze(1)[0].clamp_min(1e-12)
                )
            kl = logp - logp_r
            kl = (torch.exp(logp) * kl).sum()
            return ce + args.kl_anchor * kl, ce.detach()
        return ce, ce.detach()

    for epoch in range(args.epochs):
        tot = 0.0
        for row in train_rows:
            loss, ce = root_loss(row, agent, train=True)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(ce)
        # val: top-1 match of the distilled policy with the terminal-win label
        agent.eval()
        correct = 0
        with torch.no_grad():
            for row in val_rows:
                obs = _obs_to_torch(row["obs"], device)
                rl2 = torch.zeros((1, 1, agent.rl2_space.shape[-1]), device=device)
                tidx = torch.zeros((1, 1, 1), dtype=torch.long, device=device)
                emb, _ = agent.get_state_embedding(
                    obs=obs, rl2s=rl2, time_idxs=tidx, hidden_state=None
                )
                illegal = obs["illegal_actions"].bool()
                dist = agent.actor(emb, straight_from_obs={"illegal_actions": illegal})
                pred = int(dist.probs[..., -1, :].squeeze(1)[0].argmax())
                correct += int(pred == int(row["label"]))
        agent.train()
        print(
            f"epoch {epoch}: train CE {tot/len(train_rows):.4f} "
            f"val label top-1 {correct}/{len(val_rows)} = {correct/len(val_rows):.3f}",
            flush=True,
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(agent.state_dict(), args.out)
    print(f"saved distilled policy -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
