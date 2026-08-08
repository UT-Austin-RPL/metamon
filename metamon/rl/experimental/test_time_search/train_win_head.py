"""kimi-search M3: train the win-probability head on frozen squirtle embeddings.

Loads the frozen squirtle policy (backbone), streams battles from the online
FIFO buffer (self-play, WIN/LOSS in filenames), computes per-turn trajectory
embeddings with the frozen backbone, and trains a :class:`WinHead` (per-action
win-probability logits) with binary cross-entropy against the battle outcome.

Usage::

    export METAMON_CACHE_DIR=/home/eddie/metamon_cache
    uv run python -m metamon.rl.experimental.test_time_search.train_win_head \
        --checkpoint 975 --buffer_root ~/metamon_runs/mini_online_smogon_v0/buffer \
        --format gen1ou --max_battles 4000 --epochs 2 \
        --out /tmp/tts_kimi_m3/win_head.pt
"""

from __future__ import annotations

import argparse
import os
import re
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from metamon.data.parsed_replay_dset import MetamonDataset
from metamon.rl.metamon_to_amago import MetamonAMAGODataset
from metamon.rl.pretrained import get_pretrained_model
from amago.loading import Batch

from .win_head import WinHead, save_win_head

_RESULT_RE = re.compile(r"_(WIN|LOSS)\.json\.lz4$")


def _result_of(path: str) -> int:
    m = _RESULT_RE.search(os.path.basename(path))
    if m is None:
        return -1  # unknown / draw
    return 1 if m.group(1) == "WIN" else 0


@torch.no_grad()
def _embed_battle(agent, amago_dset, raw, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Frozen-backbone per-turn embeddings + taken-action index for one battle.

    Returns (emb (T, d_model) on CPU, a_t (T,) long). The AMAGO timestep t
    embedding is the state the t-th action is taken from; there are T+1
    timesteps for T actions (the extra is the terminal state). We keep the
    first T (decision states) and the taken (argmax of the one-hot) action.
    """
    rldata = amago_dset._process_data(raw)
    batch = Batch(
        obs={k: v.unsqueeze(0).to(device) for k, v in rldata.obs.items()},
        rl2s=rldata.rl2s.unsqueeze(0).to(device),
        rews=rldata.rews.unsqueeze(0).to(device),
        dones=rldata.dones.unsqueeze(0).to(device),
        actions=rldata.actions.unsqueeze(0).to(device),
        time_idxs=rldata.time_idxs.unsqueeze(0).to(device),
    )
    emb, _ = agent.get_state_embedding(
        obs=batch.obs,
        rl2s=batch.rl2s,
        time_idxs=batch.time_idxs,
        hidden_state=None,
    )
    emb = emb[0].detach().cpu()  # (L, d_model), L = T+1
    T = batch.actions.shape[1]
    a_t = batch.actions[0, :T].argmax(-1).detach().cpu()  # (T,) taken action
    return emb[:T], a_t


def _collect(
    agent, dset, amago_dset, files: List[str], device: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Embed every battle; return (emb (N, d), a_t (N,), y (N,)) per-turn rows."""
    embs, acts, ys = [], [], []
    n_skip = 0
    for i, path in enumerate(files):
        y = _result_of(path)
        if y < 0:
            n_skip += 1
            continue
        try:
            raw = dset.load_filename(path)
            emb, a_t = _embed_battle(agent, amago_dset, raw, device)
        except Exception:
            n_skip += 1
            continue
        T = a_t.numel()
        if T < 2:
            n_skip += 1
            continue
        embs.append(emb)
        acts.append(a_t)
        ys.append(torch.full((T,), float(y)))
        if (i + 1) % 200 == 0:
            print(
                f"  embedded {i+1}/{len(files)} battles ({n_skip} skipped)", flush=True
            )
    X = torch.cat(embs, 0)
    A = torch.cat(acts, 0)
    Y = torch.cat(ys, 0)
    return X, A, Y


def _auc(scores: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    n1 = float(y.sum())
    n0 = float(len(y) - n1)
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="squirtle")
    ap.add_argument("--checkpoint", type=int, default=975)
    ap.add_argument(
        "--buffer_root",
        default=os.path.expanduser("~/metamon_runs/mini_online_smogon_v0/buffer"),
    )
    ap.add_argument("--format", default="gen1ou")
    ap.add_argument("--max_battles", type=int, default=4000)
    ap.add_argument("--val_fraction", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d_hidden", type=int, default=512)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--embeddings_cache",
        default=None,
        help="optional .npz path: load cached (X, A, Y) splits if present, "
        "else embed and save. Lets capacity sweeps reuse one embedding pass.",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"

    cache_only = args.embeddings_cache and os.path.exists(args.embeddings_cache)
    if cache_only:
        # skip loading the 35M model when we already have embeddings
        agent = None
        maker = get_pretrained_model(args.agent)
        dset = amago_dset = None
        train_files = val_files = []
        n_val = 0
    else:
        maker = get_pretrained_model(args.agent)
        experiment = maker.initialize_agent(checkpoint=args.checkpoint, log=False)
        agent = experiment.policy.to(device)
        agent.eval()
        for p in agent.parameters():
            p.requires_grad_(False)

    if not cache_only:
        dset = MetamonDataset(
            dset_root=args.buffer_root,
            observation_space=maker.observation_space,
            action_space=maker.action_space,
            reward_function=maker.reward_function,
            formats=[args.format],
            verbose=False,
            write_index_cache=False,
        )
        amago_dset = MetamonAMAGODataset(dset)

        files = [f for f in dset.filenames if _RESULT_RE.search(os.path.basename(f))]
        rng = np.random.default_rng(args.seed)
        rng.shuffle(files)
        files = files[: args.max_battles]
        n_val = max(int(len(files) * args.val_fraction), 1)
        val_files, train_files = files[:n_val], files[n_val:]
        print(
            f"train battles: {len(train_files)}, val battles: {len(val_files)}",
            flush=True,
        )

    t0 = time.perf_counter()
    cache = args.embeddings_cache
    if cache and os.path.exists(cache):
        z = np.load(cache)
        Xtr = torch.from_numpy(z["Xtr"])
        Atr = torch.from_numpy(z["Atr"])
        Ytr = torch.from_numpy(z["Ytr"])
        Xva = torch.from_numpy(z["Xva"])
        Ava = torch.from_numpy(z["Ava"])
        Yva = torch.from_numpy(z["Yva"])
        print(f"loaded embeddings from {cache}", flush=True)
    else:
        print("embedding train split...", flush=True)
        Xtr, Atr, Ytr = _collect(agent, dset, amago_dset, train_files, device)
        print("embedding val split...", flush=True)
        Xva, Ava, Yva = _collect(agent, dset, amago_dset, val_files, device)
        if cache:
            os.makedirs(os.path.dirname(os.path.abspath(cache)), exist_ok=True)
            np.savez(
                cache,
                Xtr=Xtr.numpy(),
                Atr=Atr.numpy(),
                Ytr=Ytr.numpy(),
                Xva=Xva.numpy(),
                Ava=Ava.numpy(),
                Yva=Yva.numpy(),
            )
            print(f"saved embeddings -> {cache}", flush=True)
    print(
        f"embedded: train {tuple(Xtr.shape)}, val {tuple(Xva.shape)} "
        f"in {time.perf_counter()-t0:.0f}s",
        flush=True,
    )

    d_model = Xtr.shape[1]
    head = WinHead(
        d_model=d_model,
        action_dim=maker.action_space.gym_space.n,
        d_hidden=args.d_hidden,
        n_layers=args.n_layers,
        dropout_p=args.dropout,
    ).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)

    Xtr_d, Atr_d, Ytr_d = Xtr.to(device), Atr.to(device), Ytr.to(device)
    Xva_d, Ava_d, Yva_d = Xva.to(device), Ava.to(device), Yva.to(device)
    n = Xtr_d.shape[0]
    for epoch in range(args.epochs):
        head.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, args.batch_size):
            idx = perm[i : i + args.batch_size]
            logits = head(Xtr_d[idx])  # (B, A)
            # supervise Q_win(s_t, a_t) against the outcome for the action
            # actually taken (the search needs per-action Q(s,a); the taken
            # action is the labeled one, and "then follow the base policy" is
            # satisfied because the rest of the battle IS the base policy).
            q_taken = torch.sigmoid(
                logits.gather(1, Atr_d[idx].unsqueeze(1)).squeeze(1)
            )
            loss = F.binary_cross_entropy(q_taken, Ytr_d[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * idx.numel()
        train_loss = tot / n

        head.eval()
        with torch.no_grad():
            logits_va = head(Xva_d)
            q_va = torch.sigmoid(logits_va.gather(1, Ava_d.unsqueeze(1)).squeeze(1))
            val_loss = float(F.binary_cross_entropy(q_va, Yva_d))
            auc = _auc(q_va.cpu().numpy(), Yva_d.cpu().numpy())
            brier = float(((q_va - Yva_d) ** 2).mean())
        print(
            f"epoch {epoch}: train_loss {train_loss:.4f} "
            f"val_loss {val_loss:.4f} val_auc {auc:.3f} val_brier {brier:.4f}",
            flush=True,
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_win_head(
        head,
        args.out,
        meta={
            "agent": args.agent,
            "checkpoint": args.checkpoint,
            "n_train_turns": int(n),
            "n_train_battles": len(train_files),
            "n_val_battles": len(val_files),
            "val_auc": auc,
            "val_brier": brier,
            "d_hidden": args.d_hidden,
            "n_layers": args.n_layers,
        },
    )
    print(f"saved win head -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
