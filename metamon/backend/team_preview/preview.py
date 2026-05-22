"""
Team preview prediction: predict which pokemon to lead with given both teams.
12 inputs (6 ours + 6 opponent) -> 1 output (which of our 6 to lead).
Perceiver-style cross attention architecture.
"""

import os
import json
import random
import lz4.frame
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
import wandb
from tqdm import tqdm

from metamon.config import format_for_agent
from metamon.interface import (
    UniversalState,
    consistent_pokemon_order,
    consistent_move_order,
)
from metamon.tokenizer import PokemonTokenizer, get_tokenizer
from metamon.backend.replay_parser.str_parsing import pokemon_name, move_name
from metamon.il.model import CrossAttentionBlock, SelfAttentionBlock
from metamon.data.download import download_parsed_replays


class TeamPreviewDataset(Dataset):
    """Dataset for team preview prediction from parsed replays."""

    def __init__(
        self,
        tokenizer: PokemonTokenizer,
        battle_format: str = "gen9ou",
        dset_root: Optional[str] = None,
        min_rating: int = 1300,
        max_rating: Optional[int] = None,
        wins_losses_both: str = "both",
        max_samples: Optional[int] = None,
        shuffle: bool = True,
    ):
        self.tokenizer = tokenizer
        self.battle_format = battle_format
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.wins_losses_both = wins_losses_both
        self.shuffle = shuffle

        if dset_root is None:
            print(f"Downloading {battle_format} parsed replays...")
            format_path = download_parsed_replays(battle_format)
            dset_root = os.path.dirname(format_path)

        format_dir = os.path.join(dset_root, battle_format)
        if not os.path.exists(format_dir):
            raise ValueError(f"Format directory not found: {format_dir}")

        self.filenames = self._find_and_filter_files(format_dir)
        if len(self.filenames) == 0:
            raise ValueError(f"No replays found for {battle_format} with given filters")

        print(f"Found {len(self.filenames)} {battle_format} replays matching filters")

        if max_samples and max_samples < len(self.filenames):
            random.shuffle(self.filenames)
            self.filenames = self.filenames[:max_samples]
            print(f"Using {len(self.filenames)} samples")

    def _rating_to_int(self, rating_str: str) -> int:
        try:
            return int(rating_str)
        except ValueError:
            return 1000

    def _find_and_filter_files(self, format_dir: str) -> List[str]:
        filenames = []
        all_files = os.listdir(format_dir)
        json_files = [f for f in all_files if f.endswith((".json", ".json.lz4"))]

        has_rating_filter = self.min_rating is not None or self.max_rating is not None
        has_result_filter = self.wins_losses_both in ("wins", "losses")

        for filename in json_files:
            name_without_ext = (
                filename[:-9] if filename.endswith(".json.lz4") else filename[:-5]
            )
            parts = name_without_ext.split("_")
            if len(parts) != 7:
                continue

            battle_id, rating_str, p1_name, _, p2_name, mm_dd_yyyy, result = parts

            if has_result_filter:
                if self.wins_losses_both == "wins" and result != "WIN":
                    continue
                if self.wins_losses_both == "losses" and result != "LOSS":
                    continue

            battle_id_clean = (
                battle_id.replace("[", "").replace("]", "").replace(" ", "").lower()
            )
            if self.battle_format not in battle_id_clean:
                continue

            if has_rating_filter:
                rating = self._rating_to_int(rating_str)
                if (self.min_rating is not None and rating < self.min_rating) or (
                    self.max_rating is not None and rating > self.max_rating
                ):
                    continue

            filenames.append(os.path.join(format_dir, filename))

        if self.shuffle:
            random.shuffle(filenames)
        return filenames

    def __len__(self):
        return len(self.filenames)

    def _load_json(self, filename: str) -> dict:
        if filename.endswith(".lz4"):
            with lz4.frame.open(filename, "rb") as f:
                return json.load(f)
        with open(filename, "r") as f:
            return json.load(f)

    def __getitem__(
        self, idx
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """returns (team_tokens, additional_info_tokens, lead_idx, format_token)"""
        max_attempts = 100
        attempts = 0
        current_idx = idx

        while attempts < max_attempts:
            try:
                data = self._load_json(self.filenames[current_idx])
                first_state = UniversalState.from_dict(data["states"][0])

                our_team = [
                    first_state.player_active_pokemon
                ] + first_state.available_switches

                if len(our_team) != 6:
                    attempts += 1
                    current_idx = (current_idx + 1) % len(self.filenames)
                    continue

                our_team_sorted = consistent_pokemon_order(our_team)

                opponent_team_names = first_state.opponent_teampreview
                if len(opponent_team_names) != 6:
                    attempts += 1
                    current_idx = (current_idx + 1) % len(self.filenames)
                    continue

                opponent_team_sorted = consistent_pokemon_order(opponent_team_names)

                our_tokens = [
                    self.tokenizer[pokemon_name(p.name)] for p in our_team_sorted
                ]
                opp_tokens = [
                    self.tokenizer[pokemon_name(name)] for name in opponent_team_sorted
                ]
                team_tokens = torch.tensor(our_tokens + opp_tokens, dtype=torch.long)

                additional_info_tokens = []
                for p in our_team_sorted:
                    pokemon_info = []
                    moves = consistent_move_order(p.moves)[:4]
                    for move in moves:
                        pokemon_info.append(self.tokenizer[move_name(move.name)])
                    while len(pokemon_info) < 4:
                        pokemon_info.append(self.tokenizer["<blank>"])
                    pokemon_info.append(self.tokenizer[p.ability])
                    pokemon_info.append(self.tokenizer[p.item])
                    additional_info_tokens.append(pokemon_info)

                additional_info_tokens = torch.tensor(
                    additional_info_tokens, dtype=torch.long
                )

                lead_name = pokemon_name(first_state.player_active_pokemon.name)
                try:
                    lead_idx = next(
                        i
                        for i, p in enumerate(our_team_sorted)
                        if pokemon_name(p.name) == lead_name
                    )
                except StopIteration:
                    if attempts == 0:
                        print(f"WARNING: Active pokemon {lead_name} not found in team")
                    attempts += 1
                    current_idx = (current_idx + 1) % len(self.filenames)
                    continue

                format_str = f"<{self.battle_format}>"
                format_token = torch.tensor(
                    self.tokenizer[format_str], dtype=torch.long
                )

                return (
                    team_tokens,
                    additional_info_tokens,
                    torch.tensor(lead_idx, dtype=torch.long),
                    format_token,
                )

            except Exception as e:
                if attempts == 0:
                    print(f"ERROR loading {self.filenames[current_idx]}: {e}")
                attempts += 1
                current_idx = (current_idx + 1) % len(self.filenames)

        raise RuntimeError(
            f"Failed to load valid sample after {max_attempts} attempts from idx {idx}"
        )


class TeamPreviewModel(nn.Module):
    """Perceiver-style model: 12 pokemon + optional additional info -> predict lead (1 of 6)."""

    def __init__(
        self,
        tokenizer: PokemonTokenizer,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        latent_tokens: int = 4,
        dropout: float = 0.1,
        use_additional_info: bool = True,
        use_argmax: bool = False,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.d_model = d_model
        self.use_additional_info = use_additional_info
        self.use_argmax = use_argmax
        self.token_emb = nn.Embedding(len(tokenizer), d_model)
        self.pokemon_pos_emb = nn.Embedding(12, d_model)
        self.team_emb = nn.Embedding(2, d_model)

        if use_additional_info:
            self.info_pokemon_emb = nn.Embedding(6, d_model)
            self.info_slot_emb = nn.Embedding(6, d_model)
            self.type_emb = nn.Embedding(2, d_model)

        self.latents = nn.Parameter(torch.randn(latent_tokens, d_model) * 0.02)

        self.cross_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(d_model=d_model, n_heads=n_heads, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.self_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(d_model=d_model, n_heads=n_heads, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(latent_tokens * d_model, 6)

    def forward(
        self,
        team_tokens: torch.Tensor,
        additional_info_tokens: Optional[torch.Tensor] = None,
        format_token: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = team_tokens.shape[0]
        device = team_tokens.device

        pokemon_token_emb = self.token_emb(team_tokens)
        pokemon_pos_ids = torch.arange(12, device=device).unsqueeze(0).expand(B, -1)
        pokemon_pos_emb = self.pokemon_pos_emb(pokemon_pos_ids)

        team_ids = torch.cat(
            [
                torch.zeros(B, 6, dtype=torch.long, device=device),
                torch.ones(B, 6, dtype=torch.long, device=device),
            ],
            dim=1,
        )
        pokemon_team_emb = self.team_emb(team_ids)

        if self.use_additional_info:
            pokemon_type_ids = torch.zeros(B, 12, dtype=torch.long, device=device)
            pokemon_type_emb = self.type_emb(pokemon_type_ids)
            pokemon_emb = (
                pokemon_token_emb
                + pokemon_pos_emb
                + pokemon_team_emb
                + pokemon_type_emb
            )
        else:
            pokemon_emb = pokemon_token_emb + pokemon_pos_emb + pokemon_team_emb

        if self.use_additional_info:
            if additional_info_tokens is None:
                raise ValueError(
                    "additional_info_tokens required when use_additional_info=True"
                )

            info_token_emb = self.token_emb(additional_info_tokens)
            info_pokemon_ids = (
                torch.arange(6, device=device)
                .unsqueeze(0)
                .unsqueeze(-1)
                .expand(B, -1, 6)
            )
            info_slot_ids = (
                torch.arange(6, device=device)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(B, 6, -1)
            )

            info_pokemon_emb = self.info_pokemon_emb(info_pokemon_ids)
            info_slot_emb = self.info_slot_emb(info_slot_ids)
            info_team_ids = torch.zeros(B, 6, 6, dtype=torch.long, device=device)
            info_team_emb = self.team_emb(info_team_ids)
            info_type_ids = torch.ones(B, 6, 6, dtype=torch.long, device=device)
            info_type_emb = self.type_emb(info_type_ids)

            info_emb = (
                info_token_emb
                + info_pokemon_emb
                + info_slot_emb
                + info_team_emb
                + info_type_emb
            )
            info_emb = rearrange(info_emb, "b p i d -> b (p i) d")
            emb = torch.cat([pokemon_emb, info_emb], dim=1)
        else:
            emb = pokemon_emb

        if format_token is not None:
            format_emb = self.token_emb(format_token).unsqueeze(1)
            emb = torch.cat([format_emb, emb], dim=1)

        latents = self.latents.unsqueeze(0).expand(B, -1, -1)

        for cross, self_attn in zip(self.cross_blocks, self.self_blocks):
            latents = cross(latents, emb)
            latents = self_attn(latents)

        latents = self.final_norm(latents)
        latents_flat = rearrange(latents, "b n d -> b (n d)")
        return self.classifier(latents_flat)

    def predict_lead_from_state(
        self,
        state: UniversalState,
        device: Optional[str] = None,
    ) -> Tuple[str, torch.Tensor, List[str]]:
        """Predict lead from a UniversalState (should be first state with teampreview).

        Returns:
            Tuple of (predicted_lead_name, probs, sorted_team_names)
        """
        our_team = [state.player_active_pokemon] + state.available_switches
        if len(our_team) != 6:
            raise ValueError(f"Expected 6 pokemon in our team, got {len(our_team)}")

        opponent_team_names = state.opponent_teampreview
        if len(opponent_team_names) != 6:
            raise ValueError(
                f"Expected 6 pokemon in opponent teampreview, got {len(opponent_team_names)}"
            )

        return self.predict_lead(
            our_team=[p.name for p in our_team],
            our_team_moves=[[m.name for m in p.moves] for p in our_team],
            our_team_abilities=[p.ability for p in our_team],
            our_team_items=[p.item for p in our_team],
            opponent_team=opponent_team_names,
            device=device,
        )

    def predict_lead(
        self,
        our_team: List[str],
        our_team_moves: List[List[str]],
        our_team_abilities: List[str],
        our_team_items: List[str],
        opponent_team: List[str],
        battle_format: Optional[str] = None,
        device: Optional[str] = None,
    ) -> Tuple[str, torch.Tensor, List[str]]:
        """Predict which pokemon to lead with.

        Returns:
            Tuple of (predicted_lead_name, probs, sorted_team_names)
            - predicted_lead_name: Name of the pokemon to lead with
            - probs: Probability distribution over sorted team (probs[i] corresponds to sorted_team[i])
            - sorted_team_names: Team names in alphabetical order (matches prob indices)
        """
        if device is None:
            device = next(self.parameters()).device

        if battle_format is not None:
            battle_format = format_for_agent(battle_format)

        # sort teams consistently
        our_team_with_info = list(
            zip(our_team, our_team_moves, our_team_abilities, our_team_items)
        )
        our_team_with_info_sorted = sorted(
            our_team_with_info, key=lambda x: pokemon_name(x[0])
        )
        our_team_sorted = [name for name, _, _, _ in our_team_with_info_sorted]
        our_moves_sorted = [moves for _, moves, _, _ in our_team_with_info_sorted]
        our_abilities_sorted = [
            ability for _, _, ability, _ in our_team_with_info_sorted
        ]
        our_items_sorted = [item for _, _, _, item in our_team_with_info_sorted]
        opponent_team_sorted = consistent_pokemon_order(opponent_team)

        # tokenize pokemon
        our_tokens = [self.tokenizer[pokemon_name(name)] for name in our_team_sorted]
        opp_tokens = [
            self.tokenizer[pokemon_name(name)] for name in opponent_team_sorted
        ]
        team_tokens = torch.tensor([our_tokens + opp_tokens], dtype=torch.long).to(
            device
        )

        # tokenize additional info
        additional_info_tokens = None
        if self.use_additional_info:
            additional_info_tokens = []
            for moves, ability, item in zip(
                our_moves_sorted, our_abilities_sorted, our_items_sorted
            ):
                pokemon_info = []
                moves_sorted = consistent_move_order(moves)[:4]
                for move in moves_sorted:
                    pokemon_info.append(self.tokenizer[move_name(move)])
                while len(pokemon_info) < 4:
                    pokemon_info.append(self.tokenizer["<blank>"])
                pokemon_info.append(self.tokenizer[ability])
                pokemon_info.append(self.tokenizer[item])
                additional_info_tokens.append(pokemon_info)
            additional_info_tokens = torch.tensor(
                [additional_info_tokens], dtype=torch.long
            ).to(device)

        format_token = None
        if battle_format is not None:
            format_token = torch.tensor(
                [self.tokenizer[f"<{battle_format}>"]], dtype=torch.long
            ).to(device)

        self.eval()
        with torch.no_grad():
            logits = self(team_tokens, additional_info_tokens, format_token)
            probs = F.softmax(logits, dim=-1).squeeze(0)

        if self.use_argmax:
            lead_idx = probs.argmax().item()
        else:
            lead_idx = torch.multinomial(probs, num_samples=1).item()

        return our_team_sorted[lead_idx], probs, our_team_sorted

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        tokenizer: Optional[PokemonTokenizer] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_argmax: bool = False,
    ) -> "TeamPreviewModel":
        """load model from checkpoint. tokenizer auto-loaded if not provided."""
        ckpt = torch.load(checkpoint_path, map_location=device)

        # auto-load tokenizer if not provided
        if tokenizer is None:
            tokenizer_name = ckpt["hparams"]["data"].get(
                "tokenizer_name", "DefaultObservationSpace-v1"
            )
            tokenizer = get_tokenizer(tokenizer_name)

        model = cls(tokenizer=tokenizer, **ckpt["hparams"]["model"])
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()
        model.use_argmax = use_argmax

        model.trained_formats = ckpt["hparams"]["data"].get(
            "battle_formats", [ckpt["hparams"]["data"]["battle_format"]]
        )

        print(
            f"Loaded checkpoint from epoch {ckpt['epoch']} (val_acc={ckpt['val_acc']:.4f})"
        )
        print(f"Trained on formats: {model.trained_formats}")
        print(f"Using {'argmax' if use_argmax else 'sampling'} for lead selection")

        return model


def train_team_preview(
    tokenizer: PokemonTokenizer,
    save_dir: str,
    battle_format: str = "gen9ou",
    dset_root: Optional[str] = None,
    min_rating: int = 1300,
    max_rating: Optional[int] = None,
    wins_losses_both: str = "both",
    epochs: int = 10,
    steps_per_epoch: int = 1000,
    batch_size: int = 128,
    lr: float = 3e-4,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 3,
    latent_tokens: int = 4,
    dropout: float = 0.1,
    use_additional_info: bool = True,
    max_samples: Optional[int] = None,
    patience: int = 5,
    dloader_workers: int = 4,
    log_wandb: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """train team preview model with early stopping"""
    os.makedirs(save_dir, exist_ok=True)

    hparams = {
        "model": {
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "latent_tokens": latent_tokens,
            "dropout": dropout,
            "use_additional_info": use_additional_info,
        },
        "training": {
            "epochs": epochs,
            "steps_per_epoch": steps_per_epoch,
            "batch_size": batch_size,
            "lr": lr,
            "max_samples": max_samples,
            "patience": patience,
        },
        "data": {
            "battle_format": battle_format,
            "dset_root": dset_root,
            "min_rating": min_rating,
            "max_rating": max_rating,
            "wins_losses_both": wins_losses_both,
            "tokenizer_name": tokenizer.name,
            "train_size": None,
            "val_size": None,
        },
    }

    if log_wandb:
        wandb.init(project="metamon", entity="ut-austin-rpl-metamon", config=hparams)

    full_dataset = TeamPreviewDataset(
        tokenizer=tokenizer,
        battle_format=battle_format,
        dset_root=dset_root,
        min_rating=min_rating,
        max_rating=max_rating,
        wins_losses_both=wins_losses_both,
        max_samples=max_samples,
    )
    train_size = int(0.95 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    hparams["data"]["train_size"] = train_size
    hparams["data"]["val_size"] = val_size

    dloader_kwargs = {
        "batch_size": batch_size,
        "num_workers": dloader_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": dloader_workers > 0,
        "prefetch_factor": 2 if dloader_workers > 0 else None,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **dloader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **dloader_kwargs)

    model = TeamPreviewModel(tokenizer=tokenizer, **hparams["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"Training on {train_size} samples, validating on {val_size} samples")
    print(f"Steps per epoch: {steps_per_epoch}, Total epochs: {epochs}")

    best_val_acc = 0.0
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_checkpoint_path = os.path.join(save_dir, "best_model.pt")

    def infinite_dataloader(dataloader):
        while True:
            for batch in dataloader:
                yield batch

    train_iter = infinite_dataloader(train_loader)
    ema_alpha = 0.1

    for epoch in range(epochs):
        model.train()
        train_loss, train_acc, train_count = 0.0, 0.0, 0
        train_loss_ema, train_acc_ema = None, None

        train_pbar = tqdm(
            range(steps_per_epoch), desc=f"Epoch {epoch} [Train]", leave=False
        )
        for step in train_pbar:
            team_tokens, additional_info_tokens, lead_idx, format_token = next(
                train_iter
            )
            team_tokens = team_tokens.to(device)
            additional_info_tokens = additional_info_tokens.to(device)
            lead_idx = lead_idx.to(device)
            format_token = format_token.to(device)

            optimizer.zero_grad()
            logits = model(
                team_tokens,
                additional_info_tokens if model.use_additional_info else None,
                format_token,
            )
            loss = F.cross_entropy(logits, lead_idx)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(team_tokens)
            train_acc += (logits.argmax(1) == lead_idx).float().sum().item()
            train_count += len(team_tokens)

            batch_loss = loss.item()
            batch_acc = (logits.argmax(1) == lead_idx).float().mean().item()

            if train_loss_ema is None:
                train_loss_ema = batch_loss
                train_acc_ema = batch_acc
            else:
                train_loss_ema = (
                    ema_alpha * batch_loss + (1 - ema_alpha) * train_loss_ema
                )
                train_acc_ema = ema_alpha * batch_acc + (1 - ema_alpha) * train_acc_ema

            global_step = epoch * steps_per_epoch + step
            if log_wandb:
                wandb.log(
                    {
                        "train_loss_ema": train_loss_ema,
                        "train_acc_ema": train_acc_ema,
                        "global_step": global_step,
                    },
                    step=global_step,
                )

            train_pbar.set_postfix(
                {"loss": f"{train_loss_ema:.4f}", "acc": f"{train_acc_ema:.4f}"}
            )

        # validation
        model.eval()
        val_loss, val_acc, val_count = 0.0, 0.0, 0
        val_loss_ema, val_acc_ema = None, None

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False)
        with torch.no_grad():
            for team_tokens, additional_info_tokens, lead_idx, format_token in val_pbar:
                team_tokens = team_tokens.to(device)
                additional_info_tokens = additional_info_tokens.to(device)
                lead_idx = lead_idx.to(device)
                format_token = format_token.to(device)

                logits = model(
                    team_tokens,
                    additional_info_tokens if model.use_additional_info else None,
                    format_token,
                )
                loss = F.cross_entropy(logits, lead_idx)

                val_loss += loss.item() * len(team_tokens)
                val_acc += (logits.argmax(1) == lead_idx).float().sum().item()
                val_count += len(team_tokens)

                batch_loss = loss.item()
                batch_acc = (logits.argmax(1) == lead_idx).float().mean().item()

                if val_loss_ema is None:
                    val_loss_ema = batch_loss
                    val_acc_ema = batch_acc
                else:
                    val_loss_ema = (
                        ema_alpha * batch_loss + (1 - ema_alpha) * val_loss_ema
                    )
                    val_acc_ema = ema_alpha * batch_acc + (1 - ema_alpha) * val_acc_ema

                val_pbar.set_postfix(
                    {"loss": f"{val_loss_ema:.4f}", "acc": f"{val_acc_ema:.4f}"}
                )

        epoch_global_step = (epoch + 1) * steps_per_epoch
        metrics = {
            "epoch": epoch,
            "train_loss_epoch": train_loss / train_count,
            "train_acc_epoch": train_acc / train_count,
            "val_loss": val_loss / val_count,
            "val_acc": val_acc / val_count,
        }

        print(
            f"Epoch {epoch} (step {epoch_global_step}): "
            f"train_loss={metrics['train_loss_epoch']:.4f}, train_acc={metrics['train_acc_epoch']:.4f}, "
            f"val_loss={metrics['val_loss']:.4f}, val_acc={metrics['val_acc']:.4f}"
        )

        if log_wandb:
            wandb.log(metrics, step=epoch_global_step)

        if metrics["val_acc"] > best_val_acc:
            best_val_acc = metrics["val_acc"]
            best_val_loss = metrics["val_loss"]
            epochs_without_improvement = 0

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": epoch_global_step,
                "hparams": hparams,
                "train_loss": metrics["train_loss_epoch"],
                "train_acc": metrics["train_acc_epoch"],
                "val_loss": metrics["val_loss"],
                "val_acc": metrics["val_acc"],
            }
            torch.save(checkpoint, best_checkpoint_path)
            print(f"  New best model saved! val_acc={best_val_acc:.4f}")
        else:
            epochs_without_improvement += 1
            print(
                f"  No improvement for {epochs_without_improvement} epoch(s). Best={best_val_acc:.4f}"
            )

        if epochs_without_improvement >= patience:
            print(
                f"\nEarly stopping after {epoch + 1} epochs ({epoch_global_step} steps)"
            )
            print(f"Best val_acc: {best_val_acc:.4f}")
            break

        latest_checkpoint_path = os.path.join(save_dir, "latest_model.pt")
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": epoch_global_step,
            "hparams": hparams,
            "train_loss": metrics["train_loss_epoch"],
            "train_acc": metrics["train_acc_epoch"],
            "val_loss": metrics["val_loss"],
            "val_acc": metrics["val_acc"],
        }
        torch.save(checkpoint, latest_checkpoint_path)

    if log_wandb:
        wandb.finish()

    print(f"\nTraining complete")
    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"Best val_acc: {best_val_acc:.4f}, val_loss: {best_val_loss:.4f}")

    return TeamPreviewModel.load_from_checkpoint(
        best_checkpoint_path, tokenizer, device
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train team preview prediction model")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--battle_format", type=str, default="gen9ou")
    parser.add_argument("--dset_root", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default="DefaultObservationSpace-v1")
    parser.add_argument("--min_rating", type=int, default=1250)
    parser.add_argument("--max_rating", type=int, default=None)
    parser.add_argument(
        "--wins_losses_both",
        type=str,
        default="both",
        choices=["wins", "losses", "both"],
    )
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--steps_per_epoch", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--d_model", type=int, default=132)
    parser.add_argument("--n_heads", type=int, default=6)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--latent_tokens", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument(
        "--no-additional-info",
        action="store_false",
        dest="use_additional_info",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--dloader_workers", type=int, default=4)
    parser.add_argument("--no-wandb", action="store_true", dest="no_wandb")
    args = parser.parse_args()

    tokenizer = get_tokenizer(args.tokenizer)

    train_team_preview(
        tokenizer=tokenizer,
        save_dir=args.save_dir,
        battle_format=args.battle_format,
        dset_root=args.dset_root,
        min_rating=args.min_rating,
        max_rating=args.max_rating,
        wins_losses_both=args.wins_losses_both,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        lr=args.lr,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        latent_tokens=args.latent_tokens,
        dropout=args.dropout,
        use_additional_info=args.use_additional_info,
        max_samples=args.max_samples,
        patience=args.patience,
        dloader_workers=args.dloader_workers,
        log_wandb=not args.no_wandb,
    )
