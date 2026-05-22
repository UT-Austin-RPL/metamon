import os
from dataclasses import dataclass
from typing import Callable, Dict
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import wandb
import numpy as np
from tqdm import tqdm
from einops import rearrange
import gin

import metamon
from metamon.il.model import (
    GRUModel,
    MetamonILModel,
)
from metamon.tokenizer import get_tokenizer
from metamon.interface import (
    TokenizedObservationSpace,
    DefaultObservationSpace,
    DefaultShapedReward,
    DefaultActionSpace,
)
from metamon.data import ParsedReplayDataset

MAGIC_PAD_VAL = -1  # note we set the seq pad value to the same value as missing actions
pad = partial(pad_sequence, batch_first=True, padding_value=MAGIC_PAD_VAL)


@dataclass
class Run:
    gpu: int
    parsed_replay_dataset: ParsedReplayDataset
    run_name: str

    # Logging
    save_dir: str = "logs_and_checkpoints"
    log_to_wandb: bool = False
    wandb_project: str = None
    wandb_entity: str = None
    wandb_group_name: str = None
    verbose: bool = True
    ckpt_interval: int = 10
    log_interval: int = 250

    # Optimization
    model_Cls: Callable = GRUModel
    mask_actions: bool = True
    batch_size: int = 48
    dloader_workers: int = 8
    learning_rate: float = 1e-4
    grad_clip: float = 2.0
    l2_coeff: float = 1e-4
    early_stopping_patience: int = 2
    epochs: int = 500

    def start(self):
        self.DEVICE = torch.device(f"cuda:{self.gpu}" if self.gpu >= 0 else "cpu")
        self.init_checkpoints()
        self.init_dsets()
        self.init_model()
        self.init_optimizer()
        self.init_logger()

    def init_checkpoints(self):
        self.best_accuracy = -float("inf")
        self.ckpt_dir = os.path.join(self.save_dir, self.run_name, "ckpts")
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        self.epoch = 0

    def load_checkpoint(self, epoch: int):
        ckpt_name = f"{self.run_name}_epoch_{epoch}.ptstate"
        ckpt = torch.load(
            os.path.join(self.ckpt_dir, ckpt_name), map_location=self.DEVICE
        )
        self.policy.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.epoch = ckpt["epoch"]
        self.best_accuracy = ckpt["best_accuracy"]

    def save_checkpoint(self, saving_best: bool = False):
        if not saving_best:
            # save a training state that lets us resume training
            # by creating an identical `Run`
            state_dict = {
                "model_state": self.policy.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "epoch": self.epoch,
                "best_accuracy": self.best_accuracy,
            }
            ckpt_name = f"{self.run_name}_epoch_{self.epoch}.ptstate"
            torch.save(state_dict, os.path.join(self.ckpt_dir, ckpt_name))
        else:
            # save the raw model so we can conveniently load w/o first
            # reconstructing a blank copy
            model_name = f"{self.run_name}_BEST.pt"
            torch.save(self.policy, os.path.join(self.ckpt_dir, model_name))

    def pad_collate_replay_data_for_il(self, samples):
        # each sample is (obs, action_info, reward, done)
        # we don't need the mission action mask because we'll ignore all -1s anyway
        obs = {
            k: pad([torch.from_numpy(np.array(s[0][k])) for s in samples])
            for k in samples[0][0].keys()
        }

        # build action masks from action info dicts
        batched_action_idxs = []
        batched_illegal_action_masks = []
        num_actions = self.parsed_replay_dataset.action_space.gym_space.n
        for sample in samples:
            action_idxs = []
            illegal_action_masks = []
            action_infos = sample[1]
            for i in range(len(action_infos["chosen"])):
                action_idxs.append(action_infos["chosen"][i])
                illegal = torch.ones(num_actions, dtype=bool)
                for legal_action in action_infos["legal"][i]:
                    illegal[legal_action] = False
                illegal_action_masks.append(illegal)
            batched_action_idxs.append(torch.tensor(action_idxs, dtype=torch.int32))
            batched_illegal_action_masks.append(
                torch.stack(illegal_action_masks, dim=0)
            )
        action_idxs = pad(batched_action_idxs)
        illegal_action_masks = pad(batched_illegal_action_masks)

        return obs, action_idxs, illegal_action_masks

    def init_dsets(self):
        train_size = int(0.9 * len(self.parsed_replay_dataset))
        val_size = len(self.parsed_replay_dataset) - train_size
        # Use random_split to create train/val datasets
        self.train_dset, self.val_dset = torch.utils.data.random_split(
            self.parsed_replay_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(231),
        )
        if self.verbose:
            print(f"Training on {train_size:,d} battles")
            print(f"Validating on {val_size:,d} battles")

        dloader_kwargs = dict(
            batch_size=self.batch_size,
            num_workers=self.dloader_workers,
            pin_memory=True,
            collate_fn=self.pad_collate_replay_data_for_il,
            shuffle=True,
        )
        self.train_dloader = DataLoader(self.train_dset, **dloader_kwargs)
        self.val_dloader = DataLoader(self.val_dset, **dloader_kwargs)

    def init_logger(self):
        gin_config = gin.operative_config_str()
        config_path = os.path.join(self.save_dir, self.run_name, "config.txt")
        with open(config_path, "w") as f:
            f.write(gin_config)
        if self.log_to_wandb:
            log_dir = os.path.join(self.save_dir, self.run_name, "wandb_logs")
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            wandb.init(
                project=self.wandb_project,
                entity=self.wandb_entity,
                dir=log_dir,
                name=self.run_name,
                group=self.wandb_group_name,
            )
            wandb.save(config_path)

    def init_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.learning_rate,
            weight_decay=self.l2_coeff,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=1e-4, last_epoch=-1
        )

    def init_model(self):
        # build model, move to GPU
        obs_space = self.parsed_replay_dataset.observation_space
        action_space = self.parsed_replay_dataset.action_space
        assert isinstance(obs_space, TokenizedObservationSpace)
        self.policy = self.model_Cls(
            tokenizer=obs_space.tokenizer,
            text_features=obs_space.base_obs_space.tokenizable["text"],
            numerical_features=obs_space.gym_space["numbers"].shape[0],
            num_actions=action_space.gym_space.n,
        )
        assert isinstance(self.policy, MetamonILModel)
        self.policy.to(self.DEVICE)

        # log, display total parameter count
        total_params = sum(
            param.numel() if param.requires_grad else 0
            for name, param in self.policy.named_parameters()
        )
        if self.verbose:
            print(f"Initialized Model with {total_params:,d} Parameters")

    def log(self, key: str, metrics_dict: Dict[str, float | torch.Tensor]):
        log_dict = {}
        for k, v in metrics_dict.items():
            if isinstance(v, torch.Tensor):
                if v.ndim == 0:
                    log_dict[k] = v.detach().cpu().float().item()
            else:
                log_dict[k] = v
        if self.log_to_wandb:
            wandb.log({f"{key}/{subkey}": val for subkey, val in log_dict.items()})

    def compute_loss(
        self,
        inputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        action_masks: torch.Tensor,
    ):
        # the last observation does not have an action (label)
        inputs = {k: v[:, :-1, ...].to(self.DEVICE) for k, v in inputs.items()}
        labels = labels.to(self.DEVICE)
        predictions, _ = self.policy(
            token_inputs=inputs["text_tokens"], numerical_inputs=inputs["numbers"]
        )
        if self.mask_actions:
            predictions.masked_fill_(action_masks.to(self.DEVICE), -float("inf"))

        loss = F.cross_entropy(
            rearrange(predictions, "b l d -> (b l) d"),
            rearrange(labels.to(dtype=torch.long), "b l -> (b l)"),
            ignore_index=MAGIC_PAD_VAL,
        )
        with torch.no_grad():
            mask = labels != MAGIC_PAD_VAL
            correct = predictions.argmax(-1) == labels
            acc = (mask * correct).sum() / mask.sum()
            top2 = predictions.topk(2, dim=-1).indices
            top2_correct = top2.eq(labels.unsqueeze(-1)).any(-1)
            top2_acc = (mask * top2_correct).sum() / mask.sum()
        return loss, acc, top2_acc

    def _get_grad_norms(self):
        total_norm = 0.0
        for p in self.policy.parameters():
            try:
                param = p.grad.data
            except AttributeError:
                continue
            else:
                param_norm = param.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1.0 / 2)
        return total_norm

    def train_step(
        self,
        inputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        action_masks: torch.Tensor,
        log_step: bool,
    ):
        loss, acc, top2_acc = self.compute_loss(
            inputs, labels, action_masks=action_masks
        )
        self.optimizer.zero_grad()
        loss.backward()
        if log_step:
            self.log("train", {"Loss": loss, "Grad Norm": self._get_grad_norms()})
        grad_norm = nn.utils.clip_grad_norm_(
            self.policy.parameters(), max_norm=self.grad_clip
        )
        self.optimizer.step()
        return loss, acc, top2_acc

    def val_step(
        self,
        inputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        action_masks: torch.Tensor,
    ):
        with torch.no_grad():
            loss, acc, top2_acc = self.compute_loss(
                inputs, labels, action_masks=action_masks
            )
        return loss, acc, top2_acc

    def learn(self):
        since_best = 0
        for epoch in range(self.epoch, self.epochs):
            train_acc = []
            train_top2_acc = []
            self.policy.train()
            for step, (inputs, labels, action_masks) in tqdm(
                enumerate(self.train_dloader),
                colour="white",
                total=len(self.train_dloader),
                desc=f"Epoch: {epoch} (Train)",
            ):
                log_step = step % self.log_interval == 0
                step_loss, step_acc, step_top2_acc = self.train_step(
                    inputs, labels, action_masks=action_masks, log_step=log_step
                )
                train_acc.append(step_acc)
                train_top2_acc.append(step_top2_acc)
                if step and step % self.log_interval == 0:
                    recent_acc = train_acc[-self.log_interval :]
                    recent_top2_acc = train_top2_acc[-self.log_interval :]
                    sample_size = len(recent_acc)
                    self.log(
                        "train",
                        {
                            "Recent Accuracy": sum(recent_acc) / sample_size,
                            "Recent Top-2 Accuracy": sum(recent_top2_acc) / sample_size,
                        },
                    )

            self.scheduler.step()

            val_acc, val_loss, val_top2_acc = [], [], []
            self.policy.eval()
            for step, (inputs, labels, action_masks) in tqdm(
                enumerate(self.val_dloader), colour="red", desc=f"Epoch: {epoch} (Val)"
            ):
                step_loss, step_acc, step_top2_acc = self.val_step(
                    inputs, labels, action_masks=action_masks
                )
                val_acc.append(step_acc)
                val_top2_acc.append(step_top2_acc)
                val_loss.append(step_loss)
            avg_val_acc = sum(val_acc) / len(val_acc)
            avg_val_loss = sum(val_loss) / len(val_loss)
            avg_val_top2_acc = sum(val_top2_acc) / len(val_top2_acc)
            self.log(
                "val",
                {
                    "Accuracy": avg_val_acc,
                    "Loss": avg_val_loss,
                    "Top-2 Accuracy": avg_val_top2_acc,
                },
            )

            if self.verbose:
                print(f"Validation Accuracy: {avg_val_acc * 100: .2f}%")
                print(f"Validation Loss: {avg_val_loss : .3f}")
                print(f"Validation Top-2 Accuracy: {avg_val_top2_acc * 100: .2f}%")

            if avg_val_acc > self.best_accuracy:
                self.best_accuracy = avg_val_acc
                since_best = 0
                self.save_checkpoint(saving_best=True)
            else:
                since_best += 1

            early_stop = since_best >= self.early_stopping_patience
            self.epoch = epoch
            if (
                epoch % self.ckpt_interval == 0
                or epoch == self.epochs - 1
                or early_stop
            ):
                self.save_checkpoint()
            if early_stop:
                if self.verbose:
                    print("Early Stopping!")
                break


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="A quick way to call the built-in IL training loop with default hyperapramters."
    )
    parser.add_argument(
        "--run_name",
        required=True,
        help="Give the model a name (for checkpoints and logging)",
    )
    parser.add_argument(
        "--gpu", type=int, required=True, help="GPU device index to use."
    )
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument(
        "--formats",
        default=metamon.config.SUPPORTED_BATTLE_FORMATS,
        nargs="+",
        help="Filter the dataset directory by pokemon format (e.g., `--formats gen1ou gen2nu gen3ou`",
    )
    parser.add_argument(
        "--wins_losses_both",
        choices=["wins", "losses", "both"],
        default="both",
        help="Filter the dataset directory to only include wins or losses.",
    )
    parser.add_argument(
        "--parsed_replay_dir",
        type=str,
        default=None,
        help="Path to the directory of parsed replays. Defaults to the official huggingface version.",
    )
    parser.add_argument(
        "--wandb_username",
        type=str,
        help="Weights and Biases username. Logging is enabled when both username and project are specified.",
        default=None,
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        help="Weights and Biases project name. Logging is enabled when both username and project are specified.",
        default=None,
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="DefaultObservationSpace-v0",
    )
    parser.add_argument(
        "--turn_embedding",
        choices=["transformer", "ff"],
        default="transformer",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--model_config",
        type=str,
        required=True,
        help="Path to `.gin` configuration file for the model's hyperparameters.",
    )
    args = parser.parse_args()

    gin.parse_config_file(args.model_config)
    parsed_replay_dataset = ParsedReplayDataset(
        dset_root=args.parsed_replay_dir,
        observation_space=TokenizedObservationSpace(
            tokenizer=get_tokenizer(args.tokenizer),
            base_obs_space=DefaultObservationSpace(),
        ),
        action_space=DefaultActionSpace(),
        reward_function=DefaultShapedReward(),
        formats=args.formats,
        wins_losses_both=args.wins_losses_both,
        max_seq_len=args.max_seq_len,
        verbose=True,
    )

    enable_logging = args.wandb_project is not None and args.wandb_username is not None
    for trial in range(args.trials):
        run = Run(
            parsed_replay_dataset=parsed_replay_dataset,
            gpu=args.gpu,
            run_name=f"{args.run_name}_trial{trial+1}",
            log_to_wandb=enable_logging,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_username,
        )
        run.start()
        run.learn()
        wandb.finish()
