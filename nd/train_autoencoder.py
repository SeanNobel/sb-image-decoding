import os, sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from time import time
from tqdm import tqdm
from termcolor import cprint
from typing import Union, Optional, List
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

from nd.datasets import ThingsMEGBrainDataset, ImageNetEEGBrainDataset
from nd.models import BrainEncoder, BrainDecoder
from nd.utils.train_utils import count_parameters


class BrainAutoencoder(nn.Module):
    def __init__(self, args, subjects) -> None:
        super().__init__()

        self.encoder = BrainEncoder(args, subjects)
        self.decoder = BrainDecoder(
            args.F_mse,
            args.num_channels,
            int(args.seq_len * args.brain_resample_sfreq),
            mid_channels=args.decoder_dim,
        )

    def forward(self, X: torch.Tensor, subject_idxs: torch.Tensor):
        Z = self.encoder(X, subject_idxs)["Z_mse"]
        if Z.ndim == 3:
            Z = rearrange(Z, "b () f -> b f")

        return self.decoder(Z)


def train():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_name = args.train_name

    if sweep:
        wandb.init(config=None)

        run_name += "_" + "".join(
            [
                f"{k}-{v:.3f}_" if isinstance(v, float) else f"{k}-{v}_"
                for k, v in wandb.config.items()
            ]
        )

        wandb.run.name = run_name
        args.__dict__.update(wandb.config)
        cprint(wandb.config, "cyan")
        wandb.config.update(args.__dict__)

    run_dir = os.path.join("runs", args.dataset.lower(), run_name)
    os.makedirs(run_dir, exist_ok=True)

    device = f"cuda:{args.cuda_id}"

    assert args.num_clip_tokens == 1

    # -----------------------
    #       Dataloader
    # -----------------------
    dataset = eval(f"{args.dataset}Dataset")(args)

    if hasattr(dataset, "train_idxs"):
        train_set = torch.utils.data.Subset(dataset, dataset.train_idxs)
        test_set = torch.utils.data.Subset(dataset, dataset.test_idxs)
    else:
        train_size = int(len(dataset) * 0.8)
        train_set, test_set = torch.utils.data.random_split(
            dataset, [train_size, len(dataset) - train_size]
        )

    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "drop_last": False,
    }
    train_loader = torch.utils.data.DataLoader(dataset=train_set, shuffle=True, **loader_args)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, shuffle=False, **loader_args)

    # ---------------------
    #        Models
    # ---------------------
    subjects = dataset.subject_names if hasattr(dataset, "subject_names") else dataset.num_subjects  # fmt: skip

    model = BrainAutoencoder(args, subjects).to(device)

    if sweep:
        wandb.config.update({"brain_encoder_params": count_parameters(model)})

    # ---------------------
    #      Optimizers
    # ---------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
        )
    elif args.lr_scheduler == "multistep":
        mlstns = [int(m * args.epochs) for m in args.lr_multistep_mlstns]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=mlstns, gamma=args.lr_step_gamma
        )
    else:
        cprint("Using no scheduler.", "yellow")
        scheduler = None

    # -----------------------
    #     Strat training
    # -----------------------
    min_test_loss = np.inf
    no_best_counter = 0

    for epoch in range(args.epochs):
        train_loss, test_loss = [], []

        # -----------------------
        #       Train step
        # -----------------------
        model.train()
        for batch in tqdm(train_loader, desc="Train"):
            X, subject_idxs, categories = *batch, *[None] * (3 - len(batch))
            X = X.to(device)

            X_recon = model(X, subject_idxs)

            loss = F.mse_loss(X_recon, X, reduction=args.reduction)
            train_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # -----------------------
        #       Test step
        # -----------------------
        model.eval()
        for batch in tqdm(test_loader, desc="Test"):
            X, subject_idxs, categories = *batch, *[None] * (3 - len(batch))
            X = X.to(device)

            with torch.no_grad():
                X_recon = model(X, subject_idxs)

                loss = F.mse_loss(X_recon, X, reduction=args.reduction)
                test_loss.append(loss.item())

        print(
            f"Epoch {epoch}/{args.epochs} | ",
            f"avg train MSE loss: {np.mean(train_loss):.3f} | ",
            f"avg test MSE loss: {np.mean(test_loss):.3f} | ",
            f"lr: {optimizer.param_groups[0]['lr']:.5f}",
        )

        if sweep:
            performance_now = {
                "train_recon_loss": np.mean(train_loss),
                "test_recon_loss": np.mean(test_loss),
                "lrate": optimizer.param_groups[0]["lr"],
            }
            wandb.log(performance_now)

        if scheduler is not None:
            scheduler.step()

        torch.save(model.state_dict(), os.path.join(run_dir, "autoencoder_last.pt"))

        # NOTE: This is mean over multiple ks.
        if np.mean(test_loss) < min_test_loss:
            cprint(f"New best. Saving models to {run_dir}", color="cyan")
            torch.save(model.state_dict(), os.path.join(run_dir, "autoencoder_best.pt"))

            min_test_loss = np.mean(test_loss)
            no_best_counter = 0
        else:
            no_best_counter += 1

        if no_best_counter > args.patience:
            cprint(f"Early stopping at epoch {epoch}", color="cyan")
            break


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def run(_args: DictConfig) -> None:
    global args, sweep

    # NOTE: Using default.yaml only for specifying the experiment settings yaml.
    args = OmegaConf.load(os.path.join("configs", _args.config_path))

    sweep = _args.sweep

    if sweep:
        sweep_config = OmegaConf.to_container(
            args.sweep_config, resolve=True, throw_on_missing=True
        )

        sweep_id = wandb.sweep(sweep_config, project=args.project_name)

        wandb.agent(sweep_id, train, count=args.sweep_count)

    else:
        train()


if __name__ == "__main__":
    run()
