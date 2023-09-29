import os, sys
import numpy as np
import torch
import torch.nn as nn
from time import time
from tqdm import tqdm
from termcolor import cprint
from typing import Union, Optional
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from dalle2_pytorch import DiffusionPriorNetwork, DiffusionPrior, DiffusionPriorTrainer

from brain2face.datasets import NeuroDiffusionCLIPEmbDataset


def train(args: DictConfig) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.use_wandb:
        wandb.config = {
            k: v for k, v in dict(args).items() if k not in ["root_dir", "wandb"]
        }
        wandb.init(
            project=args.wandb.project,
            entity=args.wandb.entity,
            config=wandb.config,
            save_code=True,
        )
        wandb.run.name = args.train_name
        wandb.run.save()

    run_dir = os.path.join("runs/prior", args.dataset.lower(), args.train_name)
    os.makedirs(run_dir, exist_ok=True)

    device = f"cuda:{args.cuda_id}"

    # -----------------------
    #       Dataloader
    # -----------------------
    train_set = NeuroDiffusionCLIPEmbDataset(args.dataset, args.clip_train_name)
    test_set = NeuroDiffusionCLIPEmbDataset(args.dataset, args.clip_train_name, train=False)  # fmt: skip

    loader_args = {"drop_last": True, "num_workers": 4, "pin_memory": True}
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=args.batch_size, shuffle=True, **loader_args
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=args.batch_size, shuffle=False, **loader_args
    )

    # ---------------------
    #        Models
    # ---------------------
    prior_network = DiffusionPriorNetwork(
        dim=args.dim,
        depth=args.depth,
        dim_head=args.dim_head,
        heads=args.heads,
    ).to(device)

    diffusion_prior = DiffusionPrior(
        net=prior_network,
        image_embed_dim=args.image_embed_dim,
        timesteps=args.timesteps,
        cond_drop_prob=args.cond_drop_prob,
        condition_on_text_encodings=False,
    ).to(device)

    # ---------------------
    #        Trainer
    # ---------------------
    diffusion_prior_trainer = DiffusionPriorTrainer(
        diffusion_prior,
        lr=args.lr,
        wd=args.wd,
        ema_beta=args.ema_beta,
        ema_update_after_step=args.ema_update_after_step,
        ema_update_every=args.ema_update_every,
    )

    # -----------------------
    #     Strat training
    # -----------------------
    min_test_loss = float("inf")

    for epoch in range(args.epochs):
        train_losses = []
        test_losses = []

        # diffusion_prior.train()
        for Z, Y in tqdm(train_loader):
            Z, Y = Z.to(device), Y.to(device)

            # NOTE: CLIP embeddings with temporal dimension
            if args.temporal_emb:
                Z = Z.permute(0, 2, 1).contiguous().view(-1, Z.shape[1])
                Y = Y.permute(0, 2, 1).contiguous().view(-1, Y.shape[1])

            loss = diffusion_prior_trainer(text_embed=Z, image_embed=Y)
            # , max_batch_size=4)

            train_losses.append(loss)

            diffusion_prior_trainer.update()

        # diffusion_prior.eval()
        for Z, Y in test_loader:
            Z, Y = Z.to(device), Y.to(device)

            if args.temporal_emb:
                Z = Z.permute(0, 2, 1).contiguous().view(-1, Z.shape[1])
                Y = Y.permute(0, 2, 1).contiguous().view(-1, Y.shape[1])

            loss = diffusion_prior_trainer(text_embed=Z, image_embed=Y)
            # , max_batch_size=4)

            test_losses.append(loss)

        print(
            f"Epoch {epoch}/{args.epochs} | ",
            f"avg train loss: {np.mean(train_losses):.3f} | ",
            f"avg test loss: {np.mean(test_losses):.3f} | ",
        )

        if args.use_wandb:
            performance_now = {
                "epoch": epoch,
                "train_loss": np.mean(train_losses),
                "test_loss": np.mean(test_losses),
                "lrate": diffusion_prior_trainer.optimizer.param_groups[0]["lr"],
            }
            wandb.log(performance_now)

        torch.save(diffusion_prior.state_dict(), os.path.join(run_dir, "prior_last.pt"))

        if np.mean(test_losses) < min_test_loss:
            cprint(f"New best. Saving models to {run_dir}", color="cyan")
            torch.save(
                diffusion_prior.state_dict(), os.path.join(run_dir, "prior_best.pt")
            )

            min_test_loss = np.mean(test_losses)


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def run(_args: DictConfig) -> None:
    # NOTE: Using default.yaml only for specifying the experiment settings yaml.
    args = OmegaConf.load(os.path.join("configs", _args.config_path))

    with open_dict(args):
        args.use_wandb = _args.use_wandb

    train(args)


if __name__ == "__main__":
    run()
    # train()
