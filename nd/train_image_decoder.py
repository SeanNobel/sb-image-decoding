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

from dalle2_pytorch import Unet, Decoder, DecoderTrainer

from nd.datasets import NeuroDiffusionCLIPEmbImageDataset, ThingsMEGDecoderDataset


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

    run_dir = os.path.join("runs/decoder", args.dataset.lower(), args.train_name)
    os.makedirs(run_dir, exist_ok=True)

    device = f"cuda:{args.cuda_id}"

    # -----------------------
    #       Dataloader
    # -----------------------
    if args.dataset == "ThingsMEG":
        dataset = ThingsMEGDecoderDataset(args)

        train_set = torch.utils.data.Subset(dataset, dataset.train_idxs)
        test_set = torch.utils.data.Subset(dataset, dataset.test_idxs)
    else:
        train_set = NeuroDiffusionCLIPEmbImageDataset(
            args.dataset, args.clip_train_name
        )
        test_set = NeuroDiffusionCLIPEmbImageDataset(
            args.dataset, args.clip_train_name, train=False
        )

    loader_args = {
        "drop_last": True,
        "num_workers": args.num_workers,
        "pin_memory": True,
    }
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=args.batch_size, shuffle=True, **loader_args
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=args.batch_size, shuffle=False, **loader_args
    )

    # ---------------------
    #        Models
    # ---------------------
    unet1 = Unet(
        dim=args.unet1.dim,
        image_embed_dim=args.embed_dim,
        cond_dim=128,
        channels=3,
        dim_mults=tuple(args.unet1.dim_mults),
    ).to(device)

    unet2 = Unet(
        dim=args.unet2.dim,
        image_embed_dim=args.embed_dim,
        cond_dim=128,
        channels=3,
        dim_mults=tuple(args.unet2.dim_mults),
    ).to(device)

    decoder = Decoder(
        unet=(unet1, unet2),
        image_sizes=tuple(args.image_sizes),
        timesteps=args.timesteps,
    ).to(device)

    # ---------------------
    #        Trainer
    # ---------------------
    trainer = DecoderTrainer(
        decoder,
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
        train_losses_unet1 = []
        train_losses_unet2 = []
        test_losses_unet1 = []
        test_losses_unet2 = []

        trainer.train()
        for Y_embed, Y in tqdm(train_loader):
            Y_embed, Y = Y_embed.to(device), Y.to(device)

            loss_unet1 = trainer(
                image_embed=Y_embed,
                image=Y,
                unet_number=1,
                max_batch_size=args.sub_batch_size,
            )
            trainer.update(1)

            loss_unet2 = trainer(
                image_embed=Y_embed,
                image=Y,
                unet_number=2,
                max_batch_size=args.sub_batch_size,
            )
            trainer.update(2)

            train_losses_unet1.append(loss_unet1)
            train_losses_unet2.append(loss_unet2)

        trainer.eval()
        for Y_embed, Y in test_loader:
            Y_embed, Y = Y_embed.to(device), Y.to(device)

            with torch.no_grad():
                loss_unet1 = trainer(image_embed=Y_embed, image=Y, unet_number=1)
                loss_unet2 = trainer(image_embed=Y_embed, image=Y, unet_number=2)

            test_losses_unet1.append(loss_unet1)
            test_losses_unet2.append(loss_unet2)

        print(
            f"Epoch {epoch}/{args.epochs} | ",
            f"avg train loss unet1: {np.mean(train_losses_unet1):.3f} | ",
            f"avg train loss unet2: {np.mean(train_losses_unet2):.3f} | ",
            f"avg test loss unet1: {np.mean(test_losses_unet1):.3f} | ",
            f"avg test loss unet2: {np.mean(test_losses_unet2):.3f} | ",
        )

        if args.use_wandb:
            performance_now = {
                "epoch": epoch,
                "train_loss_unet1": np.mean(train_losses_unet1),
                "train_loss_unet2": np.mean(train_losses_unet2),
                "test_loss_unet1": np.mean(test_losses_unet1),
                "test_loss_unet2": np.mean(test_losses_unet2),
            }
            wandb.log(performance_now)

        torch.save(decoder.state_dict(), os.path.join(run_dir, "decoder_last.pt"))

        test_loss = np.mean(test_losses_unet1) + np.mean(test_losses_unet2)
        if test_loss < min_test_loss:
            cprint(f"New best. Saving models to {run_dir}", color="cyan")
            torch.save(decoder.state_dict(), os.path.join(run_dir, "decoder_best.pt"))

            min_test_loss = test_loss


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def run(_args: DictConfig) -> None:
    # NOTE: Using default.yaml only for specifying the experiment settings yaml.
    args = OmegaConf.load(os.path.join("configs", _args.config_path))

    if _args.use_wandb is not None:
        with open_dict(args):
            args.use_wandb = _args.use_wandb

    train(args)


if __name__ == "__main__":
    run()
