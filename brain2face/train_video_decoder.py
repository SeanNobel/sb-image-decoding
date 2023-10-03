import os, sys
import numpy as np
import torch
import torch.nn as nn
from time import time
from tqdm import tqdm
from termcolor import cprint
from typing import Union, Optional
import logging
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from accelerate import Accelerator

from dalle2_video.dalle2_video import (
    Unet3D,
    VideoDecoder,
)


from brain2face.datasets import (
    NeuroDiffusionCLIPEmbVideoDataset,
    CollateFunctionForVideoHDF5,
)


def train(args: DictConfig) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    accelerator = Accelerator()
    accelerator.gradient_accumulation_steps = args.deepspeed.gradient_accumulation_steps

    device = accelerator.device

    if args.use_wandb and accelerator.is_main_process:
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

    run_dir = os.path.join(
        "runs/decoder", args.dataset.lower(), args.train_name
    )
    os.makedirs(run_dir, exist_ok=True)

    logging.getLogger().setLevel(eval(f"logging.{args.log_level}"))

    # -----------------------
    #       Dataloader
    # -----------------------
    resample_nsamples = args.frame_numbers[0] if args.temporal_emb else None

    train_set = NeuroDiffusionCLIPEmbVideoDataset(
        args.dataset, args.clip_train_name, resample_nsamples
    )
    test_set = NeuroDiffusionCLIPEmbVideoDataset(
        args.dataset, args.clip_train_name, resample_nsamples, train=False
    )

    loader_args = {
        "batch_size": args.batch_size,  # this will be overwritten by accelerator
        "drop_last": True,
        "num_workers": 4,
        "pin_memory": True,
    }
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        shuffle=True,
        collate_fn=CollateFunctionForVideoHDF5(train_set.Y_ref, resample_nsamples),
        **loader_args,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        shuffle=False,
        collate_fn=CollateFunctionForVideoHDF5(test_set.Y_ref, resample_nsamples),
        **loader_args,
    )

    # ---------------------
    #        Models
    # ---------------------
    unet1 = Unet3D(
        dim=args.unet1.dim,
        video_embed_dim=args.video_embed_dim,
        channels=args.channels,
        dim_mults=tuple(args.unet1.dim_mults),
        cond_on_text_encodings=False,
    ).to(device)

    unet2 = Unet3D(
        dim=args.unet2.dim,
        video_embed_dim=args.video_embed_dim,
        channels=args.channels,
        dim_mults=tuple(args.unet2.dim_mults),
        cond_on_text_encodings=False,
    ).to(device)

    decoder = VideoDecoder(
        unet=(unet1, unet2),
        frame_sizes=tuple(args.frame_sizes),
        frame_numbers=tuple(args.frame_numbers),
        timesteps=args.timesteps,
        learned_variance=False,
    ).to(device)

    # ---------------------
    #        Trainer
    # ---------------------
    # accelerator = Accelerator(
    #     deepspeed_plugin=
    # )

    decoder_trainer = VideoDecoderTrainer(
        decoder,
        accelerator=accelerator,
        dataloaders={
            "train": train_loader,
            "val": test_loader,
        },
        **args.decoder_trainer,
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

        for Y_embed, Y in tqdm(train_loader):
            Y_embed, Y = Y_embed.to(device), Y.to(device)

            loss_unet1 = decoder_trainer(video_embed=Y_embed, video=Y, unet_number=1)
            decoder_trainer.update(1)

            loss_unet2 = decoder_trainer(video_embed=Y_embed, video=Y, unet_number=2)
            decoder_trainer.update(2)

            train_losses_unet1.append(loss_unet1)
            train_losses_unet2.append(loss_unet2)

        for Y_embed, Y in tqdm(test_loader):
            Y_embed, Y = Y_embed.to(device), Y.to(device)

            loss_unet1 = decoder_trainer(video_embed=Y_embed, video=Y, unet_number=1)

            loss_unet2 = decoder_trainer(video_embed=Y_embed, video=Y, unet_number=2)

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
                "lrate_unet1": getattr(decoder_trainer, "optim0").param_groups[0]["lr"],
                "lrate_unet2": getattr(decoder_trainer, "optim1").param_groups[0]["lr"],
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

    with open_dict(args):
        args.use_wandb = _args.use_wandb

    train(args)


if __name__ == "__main__":
    run()
