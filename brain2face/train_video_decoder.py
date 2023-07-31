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
from omegaconf import DictConfig, OmegaConf

from dalle2_video.dalle2_video import UnetTemporalConv, VideoDecoder
from dalle2_video.trainer import VideoDecoderTrainer

from brain2face.datasets import NeuroDiffusionCLIPEmbVideoDataset


def train() -> None:
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
        wandb.run.name = args.run_name
        wandb.run.save()

    run_dir = os.path.join("runs/decoder", args.dataset.lower(), args.type, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    device = f"cuda:{args.cuda_id}"

    # -----------------------
    #       Dataloader
    # -----------------------
    train_set = NeuroDiffusionCLIPEmbVideoDataset(args.dataset)
    test_set = NeuroDiffusionCLIPEmbVideoDataset(args.dataset, train=False)

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
    unet1 = UnetTemporalConv(
        dim=args.unet1.dim,
        image_embed_dim=args.image_embed_dim,
        channels=args.channels,
        dim_mults=tuple(args.unet1.dim_mults),
        cond_on_text_encodings=False,
    ).to(device)

    unet2 = UnetTemporalConv(
        dim=args.unet2.dim,
        image_embed_dim=args.image_embed_dim,
        channels=args.channels,
        dim_mults=tuple(args.unet2.dim_mults),
        cond_on_text_encodings=False,
    ).to(device)

    decoder = VideoDecoder(
        unet=(unet1, unet2),
        frame_sizes=tuple(args.frame_sizes),
        frame_numbers=tuple(args.frame_numbers),
        timesteps=args.timesteps,
    ).to(device)

    # ---------------------
    #        Trainer
    # ---------------------
    decoder_trainer = VideoDecoderTrainer(
        decoder,
        lr=args.lr,
        wd=args.wd,
        ema_beta=args.ema_beta,
        ema_update_after_step=args.ema_update_after_step,
        ema_update_every=args.ema_update_every,
    )

    # params = diffusion_prior.parameters()

    # optimizer = torch.optim.Adam(params, lr=args.lr)

    # if args.lr_scheduler == "cosine":
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #         optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    #     )
    # elif args.lr_scheduler == "multistep":
    #     mlstns = [int(m * args.epochs) for m in args.lr_multistep_mlstns]
    #     scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #         optimizer, milestones=mlstns, gamma=args.lr_step_gamma
    #     )
    # elif args.lr_scheduler == "none":
    #     scheduler = None
    # else:
    #     raise ValueError

    # -----------------------
    #     Strat training
    # -----------------------
    min_test_loss = float("inf")

    for epoch in range(args.epochs):
        train_losses_unet1 = []
        train_losses_unet2 = []
        test_losses_unet1 = []
        test_losses_unet2 = []

        # diffusion_prior.train()
        for Y_embed, Y in tqdm(train_loader):
            Y_embed, Y = Y_embed.to(device), Y.to(device)

            loss_unet1 = decoder_trainer(video_embed=Y_embed, video=Y, unet_number=1)
            # , max_batch_size=4)
            decoder_trainer.update(1)

            loss_unet2 = decoder_trainer(video_embed=Y_embed, video=Y, unet_number=2)
            # , max_batch_size=4)
            decoder_trainer.update(2)

            train_losses_unet1.append(loss_unet1)
            train_losses_unet2.append(loss_unet2)

            # optimizer.zero_grad()

            # loss.backward()

            # optimizer.step()

        # diffusion_prior.eval()
        for Y_embed, Y in test_loader:
            Y_embed, Y = Y_embed.to(device), Y.to(device)

            # with torch.no_grad():
            loss_unet1 = decoder_trainer(video_embed=Y_embed, video=Y, unet_number=1)
            # , max_batch_size=4)

            loss_unet2 = decoder_trainer(video_embed=Y_embed, image=Y, unet_number=2)

            test_losses_unet1.append(loss_unet1)
            test_losses_unet2.append(loss_unet2)

            # test_top10_accs.append(test_top10_acc)
            # test_top1_accs.append(test_top1_acc)

        print(
            f"Epoch {epoch}/{args.epochs} | ",
            f"avg train loss unet1: {np.mean(train_losses_unet1):.3f} | ",
            f"avg train loss unet2: {np.mean(train_losses_unet2):.3f} | ",
            f"avg test loss unet1: {np.mean(test_losses_unet1):.3f} | ",
            f"avg test loss unet2: {np.mean(test_losses_unet2):.3f} | ",
            # f"lr: {optimizer.param_groups[0]['lr']:.5f}",
        )

        if args.use_wandb:
            performance_now = {
                "epoch": epoch,
                "train_loss_unet1": np.mean(train_losses_unet1),
                "train_loss_unet2": np.mean(train_losses_unet2),
                "test_loss_unet1": np.mean(test_losses_unet1),
                "test_loss_unet2": np.mean(test_losses_unet2),
                # "lrate": optimizer.param_groups[0]["lr"],
            }
            wandb.log(performance_now)

        # if scheduler is not None:
        #     scheduler.step()

        torch.save(decoder.state_dict(), os.path.join(run_dir, "decoder_last.pt"))

        test_loss = np.mean(test_losses_unet1) + np.mean(test_losses_unet2)
        if test_loss < min_test_loss:
            cprint(f"New best. Saving models to {run_dir}", color="cyan")
            torch.save(decoder.state_dict(), os.path.join(run_dir, "decoder_best.pt"))

            min_test_loss = test_loss


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
