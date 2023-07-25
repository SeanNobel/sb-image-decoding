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

from dalle2_pytorch import DiffusionPriorNetwork, DiffusionPrior, DiffusionPriorTrainer

from brain2face.datasets import NeuroDiffusionCLIPEmbDataset


@hydra.main(version_base=None, config_path="../configs", config_name="diffusion_prior")
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
        wandb.run.name = args.wandb.run_name
        wandb.run.save()

    # else:
    #     run_name = args.train_name

    # run_dir = os.path.join("runs", args.dataset.lower(), run_name)
    # os.makedirs(run_dir, exist_ok=True)

    device = f"cuda:{args.cuda_id}"

    # -----------------------
    #       Dataloader
    # -----------------------
    train_set = NeuroDiffusionCLIPEmbDataset(args.dataset)
    test_set = NeuroDiffusionCLIPEmbDataset(args.dataset, train=False)

    # train_size = int(dataset.Y.shape[0] * args.train_ratio)
    # test_size = dataset.Y.shape[0] - train_size
    # train_set, test_set = torch.utils.data.random_split(
    #     dataset,
    #     lengths=[train_size, test_size],
    #     generator=torch.Generator().manual_seed(args.seed),
    # )

    loader_args = {"drop_last": True, "num_workers": 4, "pin_memory": True}
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=args.batch_size, shuffle=True, **loader_args
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=args.batch_size, shuffle=True, **loader_args
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
    #      Optimizers
    # ---------------------
    diffusion_prior_trainer = DiffusionPriorTrainer(
        diffusion_prior,
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
    # min_test_loss = float("inf")

    for epoch in range(args.epochs):
        train_losses = []
        test_losses = []

        # diffusion_prior.train()
        for Z, Y in tqdm(train_loader):
            Z, Y = Z.to(device), Y.to(device)

            loss = diffusion_prior_trainer(
                text_embed=Z, image_embed=Y
            )  # , max_batch_size=4)

            train_losses.append(loss)
            # train_top10_accs.append(train_top10_acc)
            # train_top1_accs.append(train_top1_acc)

            diffusion_prior_trainer.update()

            # optimizer.zero_grad()

            # loss.backward()

            # optimizer.step()

        # assert models.params_updated()

        # diffusion_prior.eval()
        for Z, Y in test_loader:
            Z, Y = Z.to(device), Y.to(device)

            # with torch.no_grad():
            loss = diffusion_prior_trainer(
                text_embed=Z, image_embed=Y
            )  # , max_batch_size=4)

            test_losses.append(loss)
            # test_top10_accs.append(test_top10_acc)
            # test_top1_accs.append(test_top1_acc)

        print(
            f"Epoch {epoch}/{args.epochs} | ",
            f"avg train loss: {np.mean(train_losses):.3f} | ",
            f"avg test loss: {np.mean(test_losses):.3f} | ",
            # f"lr: {optimizer.param_groups[0]['lr']:.5f}",
        )

        if args.use_wandb:
            performance_now = {
                "epoch": epoch,
                "train_loss": np.mean(train_losses),
                "test_loss": np.mean(test_losses),
                # "lrate": optimizer.param_groups[0]["lr"],
            }
            wandb.log(performance_now)

        # if scheduler is not None:
        #     scheduler.step()

        # models.save(run_dir)

        # if np.mean(test_losses) < min_test_loss:
        #     cprint(f"New best. Saving models to {run_dir}", color="cyan")
        #     models.save(run_dir, best=True)

        #     min_test_loss = np.mean(test_losses)


# @hydra.main(version_base=None, config_path="../configs", config_name="default")
# def run(_args: DictConfig) -> None:
#     global args, sweep

#     # NOTE: Using default.yaml only for specifying the experiment settings yaml.
#     args = OmegaConf.load(os.path.join("configs", _args.config_path))

#     sweep = _args.sweep

#     if sweep:
#         sweep_config = OmegaConf.to_container(
#             args.sweep_config, resolve=True, throw_on_missing=True
#         )

#         sweep_id = wandb.sweep(sweep_config, project=args.project_name)

#         wandb.agent(sweep_id, train, count=args.sweep_count)

#     else:
#         train()


if __name__ == "__main__":
    # run()
    train()
