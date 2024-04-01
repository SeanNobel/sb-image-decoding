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

import clip

from nd.datasets.things_text import ThingsTextCLIPDataset
from nd.utils.loss import build_clip
from nd.utils.plots import plot_latents_2d, plot_2d_latents_with_sorted_categories


# https://github.com/openai/CLIP/issues/57
def convert_models_to_fp32(model) -> None:
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()


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

    run_dir = os.path.join("runs/thingstext", run_name)
    os.makedirs(run_dir, exist_ok=True)

    device = f"cuda:{args.cuda_id}"

    # -----------------------
    #       Dataloader
    # -----------------------
    dataset = ThingsTextCLIPDataset(args)

    train_set = torch.utils.data.Subset(dataset, dataset.train_idxs)
    test_set = torch.utils.data.Subset(dataset, dataset.test_idxs)

    loader_args = {
        "num_workers": args.num_workers,
        "pin_memory": True,
        "drop_last": False,
    }

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=args.batch_size, shuffle=True, **loader_args
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=len(test_set) if args.test_with_whole else args.batch_size,
        shuffle=False,
        **loader_args,
    )

    # ---------------
    #      Loss
    # ---------------
    assert args.loss == "clip", "Not running text training with other loss functions."
    loss_func = build_clip(args, dataset, device)

    # ---------------
    #     Model
    # ---------------
    model, _ = clip.load(args.clip_model, device=device, jit=False)
    model.initialize_parameters()

    # ---------------------
    #      Optimizer
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
    max_test_acc = 0.0
    no_best_counter = 0

    for epoch in range(args.epochs):
        train_metrics = {"clip_loss": [], "mse_loss": [], "topk_accs": []}
        test_metrics = {"clip_loss": [], "mse_loss": [], "topk_accs": []}

        # For plotting latents
        train_Y_list = []
        train_Z_list = []
        train_categories_list = []
        test_Y_list = []
        test_Z_list = []
        test_categories_list = []

        # -----------------------
        #       Train step
        # -----------------------
        model.train()
        for X, Y, _, classes in tqdm(train_loader, desc="Train"):
            X, Y = X.to(device), Y.to(device)

            Z = model.encode_text(X).float()

            # -----------------------
            #       Loss step
            # -----------------------
            optimizer.zero_grad()

            clip_loss, mse_loss = loss_func(Z, Y), None

            if args.lambd == 1.0:
                clip_loss.backward()

            elif args.lambd < 1.0:
                mse_loss = F.mse_loss(
                    Z, rearrange(Y, "b t d -> b (t d)"), reduction=args.reduction
                )
                loss = args.lambd * clip_loss + (1 - args.lambd) * mse_loss

                loss.backward()

            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)

            loss_func.clamp_params()

            # -----------------------
            #        Accuracy
            # -----------------------
            topk_accs = loss_func.accuracy(Z, Y)

            # -----------------------
            #        Logging
            # -----------------------
            train_metrics["clip_loss"].append(clip_loss.item())
            if mse_loss is not None:
                train_metrics["mse_loss"].append(mse_loss.item())

            train_metrics["topk_accs"].append(topk_accs)

            if args.plot_latents or args.F == 2:
                train_Y_list.append(Y.detach().cpu().numpy())
                train_Z_list.append(Z.detach().cpu().numpy())

                if classes is not None:
                    train_categories_list.append(classes.numpy())

        # -----------------------
        #       Test step
        # -----------------------
        model.eval()
        for X, Y, y_idxs, classes in tqdm(test_loader, desc="Test"):
            X, Y, y_idxs = X.to(device), Y.to(device), y_idxs.to(device)

            with torch.no_grad():
                Z = model.encode_text(X).float()

                # -----------------------
                #       Loss step
                # -----------------------
                clip_loss, mse_loss = loss_func(Z, Y), None

                if args.lambd < 1.0:
                    mse_loss = F.mse_loss(
                        Z, rearrange(Y, "b t d -> b (t d)"), reduction=args.reduction
                    )

                # -----------------------
                #     Classification
                # -----------------------
                topk_accs = loss_func.label_accuracy(
                    Z, y_idxs, None, sequential=args.test_with_whole
                )

            # -----------------------
            #        Logging
            # -----------------------
            test_metrics["clip_loss"].append(clip_loss.item())
            if mse_loss is not None:
                test_metrics["mse_loss"].append(mse_loss.item())

            test_metrics["topk_accs"].append(topk_accs)

            if args.plot_latents or args.F == 2:
                test_Y_list.append(Y.detach().cpu().numpy())
                test_Z_list.append(Z.detach().cpu().numpy())

                if classes is not None:
                    test_categories_list.append(classes.numpy())

        train_topk_accs = np.stack(train_metrics["topk_accs"])
        test_topk_accs = np.stack(test_metrics["topk_accs"])

        print(
            f"Epoch {epoch}/{args.epochs} | ",
            f"avg train CLIP loss: {np.mean(train_metrics['clip_loss']):.3f} | ",
            f"avg test CLIP loss: {np.mean(test_metrics['clip_loss']):.3f} | ",
            f"lr: {optimizer.param_groups[0]['lr']:.5f}",
        )

        if sweep:
            performance_now = {
                "epoch": epoch,
                "train_clip_loss": np.mean(train_metrics["clip_loss"]),
                "test_clip_loss": np.mean(test_metrics["clip_loss"]),
                "lrate": optimizer.param_groups[0]["lr"],
            }

            performance_now.update(
                {
                    f"train_top{k}_acc": np.mean(train_topk_accs[:, i])
                    for i, k in enumerate(args.acc_topk)
                }
            )
            performance_now.update(
                {
                    f"test_top{k}_acc": np.mean(test_topk_accs[:, i])
                    for i, k in enumerate(args.acc_topk)
                }
            )

            if len(train_metrics["mse_loss"]) > 0:
                performance_now.update({"train_mse_loss": np.mean(train_metrics["mse_loss"]), "test_mse_loss": np.mean(test_metrics["mse_loss"])})  # fmt: skip

            if hasattr(loss_func, "temp"):
                performance_now.update({"temp": loss_func.temp.item()})

            if args.F == 2:
                plots = plot_2d_latents_with_sorted_categories(
                    np.concatenate(train_Z_list),
                    np.concatenate(train_Y_list),
                    np.concatenate(train_categories_list),
                    np.concatenate(test_Z_list),
                    np.concatenate(test_Y_list),
                    np.concatenate(test_categories_list),
                )
                performance_now.update({"latents": wandb.Image(plots)})

            wandb.log(performance_now)

        if scheduler is not None:
            scheduler.step()

        torch.save(model.state_dict(), os.path.join(run_dir, "clip_last.pt"))

        # NOTE: This is mean over multiple ks.
        if np.mean(test_topk_accs) > max_test_acc:
            cprint(f"New best. Saving models to {run_dir}", color="cyan")
            torch.save(model.state_dict(), os.path.join(run_dir, "clip_best.pt"))

            max_test_acc = np.mean(test_topk_accs)
            no_best_counter = 0

            if args.plot_latents:
                best_train_Z = np.concatenate(train_Z_list)
                best_train_categories = np.concatenate(train_categories_list)
                best_test_Z = np.concatenate(test_Z_list)
                best_test_categories = np.concatenate(test_categories_list)
        else:
            no_best_counter += 1

        # if args.plot_latents and epoch == 0:
        #     plot_latents_2d(np.concatenate(train_Y_list), np.concatenate(train_categories_list), save_path=os.path.join(run_dir, f"plots/image_latents/train_epoch0.png"))  # fmt: skip
        #     plot_latents_2d(np.concatenate(test_Y_list), np.concatenate(test_categories_list), save_path=os.path.join(run_dir, f"plots/image_latents/test_epoch0.png"))  # fmt: skip

        if no_best_counter > args.patience:
            cprint(f"Early stopping at epoch {epoch}", color="cyan")
            break

    if args.plot_latents:
        plot_latents_2d(best_train_Z, best_train_categories, save_path=os.path.join(run_dir, "plots/brain_latents/best_train.png"))  # fmt: skip
        plot_latents_2d(best_test_Z, best_test_categories, save_path=os.path.join(run_dir, "plots/brain_latents/best_test.png"))  # fmt: skip
        plot_latents_2d(np.concatenate(train_Z_list), np.concatenate(train_categories_list), save_path=os.path.join(run_dir, "plots/brain_latents/last_train.png"))  # fmt: skip
        plot_latents_2d(np.concatenate(test_Z_list), np.concatenate(test_categories_list), save_path=os.path.join(run_dir, "plots/brain_latents/last_test.png"))  # fmt: skip


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
