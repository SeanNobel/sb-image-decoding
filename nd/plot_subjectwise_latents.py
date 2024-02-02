import os, sys
import numpy as np
import torch
import torch.nn as nn
from time import time
from tqdm import tqdm
from termcolor import cprint
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from nd.train_clip import build_dataloaders, build_models
from nd.utils.eval_utils import update_with_eval, get_run_dir
from nd.utils.plots import plot_latents_2d


@torch.no_grad()
def infer(args: DictConfig) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_dir = get_run_dir(args)
    cprint(f"Using model params in: {run_dir}", "cyan")

    save_dir = os.path.join("out/subjectwise_latents", run_dir.split("/")[-1])

    device = f"cuda:{args.cuda_id}"

    # -----------------------
    #       Dataloader
    # -----------------------
    with open_dict(args):
        args.test_with_whole = False

    train_loader, test_loader, dataset = build_dataloaders(args, split=True)

    # ---------------------
    #        Models
    # ---------------------
    brain_encoder, _, _ = build_models(args, dataset, device)

    brain_encoder.load_state_dict(
        torch.load(os.path.join(run_dir, "brain_encoder_best.pt"), map_location=device)
    )
    brain_encoder.eval()

    # -----------------------
    #       Evaluation
    # -----------------------
    for split, loader in zip(["train", "test"], [train_loader, test_loader]):
        Z_list = []
        Z_mse_list = []
        subject_idxs_list = []

        for batch in tqdm(loader, f"Embedding for {split} set."):
            X, _, subject_idxs, _, _, _ = *batch, *[None] * (6 - len(batch))

            X = X.to(device)

            ret_dict = brain_encoder(X, subject_idxs)
            Z, Z_mse = ret_dict["Z_clip"], ret_dict["Z_mse"]

            has_time = Z.ndim == 3
            if has_time:
                b, d, t = Z.shape

                Z = Z.reshape(b, -1)
                Z_mse = Z_mse.reshape(b, -1)
            else:
                assert Z.ndim == 2, f"Z.ndim: {Z.ndim}"

            Z /= Z.norm(dim=-1, keepdim=True)
            Z_mse /= Z_mse.norm(dim=-1, keepdim=True)

            if has_time:
                Z = Z.reshape(b, d, t)
                Z_mse = Z_mse.reshape(b, d, t)

            Z_list.append(Z.cpu().numpy())
            Z_mse_list.append(Z_mse.cpu().numpy())
            subject_idxs_list.append(subject_idxs.cpu().numpy())

        plot_latents_2d(
            np.concatenate(Z_list, axis=0),
            np.concatenate(subject_idxs_list, axis=0),
            save_path=os.path.join(save_dir, f"{split}_clip.png"),
        )
        plot_latents_2d(
            np.concatenate(Z_mse_list, axis=0),
            np.concatenate(subject_idxs_list, axis=0),
            save_path=os.path.join(save_dir, f"{split}_mse.png"),
        )


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def run(_args: DictConfig) -> None:
    args = OmegaConf.load(os.path.join("configs", _args.config_path))

    args = update_with_eval(args)
    # args.__dict__.update(args.eval)

    infer(args)


if __name__ == "__main__":
    run()
