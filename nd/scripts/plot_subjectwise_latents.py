import os, sys
import random
import numpy as np
import torch
import torch.nn as nn
from time import time
import torch.utils
from tqdm import tqdm
from termcolor import cprint
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from nd.train_autoencoder import BrainAutoencoder
from nd.datasets import ImageNetEEGBrainDataset, ThingsMEGBrainDataset
from nd.utils.eval_utils import update_with_eval, get_run_dir
from nd.utils.plots import plot_latents_2d


@torch.no_grad()
def run(args: DictConfig) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_dir = get_run_dir(args)
    cprint(f"Using model params in: {run_dir}", "cyan")

    save_dir = os.path.join("figures/subjectwise_latents", run_dir.split("/")[-1])
    os.makedirs(save_dir, exist_ok=True)

    device = f"cuda:{args.cuda_id}"

    # -----------------------
    #       Dataloader
    # -----------------------
    dataset = eval(f"{args.dataset}Dataset")(args)

    # FIXME
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ---------------------
    #        Models
    # ---------------------
    subjects = dataset.subject_names if hasattr(dataset, "subject_names") else dataset.num_subjects  # fmt: skip

    model = BrainAutoencoder(args, subjects).to(device)

    model.load_state_dict(
        torch.load(os.path.join(run_dir, "autoencoder_best.pt"), map_location=device)
    )
    model.eval()

    # -----------------------
    #       Evaluation
    # -----------------------
    # for split, loader in zip(["train", "test"], [train_loader, test_loader]):
    Z_list = []
    # Z_mse_list = []
    subject_idxs_list = []

    for batch in tqdm(dataloader, "Embedding for the whole set."):
        X, subject_idxs, _, = *batch, *[None] * (3 - len(batch))  # fmt: skip
        X = X.to(device)

        Z = model.encode(X, subject_idxs)

        Z_list.append(Z.cpu().numpy())
        subject_idxs_list.append(subject_idxs.cpu().numpy())

    plot_latents_2d(
        np.concatenate(Z_list, axis=0),
        np.concatenate(subject_idxs_list, axis=0),
        save_path=os.path.join(save_dir, f"latents.png"),
    )


@hydra.main(version_base=None, config_path="../../configs", config_name="default")
def run_(args_: DictConfig) -> None:
    args = OmegaConf.load(os.path.join("configs", args_.config_path))

    args = update_with_eval(args)

    run(args)


if __name__ == "__main__":
    run_()
