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

from sbid.train_clip import build_dataloaders
from sbid.models import BrainEncoder, BrainAutoencoder, BrainMAE
from sbid.datasets import ImageNetEEGBrainDataset, ImageNetEEGCLIPDataset
from sbid.utils.eval_utils import update_with_eval, get_run_dir
from sbid.utils.plots import plot_latents_2d

POSTFIX = "last"
PERPLEXITY = 50


@torch.no_grad()
def run(args: DictConfig) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_dir = get_run_dir(args)
    cprint(f"Using model params in: {run_dir}", "cyan")

    save_dir = os.path.join("figures", args.dataset.lower(), run_dir.split("/")[-1])
    os.makedirs(save_dir, exist_ok=True)

    device = f"cuda:{args.cuda_id}"

    # -----------------------
    #       Dataloader
    # -----------------------
    train_loader, test_loader, dataset = build_dataloaders(args)
    
    # dataset = eval(f"{args.dataset}Dataset")(args)
    # train_set = torch.utils.data.Subset(dataset, dataset.train_idxs)

    # dataloader = torch.utils.data.DataLoader(
    #     train_set,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=args.num_workers,
    #     pin_memory=True,
    # )

    # ---------------------
    #        Models
    # ---------------------
    subjects = dataset.subject_names if hasattr(dataset, "subject_names") else dataset.num_subjects  # fmt: skip

    if args.dataset.endswith("CLIP"):
        model = BrainEncoder(args, subjects).to(device)
    else:
        if args.masked:
            model = BrainMAE(args, subjects, mask_ratio=0).to(device)
        else:
            model = BrainAutoencoder(args, subjects).to(device)

    prefix = "brain_encoder" if args.dataset.endswith("CLIP") else "autoencoder"
    # FIXME!!!!!!!!!!!!!
    model.load_state_dict(
        torch.load(os.path.join("../../..", run_dir, f"{prefix}_{POSTFIX}.pt"), map_location=device)
    )
    model.eval()

    # -----------------------
    #       Evaluation
    # -----------------------
    Z_list = []
    subject_idxs_list = []

    for batch in tqdm(test_loader, "Embedding for the whole set."):
        X, subject_idxs, _, = *batch, *[None] * (3 - len(batch))  # fmt: skip
        # X = X.to(device)

        # Z = model.encode(X, subject_idxs, return_mse=False)

        # Z_list.append(Z.cpu().numpy())
        Z_list.append(X.numpy())
        subject_idxs_list.append(subject_idxs.cpu().numpy())
        
    Z = np.concatenate(Z_list, axis=0)
    Z = Z.reshape(Z.shape[0], -1)[::80]
    Z /= np.linalg.norm(Z, axis=-1, keepdims=True)
    sim = Z @ Z.T
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(sim)
    fig.savefig(os.path.join(save_dir, f"similarity_{POSTFIX}.png"), dpi=300)
    sys.exit()

    plot_latents_2d(
        np.concatenate(Z_list, axis=0),
        np.concatenate(subject_idxs_list, axis=0),
        perplexities=[PERPLEXITY],
        pca=False,
        save_path=os.path.join(save_dir, f"subjectwise_latents_{POSTFIX}.png"),
        off_axis=True,
    )


@hydra.main(config_path="../../configs", config_name="default")
def run_(args_: DictConfig) -> None:
    args = OmegaConf.load(os.path.join("../../..", "configs", args_.config_path))

    args = update_with_eval(args)

    run(args)


if __name__ == "__main__":
    run_()
