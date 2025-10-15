import os, sys
import numpy as np
import torch
import torch.nn as nn
from time import time
from tqdm import tqdm
from termcolor import cprint
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import csv
import matplotlib.pyplot as plt

from sbid.models import LabelClassifier
from sbid.train_clip import build_dataloaders, build_models
from sbid.utils.loss import calc_similarity
from sbid.utils.eval_utils import update_with_eval, get_run_dir
from sbid.utils.plots import plot_latents_2d

sorted_high_categories = [
    "bird",
    "insect",
    "plant",
    "body part",
    "animal",
    "vegetable",
    "fruit",
    "drink",
    "dessert",
    "food",
    "clothing",
    "clothing accessory",
    "furniture",
    "home decor",
    "toy",
    "kitchen appliance",
    "kitchen tool",
    "office supply",
    "sports equipment",
    "medical equipment",
    "tool",
    "musical instrument",
    "electronic device",
    "weapon",
    "vehicle",
    "part of car",
    "container",
]

ALIGN_TO = "text"  # vision / text
ALIGN_TOKENS = "mean"  # cls / mean


@torch.no_grad()
def infer(args: DictConfig) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open_dict(args):
        args.align_to = ALIGN_TO
        args.align_tokens = ALIGN_TOKENS

    # -----------------------
    # Sort categories according to high-level categories
    # -----------------------
    hc_path = os.path.join(args.things_dir, "27 higher-level categories/category_mat_manual.tsv")  # fmt: skip
    high_categories = np.loadtxt(hc_path, dtype=int, delimiter="\t", skiprows=1)
    # ( 1854, 27 )
    print({k: v for k, v in zip(sorted_high_categories, high_categories.sum(axis=0))})

    unsorted_high_categories = np.loadtxt(hc_path, dtype=str, delimiter="\t")[0]

    arange = np.arange(len(high_categories))  # ( 1854, )
    argsort = []
    for shc in sorted_high_categories:
        idxs_in_hc = np.where(high_categories[:, unsorted_high_categories == shc] == 1)[0]  # fmt: skip
        for i in idxs_in_hc:
            if arange[i] != -1:
                argsort.append(i)
                arange[i] = -1

    # Appending images that don't belong to any high category
    for i in arange:
        if i != -1:
            argsort.append(i)

    # -----------------------
    #       Dataloader
    # -----------------------
    dataloader, dataset = build_dataloaders(args, split=False)

    # -----------------------
    #       Evaluation
    # -----------------------
    all_categories = np.loadtxt(
        os.path.join(args.things_dir, "things_concepts.tsv"),
        dtype=str,
        delimiter="\t",
        skiprows=1,
        usecols=0,
    )
    num_categories = len(all_categories)

    Y_list = []
    categories_list = []
    subject_idxs_list = []

    for batch in tqdm(dataloader, "Loading all samples."):
        _, Y, subject_idxs, _, categories, _ = *batch, *[None] * (6 - len(batch))

        Y_list.append(Y)
        categories_list.append(categories)
        subject_idxs_list.append(subject_idxs)

    subject_idxs = torch.cat(subject_idxs_list, dim=0)[: dataset.num_samples]
    assert torch.equal(subject_idxs, torch.zeros_like(subject_idxs))

    Y = torch.cat(Y_list, dim=0)[: dataset.num_samples]
    categories = torch.cat(categories_list, dim=0)[: dataset.num_samples].numpy()

    similarity = calc_similarity(Y, Y, sequential=True).numpy()  # ( 27048, 27048 )

    sim_cat = np.zeros((num_categories, num_categories))
    counts = np.zeros_like(sim_cat)

    for i in tqdm(range(similarity.shape[0]), "Creating similarity matrix for categories."):  # fmt: skip
        for j in range(similarity.shape[1]):
            # Skip catch samples
            if categories[i] == num_categories or categories[j] == num_categories:
                continue

            sim_cat[categories[i], categories[j]] += similarity[i, j]
            counts[categories[i], categories[j]] += 1

    sim_cat /= counts
    sim_cat = sim_cat[argsort][:, argsort]

    fig = plt.figure(figsize=(10, 10), tight_layout=True)
    fig.suptitle("Similarity between categories")
    plt.imshow(sim_cat, cmap="viridis")
    plt.colorbar()
    fig.savefig(f"out/{ALIGN_TO}_category_similarities.png")


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def run(_args: DictConfig) -> None:
    args = OmegaConf.load(os.path.join("configs", _args.config_path))

    args = update_with_eval(args)

    infer(args)


if __name__ == "__main__":
    run()
