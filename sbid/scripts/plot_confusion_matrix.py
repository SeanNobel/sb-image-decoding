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

from sbid.models import LabelClassifier
from sbid.train_clip import build_dataloaders, build_models
from sbid.utils.eval_utils import update_with_eval, get_run_dir
from sbid.utils.plots import plot_latents_2d


@torch.no_grad()
def infer(args: DictConfig) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_dir = get_run_dir(args)
    cprint(f"Using model params in: {run_dir}", "cyan")

    save_dir = os.path.join("out/confusion_matrix", run_dir.split("/")[-1])

    device = f"cuda:{args.cuda_id}"

    # -----------------------
    #       Dataloader
    # -----------------------
    with open_dict(args):
        args.test_with_whole = False

    _, test_loader, dataset = build_dataloaders(args, split=True)

    # ---------------------
    #        Models
    # ---------------------
    brain_encoder, _, _ = build_models(args, dataset, device)

    brain_encoder.load_state_dict(
        torch.load(os.path.join(run_dir, "brain_encoder_best.pt"), map_location=device)
    )
    brain_encoder.eval()

    classifier = LabelClassifier(dataset, args.acc_topk, device)

    # -----------------------
    #       Evaluation
    # -----------------------
    Z_list = []
    Y_list = []
    y_idxs_list = []
    categories_list = []

    all_categories = np.loadtxt(
        os.path.join(args.things_dir, "things_concepts.tsv"),
        dtype=str,
        delimiter="\t",
        skiprows=1,
        usecols=0,
    )
    print(all_categories)

    for batch in tqdm(test_loader, "Embedding test set samples."):
        X, Y, subject_idxs, y_idxs, categories, _ = *batch, *[None] * (6 - len(batch))

        Z = brain_encoder(X.to(device), subject_idxs)["Z_clip"]

        Z_list.append(Z)
        Y_list.append(Y)
        y_idxs_list.append(y_idxs.to(device))
        categories_list.append(categories)

    Z = torch.cat(Z_list, dim=0)
    y_idxs = torch.cat(y_idxs_list, dim=0)
    categories = torch.cat(categories_list, dim=0)

    similarity = classifier(Z, y_idxs, y_encoder=None, sequential=True, return_sim=True)
    # ( b, 200 )
    preds = torch.topk(similarity, args.acc_topk[-1], dim=1).indices  # ( b, k )
    pred_categories = classifier.categories[preds]  # ( b, k )

    count = 0
    for i, (pred, true) in enumerate(zip(pred_categories, categories)):
        pred = [all_categories[pred_] for pred_ in pred]
        true = all_categories[true]

        print(f"{true} -> {pred}")

        if true in pred:
            count += 1

    print(f"Accuracy: {count / len(categories)}")


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def run(_args: DictConfig) -> None:
    args = OmegaConf.load(os.path.join("configs", _args.config_path))

    args = update_with_eval(args)
    # args.__dict__.update(args.eval)

    infer(args)


if __name__ == "__main__":
    run()
