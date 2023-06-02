import os, sys
import numpy as np
import torch
import torch.nn as nn
from time import time
from tqdm import tqdm
from termcolor import cprint
import wandb
import hydra
from omegaconf import DictConfig

from brain2face.datasets import Brain2FaceStyleGANDataset
from brain2face.utils.layout import ch_locations_2d
from brain2face.models.brain_encoder import BrainEncoder, Classifier
from brain2face.utils.loss import CLIPLoss


@hydra.main(version_base=None, config_path="configs", config_name="config")
def eval(args: DictConfig):
    if args.seed is not None:
        print(f"Setting random seed: {args.seed}")
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    run_dir = f"runs/{args.train_name}/"
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)

    device = f"cuda:{args.cuda_id}"

    # -----------------------
    #       Dataloader
    # -----------------------
    if args.split == "shallow":
        dataset = Brain2FaceStyleGANDataset(args)

        train_size = int(dataset.X.shape[0] * args.train_ratio)
        test_size = dataset.X.shape[0] - train_size
        _, test_set = torch.utils.data.random_split(
            dataset,
            lengths=[train_size, test_size],
            generator=torch.Generator().manual_seed(args.seed),
        )

        num_subjects = dataset.num_subjects
    else:
        test_set = Brain2FaceStyleGANDataset(args, train=False)

        num_subjects = test_set.num_subjects
        test_size = test_set.X.shape[0]

    loader_args = {"drop_last": True, "num_workers": 4, "pin_memory": True}

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=test_size,  # args.batch_size,
        shuffle=False,
        **loader_args,
    )

    # ---------------------
    #        Models
    # ---------------------
    brain_encoder = BrainEncoder(
        args, num_subjects=num_subjects, layout_fn=ch_locations_2d
    ).to(device)
    brain_encoder.load_state_dict(torch.load(run_dir + "brain_encoder_best.pt"))
    brain_encoder.eval()

    classifier = Classifier(args)

    # ---------------
    #      Loss
    # ---------------
    loss_func = CLIPLoss(args).to(device)
    loss_func.eval()

    # -----------------------
    #       Evaluation
    # -----------------------
    min_test_loss = float("inf")

    test_losses = []
    test_top10_accs = []
    test_top1_accs = []
    inference_times = []

    with torch.no_grad():
        for X, Y, subject_idxs in tqdm(test_loader):
            X, Y = X.to(device), Y.to(device)

            Z = brain_encoder(X, subject_idxs)

            stime = time()
            inference_times.append(time() - stime)

            loss = loss_func(Y, Z)

            test_top1_acc, test_top10_acc, Y_pred = classifier(
                Z, Y, test=True, return_pred=True
            )

            test_losses.append(loss.item())
            test_top10_accs.append(test_top10_acc)
            test_top1_accs.append(test_top1_acc)

    # fmt: off
    Y_pred = torch.index_select(test_set.Y_all, dim=0, index=Y_pred)
    Y_pred = Y_pred.reshape(*Y_pred.shape[:2], args.fps*args.seq_len, -1).permute(0, 2, 3, 1)
    Y_pred = torch.split(Y_pred, test_set.session_lengths)
    torch.save(Y_pred, run_dir + "test_y_pred.pt")

    cprint(f"test top10: {test_top10_accs} | test top1: {test_top1_accs}", "green")


if __name__ == "__main__":
    eval()
