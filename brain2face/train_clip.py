import os, sys
import numpy as np
import torch
import torch.nn as nn
from time import time
from tqdm import tqdm
from termcolor import cprint
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

from brain2face.datasets import Brain2FaceYLabECoGDataset, Brain2FaceStyleGANDataset
from brain2face.utils.layout import ch_locations_2d

from brain2face.models.brain_encoder import BrainEncoder
from brain2face.models.classifier import Classifier
from brain2face.utils.loss import CLIPLoss


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def train(args: DictConfig):
    # NOTE: Using default.yaml only for specifying the experiment settings yaml.
    args = OmegaConf.load(os.path.join("configs", args.config))

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    run_dir = f"runs/{args.train_name}/"
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)

    if args.use_wandb:
        wandb.config = {k: v for k, v in args.__dict__.items() if not k.startswith("__")}
        wandb.init(
            project=args.project_name,
            config=wandb.config,
            save_code=True,
        )
        wandb.run.name = args.train_name
        wandb.run.save()

    device = f"cuda:{args.cuda_id}"

    # -----------------------
    #       Dataloader
    # -----------------------
    if args.split == "shallow":
        dataset = eval(args.dataset)(args)

        train_size = int(dataset.X.shape[0] * args.train_ratio)
        test_size = dataset.X.shape[0] - train_size
        train_set, test_set = torch.utils.data.random_split(
            dataset,
            lengths=[train_size, test_size],
            generator=torch.Generator().manual_seed(args.seed),
        )

        num_subjects = dataset.num_subjects
    else:
        train_set = eval(args.dataset)(args)
        test_set = eval(args.dataset)(args, train=False)

        num_subjects = train_set.num_subjects
        test_size = test_set.X.shape[0]

    cprint(f"Test size: {test_size}", "cyan")

    loader_args = {"drop_last": True, "num_workers": 4, "pin_memory": True}
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=args.batch_size, shuffle=True, **loader_args
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=test_size, shuffle=True, **loader_args
    )

    # ---------------------
    #        Models
    # ---------------------
    brain_encoder = BrainEncoder(
        args, num_subjects=num_subjects, layout_fn=ch_locations_2d
    ).to(device)

    classifier = Classifier(args)

    # ---------------
    #      Loss
    # ---------------
    loss_func = CLIPLoss(args).to(device)

    # ---------------------
    #      Optimizers
    # ---------------------
    # NOTE: Brain optim optimizes loss temperature
    optimizer = torch.optim.Adam(
        list(brain_encoder.parameters()) + list(loss_func.parameters()), lr=args.lr
    )

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
        raise ValueError()

    # -----------------------
    #     Strat training
    # -----------------------
    min_test_loss = float("inf")

    for epoch in range(args.epochs):
        train_losses = []
        test_losses = []
        train_top10_accs = []
        train_top1_accs = []
        test_top10_accs = []
        test_top1_accs = []
        inference_times = []

        brain_encoder.train()
        loss_func.train()
        for X, Y, subject_idxs in tqdm(train_loader):
            X, Y = X.to(device), Y.to(device)

            Z = brain_encoder(X, subject_idxs)

            loss = loss_func(Y, Z)

            with torch.no_grad():
                train_top1_acc, train_top10_acc, _ = classifier(Z, Y)

            train_losses.append(loss.item())
            train_top10_accs.append(train_top10_acc)
            train_top1_accs.append(train_top1_acc)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        brain_encoder.eval()
        loss_func.eval()
        for X, Y, subject_idxs in test_loader:
            X, Y = X.to(device), Y.to(device)

            with torch.no_grad():
                # NOTE: Avoid CUDA out of memory
                Z = torch.cat(
                    [
                        brain_encoder(_X, _subject_idxs)
                        for _X, _subject_idxs in zip(
                            torch.split(X, args.batch_size),
                            torch.split(subject_idxs, args.batch_size),
                        )
                    ]
                )

                stime = time()
                inference_times.append(time() - stime)

                loss = loss_func(Y, Z)

                test_top1_acc, test_top10_acc, _ = classifier(Z, Y, sequential=True)

            test_losses.append(loss.item())
            test_top10_accs.append(test_top10_acc)
            test_top1_accs.append(test_top1_acc)

        print(
            f"Epoch {epoch}/{args.epochs} | ",
            f"avg train loss: {np.mean(train_losses):.3f} | ",
            f"avg test loss: {np.mean(test_losses):.3f} | ",
            f"lr: {optimizer.param_groups[0]['lr']:.5f}",
        )

        if args.use_wandb:
            performance_now = {
                "epoch": epoch,
                "train_loss": np.mean(train_losses),
                "test_loss": np.mean(test_losses),
                "train_top10_acc": np.mean(train_top10_accs),
                "train_top1_acc": np.mean(train_top1_accs),
                "test_top10_acc": np.mean(test_top10_accs),
                "test_top1_acc": np.mean(test_top1_accs),
                "lrate": optimizer.param_groups[0]["lr"],
                "temp": loss_func.temp.item(),
                "FaceEncoder avg inference time": np.mean(inference_times),
            }
            wandb.log(performance_now)

        scheduler.step()

        # Save models
        torch.save(brain_encoder.state_dict(), run_dir + "brain_encoder_last.pt")

        if np.mean(test_losses) < min_test_loss:
            cprint(f"New best. Saving models to {run_dir}", color="cyan")

            torch.save(brain_encoder.state_dict(), run_dir + "brain_encoder_best.pt")

            min_test_loss = np.mean(test_losses)


if __name__ == "__main__":
    train()
