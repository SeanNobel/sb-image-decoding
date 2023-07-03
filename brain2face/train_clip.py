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

from brain2face.datasets import (
    Brain2FaceUHDDataset,
    Brain2FaceYLabECoGDataset,
    Brain2FaceStyleGANDataset,
)
from brain2face.models.brain_encoder import BrainEncoder, BrainEncoderReduceTime
from brain2face.models.face_encoders import ViT, ViViT, OpenFaceMapper
from brain2face.models.classifier import Classifier
from brain2face.utils.loss import CLIPLoss
from brain2face.utils.train_utils import Models, sequential_apply


def train():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if sweep:
        wandb.init(config=None)

        run_name = "".join([k + "-" + str(v) + "_" for k, v in wandb.config.items()])

        wandb.run.name = run_name
        args.__dict__.update(wandb.config)
        cprint(wandb.config, "cyan")

    else:
        run_name = args.train_name

    run_dir = os.path.join("runs", args.dataset.lower(), run_name)
    os.makedirs(run_dir, exist_ok=True)

    device = f"cuda:{args.cuda_id}"

    # -----------------------
    #       Dataloader
    # -----------------------
    if args.split == "shallow":
        dataset = eval(f"Brain2Face{args.dataset}Dataset")(args)

        train_size = int(dataset.X.shape[0] * args.train_ratio)
        test_size = dataset.X.shape[0] - train_size
        train_set, test_set = torch.utils.data.random_split(
            dataset,
            lengths=[train_size, test_size],
            generator=torch.Generator().manual_seed(args.seed),
        )

        num_subjects = dataset.num_subjects

    # NOTE: If not shallow, split is done inside dataset class
    else:
        train_set = eval(f"Brain2Face{args.dataset}Dataset")(args)
        test_set = eval(f"Brain2Face{args.dataset}Dataset")(args, train=False)

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

    # ---------------
    #      Loss
    # ---------------
    loss_func = CLIPLoss(args).to(device)

    # ---------------------
    #        Models
    # ---------------------
    if args.face.type == "dynamic":
        # FIXME: Temporarily other than YLab are not working.
        # brain_encoder = BrainEncoder(args, num_subjects=num_subjects).to(device)
        brain_encoder = BrainEncoderReduceTime(args, num_subjects=num_subjects, time_multiplier=3)

        if args.face.encoded:
            face_encoder = None
        else:
            face_encoder = eval(args.face.model)(
                out_channels=args.F, **args.face_encoder
            ).to(device)
            # FIXME: Temporarily other than YLab are not working.
            #     num_frames=args.seq_len * args.fps, dim=args.F, **args.vivit
            # ).to(device)

    elif args.face.type == "static":
        brain_encoder = BrainEncoderReduceTime(args, num_subjects=num_subjects).to(device)

        if args.face.encoded:
            face_encoder = None
        else:
            face_encoder = ViT(dim=args.F, **args.vit).to(device)

    else:
        raise ValueError("Face type is only static or dynamic.")

    classifier = Classifier(args)

    models = Models(brain_encoder, face_encoder, loss_func)

    # ---------------------
    #      Optimizers
    # ---------------------
    params = models.get_params()

    optimizer = torch.optim.Adam(params, lr=args.lr)

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

        models.train()
        for X, Y, subject_idxs in tqdm(train_loader):
            X, Y = X.to(device), Y.to(device)

            Z = brain_encoder(X, subject_idxs)

            if face_encoder is not None:
                Y = face_encoder(Y)

            loss = loss_func(Y, Z)

            with torch.no_grad():
                train_top1_acc, train_top10_acc, _ = classifier(Z, Y)

            train_losses.append(loss.item())
            train_top10_accs.append(train_top10_acc)
            train_top1_accs.append(train_top1_acc)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        assert models.params_updated()

        models.eval()
        for X, Y, subject_idxs in test_loader:
            X, Y = X.to(device), Y.to(device)

            with torch.no_grad():
                # NOTE: Avoid CUDA out of memory
                Z = sequential_apply(
                    X, brain_encoder, args.batch_size, subject_idxs=subject_idxs
                )

                if face_encoder is not None:
                    Y = sequential_apply(Y, face_encoder, args.batch_size)

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

        if sweep:
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

        models.save(run_dir)

        if np.mean(test_losses) < min_test_loss:
            cprint(f"New best. Saving models to {run_dir}", color="cyan")
            models.save(run_dir, best=True)

            min_test_loss = np.mean(test_losses)


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
