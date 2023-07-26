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

import clip

from brain2face.datasets import (
    YLabGODCLIPDataset,
    YLabE0030CLIPDataset,
    UHDCLIPDataset,
    StyleGANCLIPDataset,
)
from brain2face.models.brain_encoder import BrainEncoder, BrainEncoderReduceTime
from brain2face.models.face_encoders import ViT, ViViT, OpenFaceMapper
from brain2face.models.classifier import Classifier
from brain2face.utils.layout import ch_locations_2d, DynamicChanLoc2d
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
        dataset = eval(f"{args.dataset}CLIPDataset")(args)

        train_size = int(len(dataset.X) * args.train_ratio)
        test_size = len(dataset.X) - train_size
        train_set, test_set = torch.utils.data.random_split(
            dataset,
            lengths=[train_size, test_size],
            generator=torch.Generator().manual_seed(args.seed),
        )

        subject_names = dataset.subject_names

    # NOTE: If not shallow, split is done inside dataset class
    else:
        train_set = eval(f"{args.dataset}CLIPDataset")(args)
        test_set = eval(f"{args.dataset}CLIPDataset")(args, train=False)

        subject_names = train_set.subject_names
        test_size = len(test_set.X)

    cprint(f"Test size: {test_size}", "cyan")

    loader_args = {"drop_last": True, "num_workers": 4, "pin_memory": True}
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=args.batch_size, shuffle=True, **loader_args
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=test_size if args.test_with_whole else args.batch_size,
        shuffle=True,
        **loader_args,
    )

    # ---------------
    #      Loss
    # ---------------
    loss_func = CLIPLoss(args).to(device)

    # ---------------------
    #        Models
    # ---------------------
    if not args.reduce_time:
        brain_encoder = BrainEncoder(
            args,
            subject_names=subject_names,
            layout=eval(args.layout),
        ).to(device)

    else:
        brain_encoder = BrainEncoderReduceTime(
            args,
            subject_names=subject_names,
            layout=eval(args.layout),
            time_multiplier=args.time_multiplier,
        ).to(device)

    if args.vision.pretrained:
        vision_encoder, preprocess = clip.load(
            args.vision.pretrained_model, device=device
        )
    else:
        vision_encoder = eval(args.vision.model)(**args.vision_encoder).to(device)

    classifier = Classifier(args)

    models = Models(brain_encoder, vision_encoder, loss_func)

    # ---------------------
    #      Optimizers
    # ---------------------
    optimizer = torch.optim.Adam(models.get_params(), lr=args.lr)

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
        if args.accum_grad:
            optimizer.zero_grad()

        for X, Y, subject_idxs in tqdm(train_loader):
            if args.vision.pretrained:
                Y = sequential_apply(Y.numpy(), preprocess, batch_size=1)

            X, Y = X.to(device), Y.to(device)

            Z = brain_encoder(X, subject_idxs)

            if args.vision.pretrained:
                Y = vision_encoder.encode_image(Y).float()
            else:
                Y = vision_encoder(Y)

            loss = loss_func(Y, Z)

            with torch.no_grad():
                train_top1_acc, train_top10_acc, _ = classifier(Z, Y)

            train_losses.append(loss.item())
            train_top10_accs.append(train_top10_acc)
            train_top1_accs.append(train_top1_acc)

            if args.accum_grad:
                loss.backward()
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if args.accum_grad:
            optimizer.step()

        _ = models.params_updated()

        models.eval()
        for X, Y, subject_idxs in test_loader:
            if args.vision.pretrained:
                Y = sequential_apply(Y.numpy(), preprocess, batch_size=1)

            X, Y = X.to(device), Y.to(device)

            with torch.no_grad():
                stime = time()

                # NOTE: sequential_apply doesn't do sequential application if batch_size == X.shape[0].

                Z = sequential_apply(
                    X, brain_encoder, args.batch_size, subject_idxs=subject_idxs
                )

                if args.vision.pretrained:
                    Y = sequential_apply(
                        Y, vision_encoder.encode_image, args.batch_size
                    ).float()
                else:
                    Y = sequential_apply(Y, vision_encoder, args.batch_size)

                inference_times.append(time() - stime)

                loss = loss_func(Y, Z)

                test_top1_acc, test_top10_acc, _ = classifier(
                    Z, Y, sequential=args.test_with_whole
                )

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
                "VisionEncoder avg inference time": np.mean(inference_times),
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
