import os, sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
from tqdm import tqdm
from termcolor import cprint
from typing import Union, Optional
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

import clip
from transformers import AutoProcessor, CLIPVisionModel

from brainmagick.bm.models.simpleconv import SimpleConv

from nd.datasets.datasets import (
    YLabGODCLIPDataset,
    YLabE0030CLIPDataset,
    UHDCLIPDataset,
    StyleGANCLIPDataset,
    CollateFunctionForVideoHDF5,
    NeuroDiffusionCLIPDatasetBase,
)
from nd.datasets.things_meg import ThingsMEGCLIPDataset
from nd.models.brain_encoder import BrainEncoder
from nd.models.eeg_net import EEGNetDeep
from nd.models.vision_encoders import (
    ViT,
    ViViT,
    ViViTReduceTime,
    Unet3DEncoder,
    OpenFaceMapper,
)
from nd.models.classifier import DiagonalClassifier, LabelClassifier
from nd.utils.layout import ch_locations_2d, DynamicChanLoc2d
from nd.utils.loss import (
    CLIPLoss,
    NearestNeighborCLIPLoss,
    CosFaceCLIPLoss,
    CircleCLIPLoss,
)
from nd.utils.train_utils import Models, sequential_apply, count_parameters
from nd.utils.plots import plot_latents_2d


def build_dataloaders(args, split=True):
    dataset = eval(f"{args.dataset}CLIPDataset")(args)

    if isinstance(dataset, NeuroDiffusionCLIPDatasetBase):
        if args.split in ["shallow", "mixed_shallow"]:
            train_size = int(len(dataset.X) * args.train_ratio)
            test_size = len(dataset.X) - train_size
            train_set, test_set = torch.utils.data.random_split(
                dataset,
                lengths=[train_size, test_size],
                generator=torch.Generator().manual_seed(args.seed),
            )

        # NOTE: If not shallow, split is done inside dataset class
        else:
            train_set = dataset
            test_set = eval(f"{args.dataset}CLIPDataset")(args, train=False)

            assert len(dataset.Y_ref) == len(test_set.Y_ref), "train set Y_ref and test set Y_ref have different lengths."  # fmt: skip

        if len(dataset.Y_ref) > 0:
            collate_fn = CollateFunctionForVideoHDF5(
                dataset.Y_ref,
                # NOTE: Resampling in collate function is too costly.
                # resample_nsamples=args.vision.resample_nsamples,
                frame_size=args.vision_encoder.image_size,
            )
        else:
            collate_fn = None
    else:
        train_set = torch.utils.data.Subset(dataset, dataset.train_idxs)
        test_set = torch.utils.data.Subset(dataset, dataset.test_idxs)

        collate_fn = None

    loader_args = {
        "collate_fn": collate_fn,
        "num_workers": args.num_workers,
        "pin_memory": True,
    }
    if split:
        loader_args.update({"drop_last": True})

        train_loader = torch.utils.data.DataLoader(
            dataset=train_set, batch_size=args.batch_size, shuffle=True, **loader_args
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=len(test_set) if args.test_with_whole else args.batch_size,
            shuffle=False,
            **loader_args,
        )

        return train_loader, test_loader, dataset
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            **loader_args,
        )

        return dataloader, dataset


def build_models(args, dataset, device):
    if args.brain_encoder == "brain_encoder":
        brain_encoder = BrainEncoder(
            args,
            subject_names=dataset.subject_names,
            layout=eval(args.layout),
            vq=args.vq,
            blocks=args.blocks,
            downsample=args.downsample,
            temporal_aggregation=args.temporal_aggregation,
        ).to(device)

    elif args.brain_encoder == "eegnet":
        brain_encoder = EEGNetDeep(args, duration=dataset.X.shape[-1]).to(device)

    elif args.brain_encoder == "brainmagick":
        brain_encoder = SimpleConv(
            in_channels={"meg": args.num_channels},
            out_channels=args.F,
            n_subjects=len(dataset.subject_names),
            **args.simpleconv,
        )
    else:
        raise NotImplementedError

    if args.vision.pretrained:
        if isinstance(dataset, NeuroDiffusionCLIPDatasetBase):
            vision_encoder, preprocess = clip.load(args.vision.pretrained_model)
            vision_encoder = vision_encoder.eval().to(device)
        else:
            vision_encoder = None
            preprocess = None
    else:
        vision_encoder = eval(args.vision.model)(**args.vision_encoder).to(device)
        preprocess = None

    return brain_encoder, vision_encoder, preprocess


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

    run_dir = os.path.join("runs", args.dataset.lower(), run_name)
    os.makedirs(run_dir, exist_ok=True)

    device = f"cuda:{args.cuda_id}"

    # -----------------------
    #       Dataloader
    # -----------------------
    train_loader, test_loader, dataset = build_dataloaders(args)

    # ---------------
    #      Loss
    # ---------------
    if args.loss == "clip":
        loss_func = CLIPLoss(args).to(device)
    elif args.loss == "nnclip":
        loss_func = NearestNeighborCLIPLoss(args).to(device)
    elif args.loss == "cosfaceclip":
        loss_func = CosFaceCLIPLoss(
            args,
            dataset.num_categories,
            dataset.num_high_categories if args.use_high_categories else None,
        ).to(device)
    elif args.loss == "circleclip":
        loss_func = CircleCLIPLoss(
            args,
            dataset.num_categories,
            dataset.num_high_categories if args.use_high_categories else None,
        ).to(device)
    else:
        raise ValueError(f"Invalid loss function: {args.loss}")

    # ---------------------
    #        Models
    # ---------------------
    brain_encoder, vision_encoder, preprocess = build_models(args, dataset, device)

    trained_models = Models(
        brain_encoder, vision_encoder if not args.vision.pretrained else None, loss_func
    )

    if sweep:
        wandb.config.update({"brain_encoder_params": count_parameters(brain_encoder)})

    # ---------------------
    #      Classifier
    # ---------------------
    train_classifier = DiagonalClassifier(args.acc_topk)

    if isinstance(dataset, NeuroDiffusionCLIPDatasetBase):
        test_classifier = DiagonalClassifier(args.acc_topk)

    elif isinstance(dataset, ThingsMEGCLIPDataset):
        test_classifier = LabelClassifier(dataset, args.acc_topk, device)
    else:
        raise NotImplementedError

    # ---------------------
    #      Optimizers
    # ---------------------
    optimizer = torch.optim.Adam(trained_models.get_params(), lr=args.lr)

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
    vq_brain = args.vq is not None

    max_test_acc = 0.0
    no_best_counter = 0

    for epoch in range(args.epochs):
        train_clip_losses = []
        train_mse_losses = []
        train_vq_losses = []
        test_clip_losses = []
        test_mse_losses = []
        test_vq_losses = []
        train_topk_accs = []
        test_topk_accs = []
        train_perplexities = []
        test_perplexities = []

        # For plotting latents
        train_Y_list = []
        train_Z_list = []
        train_categories_list = []

        trained_models.train()
        if args.accum_grad:
            optimizer.zero_grad()

        for batch in tqdm(train_loader, desc="Train"):
            X, Y, subject_idxs, y_idxs, classes, high_categories = *batch, *[None] * (6 - len(batch))  # fmt: skip

            if preprocess is not None:
                Y = sequential_apply(Y.numpy(), preprocess, batch_size=1)

            X, Y = X.to(device), Y.to(device)

            if vision_encoder is None:
                pass
            elif isinstance(vision_encoder, clip.model.CLIP):
                with torch.no_grad():
                    Y = vision_encoder.encode_image(Y).float()
            else:
                Y = vision_encoder(Y)

            if isinstance(brain_encoder, BrainEncoder):
                if vq_brain:
                    Z, Z_mse, vq_loss, perplexity = brain_encoder(X, subject_idxs)
                else:
                    Z, Z_mse = brain_encoder(X, subject_idxs)

            elif isinstance(brain_encoder, EEGNetDeep):
                assert not vq_brain, "EEGNetDeep doesn't support vector quantization."

                Z, Z_mse = brain_encoder(X), None
            else:
                raise NotImplementedError

            if isinstance(loss_func, CosFaceCLIPLoss):
                if args.use_high_categories:
                    clip_loss = loss_func(Y, Z, classes, high_categories)
                else:
                    clip_loss = loss_func(Y, Z, classes)
            else:
                clip_loss = loss_func(Y, Z)

            if Z_mse is not None:
                mse_loss = F.mse_loss(Y, Z_mse, reduction=args.reduction)

                loss = args.lambd * clip_loss + (1 - args.lambd) * mse_loss
            else:
                loss = clip_loss

            if vq_brain and not args.vq_alternate:
                loss = loss + vq_loss

            with torch.no_grad():
                if isinstance(train_classifier, DiagonalClassifier):
                    topk_accs, _ = train_classifier(Z, Y)
                elif isinstance(train_classifier, LabelClassifier):
                    topk_accs = train_classifier(Z, y_idxs.to(device))
                else:
                    raise NotImplementedError

            train_clip_losses.append(clip_loss.item())
            train_topk_accs.append(topk_accs)

            if Z_mse is not None:
                train_mse_losses.append(mse_loss.item())

            if vq_brain:
                train_vq_losses.append(vq_loss.item())
                train_perplexities.append(perplexity.item())

            if args.accum_grad:
                loss.backward()
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if args.plot_latents:
                train_Y_list.append(Y.detach().cpu().numpy())
                train_Z_list.append(Z.detach().cpu().numpy())

                if high_categories is not None:
                    train_categories_list.append(high_categories.numpy())
                elif classes is not None:
                    train_categories_list.append(classes.numpy())
                else:
                    raise ValueError("plot_latents is True but no classes are given.")

        if args.accum_grad:
            optimizer.step()

        loss_func.clamp_params()

        _ = trained_models.params_updated()

        trained_models.eval()
        for batch in tqdm(test_loader, desc="Test"):
            X, Y, subject_idxs, y_idxs, classes, high_categories = *batch, *[None] * (6 - len(batch))  # fmt: skip

            if preprocess is not None:
                Y = sequential_apply(Y.numpy(), preprocess, batch_size=1)

            X, Y = X.to(device), Y.to(device)

            with torch.no_grad():
                if vision_encoder is None:
                    pass
                elif isinstance(vision_encoder, clip.model.CLIP):
                    Y = sequential_apply(
                        Y,
                        vision_encoder.encode_image,
                        args.batch_size,
                        desc="VisionEncoder (pretrained)",
                    ).float()
                else:
                    Y = sequential_apply(
                        Y, vision_encoder, args.batch_size, desc="VisionEncoder"
                    )

                if not isinstance(brain_encoder, BrainEncoder):
                    subject_idxs = None

                # NOTE: sequential_apply doesn't do sequential application if batch_size == X.shape[0].
                Z = sequential_apply(
                    X,
                    brain_encoder,
                    args.batch_size,
                    subject_idxs=subject_idxs,
                    desc="BrainEncoder",
                    reduction=args.reduction,
                )

                if isinstance(brain_encoder, BrainEncoder):
                    if vq_brain:
                        Z, Z_mse, vq_loss, perplexity = Z
                    else:
                        Z, Z_mse = Z
                elif isinstance(brain_encoder, EEGNetDeep):
                    Z_mse = None
                else:
                    raise NotImplementedError

                if isinstance(loss_func, CosFaceCLIPLoss):
                    if args.use_high_categories:
                        clip_loss = loss_func(Y, Z, classes, high_categories)
                    else:
                        clip_loss = loss_func(Y, Z, classes)
                else:
                    clip_loss = loss_func(Y, Z)

                if isinstance(test_classifier, DiagonalClassifier):
                    topk_accs, _ = test_classifier(
                        Z, Y, sequential=args.test_with_whole
                    )
                elif isinstance(test_classifier, LabelClassifier):
                    topk_accs = test_classifier(
                        Z, y_idxs.to(device), sequential=args.test_with_whole
                    )
                else:
                    raise NotImplementedError

            test_clip_losses.append(clip_loss.item())
            test_mse_losses.append(mse_loss.item())
            test_topk_accs.append(topk_accs)

            if Z_mse is not None:
                test_mse_losses.append(
                    F.mse_loss(Y, Z_mse, reduction=args.reduction).item()
                )
            if vq_brain:
                test_vq_losses.append(vq_loss.item())
                test_perplexities.append(perplexity.item())

        train_topk_accs = np.stack(train_topk_accs)
        test_topk_accs = np.stack(test_topk_accs)

        print(
            f"Epoch {epoch}/{args.epochs} | ",
            f"avg train CLIP loss: {np.mean(train_clip_losses):.3f} | ",
            f"avg test CLIP loss: {np.mean(test_clip_losses):.3f} | ",
            f"lr: {optimizer.param_groups[0]['lr']:.5f}",
        )

        if sweep:
            performance_now = {
                "epoch": epoch,
                "train_clip_loss": np.mean(train_clip_losses),
                "test_clip_loss": np.mean(test_clip_losses),
                "lrate": optimizer.param_groups[0]["lr"],
                "temp": loss_func.temp.item(),
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

            if len(train_mse_losses) > 0:
                assert len(test_mse_losses) > 0

                performance_now.update(
                    {
                        "train_mse_loss": np.mean(train_mse_losses),
                        "test_mse_loss": np.mean(test_mse_losses),
                    }
                )

            if vq_brain:
                performance_now.update(
                    {
                        "train_vq_loss": np.mean(train_vq_losses),
                        "test_vq_loss": np.mean(test_vq_losses),
                        "train_perplexity": np.mean(train_perplexities),
                        "test_perplexity": np.mean(test_perplexities),
                    }
                )

            if isinstance(loss_func, CosFaceCLIPLoss):
                performance_now.update({"margin": loss_func.margin.item()})

            wandb.log(performance_now)

        if scheduler is not None:
            scheduler.step()

        trained_models.save(run_dir)

        # NOTE: This is mean over multiple ks.
        if np.mean(test_topk_accs) > max_test_acc:
            cprint(f"New best. Saving models to {run_dir}", color="cyan")
            trained_models.save(run_dir, best=True)

            max_test_acc = np.mean(test_topk_accs)
            no_best_counter = 0
        else:
            no_best_counter += 1

        if len(train_categories_list) > 0:
            if epoch == 0:
                plot_latents_2d(
                    np.concatenate(train_Y_list),
                    np.concatenate(train_categories_list),
                    epoch=epoch,
                    save_dir=os.path.join(run_dir, "plots/image_latents"),
                )
            if epoch % 50 == 0:
                plot_latents_2d(
                    np.concatenate(train_Z_list),
                    np.concatenate(train_categories_list),
                    epoch=epoch,
                    save_dir=os.path.join(run_dir, "plots/ecog_latents"),
                )

        if no_best_counter > args.patience:
            cprint(f"Early stopping at epoch {epoch}", color="cyan")
            break


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
