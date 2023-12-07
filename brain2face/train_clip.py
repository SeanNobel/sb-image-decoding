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

from brain2face.datasets.datasets import (
    YLabGODCLIPDataset,
    YLabE0030CLIPDataset,
    UHDCLIPDataset,
    StyleGANCLIPDataset,
    CollateFunctionForVideoHDF5,
    NeuroDiffusionCLIPDatasetBase,
)
from brain2face.datasets.things_meg import ThingsMEGCLIPDataset
from brain2face.models.brain_encoder import BrainEncoder
from brain2face.models.eeg_net import EEGNetDeep
from brain2face.models.vision_encoders import (
    ViT,
    ViViT,
    ViViTReduceTime,
    Unet3DEncoder,
    OpenFaceMapper,
)
from brain2face.models.classifier import DiagonalClassifier, LabelClassifier
from brain2face.utils.layout import ch_locations_2d, DynamicChanLoc2d
from brain2face.utils.loss import CLIPLoss
from brain2face.utils.train_utils import Models, sequential_apply, count_parameters
from brain2face.utils.plots import plot_latents_2d


def train():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_name = args.train_name

    if sweep:
        wandb.init(config=None)

        run_name += "_" + "".join(
            [k + "-" + str(v) + "_" for k, v in wandb.config.items()]
        )

        wandb.run.name = run_name
        args.__dict__.update(wandb.config)
        cprint(wandb.config, "cyan")

    run_dir = os.path.join("runs", args.dataset.lower(), run_name)
    os.makedirs(run_dir, exist_ok=True)

    device = f"cuda:{args.cuda_id}"

    # -----------------------
    #       Dataloader
    # -----------------------
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
        "drop_last": True,
        "num_workers": args.num_workers,
        "pin_memory": True,
    }
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=args.batch_size, shuffle=True, **loader_args
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=len(test_set) if args.test_with_whole else args.batch_size,
        shuffle=False,
        **loader_args,
    )

    # ---------------
    #      Loss
    # ---------------
    loss_func = CLIPLoss(args).to(device)

    # ---------------------
    #        Models
    # ---------------------
    if args.brain_encoder == "brain_encoder":
        brain_encoder = BrainEncoder(
            args,
            subject_names=dataset.subject_names,
            layout=eval(args.layout),
            vq=args.vq_brain,
            num_conv_blocks=args.num_conv_blocks,
            downsample=args.downsample,
            temporal_aggregation=args.temporal_aggregation,
        ).to(device)

    elif args.brain_encoder == "eegnet":
        brain_encoder = EEGNetDeep(args, duration=dataset.X.shape[-1]).to(device)

    if args.vision.pretrained:
        vision_encoder = dataset.clip_model
        preprocess = dataset.preprocess
    else:
        vision_encoder = eval(args.vision.model)(**args.vision_encoder).to(device)

    trained_models = Models(
        brain_encoder, vision_encoder if not args.vision.pretrained else None, loss_func
    )

    if sweep:
        wandb.config.update({"brain_encoder_params": count_parameters(brain_encoder)})

    # ---------------------
    #      Classifier
    # ---------------------
    if isinstance(dataset, NeuroDiffusionCLIPDatasetBase):
        train_classifier = test_classifier = DiagonalClassifier(args.acc_topk)

    elif isinstance(dataset, ThingsMEGCLIPDataset):
        # if args.large_test_set:
        #     classifier = DiagonalClassifier(args.acc_topk)
        # else:
        #     assert not args.test_with_whole, "No need to test with whole for ThingsMEG small test set."  # fmt: skip

        train_classifier = DiagonalClassifier(args.acc_topk)
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
        raise ValueError()

    # -----------------------
    #     Strat training
    # -----------------------
    min_test_loss = float("inf")

    for epoch in range(args.epochs):
        train_clip_losses = []
        train_vq_losses = []
        test_clip_losses = []
        test_vq_losses = []
        train_topk_accs = []
        test_topk_accs = []
        train_perplexities = []
        test_perplexities = []

        # For plotting latents
        train_Y_list = []
        train_Z_list = []
        train_classes_list = []

        trained_models.train()
        if args.accum_grad:
            optimizer.zero_grad()

        for batch in tqdm(train_loader, desc="Train"):
            if len(batch) == 5:
                X, Y, subject_idxs, classes, y_idxs = batch
            elif len(batch) == 4:
                X, Y, subject_idxs, classes, y_idxs = *batch, None
            else:
                X, Y, subject_idxs, classes, y_idxs = *batch, None, None

            if args.vision.pretrained and isinstance(
                dataset, NeuroDiffusionCLIPDatasetBase
            ):
                Y = sequential_apply(Y.numpy(), preprocess, batch_size=1)

            X, Y = X.to(device), Y.to(device)

            if args.vision.pretrained:
                with torch.no_grad():
                    Y = vision_encoder.encode_image(Y).float()
            else:
                Y = vision_encoder(Y)

            if args.vq_brain:
                assert isinstance(brain_encoder, BrainEncoder), "Please set vq_brain=False when it's not BrainEncoder."  # fmt: skip

                Z, vq_loss, perplexity = brain_encoder(X, subject_idxs)
                clip_loss = loss_func(Y, Z)

                loss = clip_loss + vq_loss
            else:
                if isinstance(brain_encoder, BrainEncoder):
                    Z = brain_encoder(X, subject_idxs)
                elif isinstance(brain_encoder, EEGNetDeep):
                    Z = brain_encoder(X)
                else:
                    raise NotImplementedError

                vq_loss, perplexity = None, None

                loss = clip_loss = loss_func(Y, Z)

            with torch.no_grad():
                if isinstance(train_classifier, DiagonalClassifier):
                    topk_accs, _ = train_classifier(Z, Y)
                elif isinstance(train_classifier, LabelClassifier):
                    topk_accs = train_classifier(Z, y_idxs.to(device))
                else:
                    raise NotImplementedError

            train_clip_losses.append(clip_loss.item())
            train_topk_accs.append(topk_accs)
            if args.vq_brain:
                train_vq_losses.append(vq_loss.item())
                train_perplexities.append(perplexity.item())

            if args.accum_grad:
                loss.backward()
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if classes is not None and args.plot_latents:
                train_Y_list.append(Y.detach().cpu().numpy())
                train_Z_list.append(Z.detach().cpu().numpy())
                train_classes_list.append(classes.numpy())

        if args.accum_grad:
            optimizer.step()

        loss_func.temp.data.clamp_(min=args.clip_temp_min, max=args.clip_temp_max)

        _ = trained_models.params_updated()

        trained_models.eval()
        for batch in tqdm(test_loader, desc="Test"):
            if len(batch) == 5:
                X, Y, subject_idxs, classes, y_idxs = batch
            elif len(batch) == 4:
                X, Y, subject_idxs, classes, y_idxs = *batch, None
            else:
                X, Y, subject_idxs, classes, y_idxs = *batch, None, None

            if args.vision.pretrained and isinstance(dataset, NeuroDiffusionCLIPDatasetBase):  # fmt: skip
                Y = sequential_apply(Y.numpy(), preprocess, batch_size=1)

            X, Y = X.to(device), Y.to(device)

            with torch.no_grad():
                if args.vision.pretrained:
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

                if args.vq_brain:
                    assert (not args.test_with_whole), "vq doesn't support test_with_whole for now"  # fmt: skip

                    Z, vq_loss, perplexity = brain_encoder(X, subject_idxs)
                else:
                    if not isinstance(brain_encoder, BrainEncoder):
                        subject_idxs = None

                    # NOTE: sequential_apply doesn't do sequential application if batch_size == X.shape[0].
                    Z = sequential_apply(
                        X,
                        brain_encoder,
                        args.batch_size,
                        subject_idxs=subject_idxs,
                        desc="BrainEncoder",
                    )

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
            test_topk_accs.append(topk_accs)
            if args.vq_brain:
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

            if args.vq_brain:
                performance_now.update(
                    {
                        "train_vq_loss": np.mean(train_vq_losses),
                        "test_vq_loss": np.mean(test_vq_losses),
                        "train_perplexity": np.mean(train_perplexities),
                        "test_perplexity": np.mean(test_perplexities),
                    }
                )

            wandb.log(performance_now)

        scheduler.step()

        trained_models.save(run_dir)

        if np.mean(test_clip_losses) < min_test_loss:
            cprint(f"New best. Saving models to {run_dir}", color="cyan")
            trained_models.save(run_dir, best=True)

            min_test_loss = np.mean(test_clip_losses)

        if len(train_classes_list) > 0:
            if epoch == 0:
                plot_latents_2d(
                    np.concatenate(train_Y_list),
                    np.concatenate(train_classes_list),
                    epoch=epoch,
                    save_dir=os.path.join(run_dir, "plots/image_latents"),
                )
            if epoch % 50 == 0:
                plot_latents_2d(
                    np.concatenate(train_Z_list),
                    np.concatenate(train_classes_list),
                    epoch=epoch,
                    save_dir=os.path.join(run_dir, "plots/ecog_latents"),
                )


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
