import os, sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
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
from nd.models import (
    BrainEncoderBase,
    BrainEncoder,
    Wav2Vec2ConformerSpatialMixer,
    EEGNetDeep,
    ViT,
    ViViT,
    ViViTReduceTime,
    Unet3DEncoder,
    OpenFaceMapper,
    DiagonalClassifier,
    LabelClassifier,
    LatentsQuantizer,
    GumbelVectorQuantizer,
    GumbelVectorQuantizerV2,
    MLPTemporalReducer,
    MLP,
)
from nd.utils.layout import ch_locations_2d, DynamicChanLoc2d
from nd.utils.loss import (
    CLIPLoss,
    KLRegCLIPLoss,
    OrthoRegCLIPLoss,
    LargeEntropyCLIPLoss,
    AdaptiveCLIPLoss,
    AdditionalPositivesCLIPLoss,
    CosFaceCLIPLoss,
    ArcFaceCLIPLoss,
    AdaptiveMarginCLIPLoss,
    CircleCLIPLoss,
    GeometricCLIPLoss,
    CLIPWithClassCosFaceLoss,
    CLIPWithClassCircleLoss,
    NearestNeighborCLIPLoss,
)
from nd.utils.train_utils import Models, sequential_apply, count_parameters
from nd.utils.plots import plot_latents_2d, plot_2d_latents_with_sorted_categories


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
        "drop_last": False,
    }
    if split:
        # loader_args.update({"drop_last": True})

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
            # drop_last=False,
            **loader_args,
        )

        return dataloader, dataset


def build_models(args, dataset, device):
    subjects = dataset.subject_names if hasattr(dataset, "subject_names") else dataset.num_subjects  # fmt: skip

    if args.brain_encoder == "brain_encoder":
        brain_encoder = BrainEncoder(args, subjects=subjects).to(device)

    elif args.brain_encoder == "wav2vec2":
        brain_encoder = Wav2Vec2ConformerSpatialMixer(args, subjects=subjects).to(device)  # fmt: skip

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

    vision_encoder, preprocess = None, None

    if args.vision.pretrained:
        if isinstance(dataset, NeuroDiffusionCLIPDatasetBase):
            vision_encoder, preprocess = clip.load(args.vision.pretrained_model)
            vision_encoder = vision_encoder.eval().to(device)

        elif args.vision_quantize:
            assert args.vq is None, "Not expecting quantizing both vision and brain."  # fmt: skip

            vision_encoder = GumbelVectorQuantizer(
                args,
                in_tokens=args.orig_clip_tokens,
                out_tokens=args.num_clip_tokens,
                time_first=True,
            ).to(device)

        elif args.orig_clip_tokens != args.num_clip_tokens:
            vision_encoder = MLPTemporalReducer(
                args.orig_clip_tokens, args.num_clip_tokens
            ).to(device)

        elif args.F == 2:
            vision_encoder = MLP(args.orig_F, 2).to(device)
    else:
        vision_encoder = eval(args.vision.model)(**args.vision_encoder).to(device)

    return brain_encoder, vision_encoder, preprocess


def build_loss(args, dataset, device):
    if args.loss == "clip":
        loss_func = CLIPLoss(args).to(device)
    elif args.loss == "klclip":
        loss_func = KLRegCLIPLoss(args, alpha=args.klclip_alpha).to(device)
    elif args.loss == "orclip":
        loss_func = OrthoRegCLIPLoss(args, alpha=args.orclip_alpha).to(device)
    elif args.loss == "leclip":
        loss_func = LargeEntropyCLIPLoss(args, alpha=args.leclip_alpha).to(device)
    elif args.loss == "adaptiveclip":
        loss_func = AdaptiveCLIPLoss(args).to(device)
    elif args.loss == "apclip":
        loss_func = AdditionalPositivesCLIPLoss(args).to(device)
    elif args.loss == "cosfaceclip":
        loss_func = CosFaceCLIPLoss(args).to(device)
    elif args.loss == "arcfaceclip":
        loss_func = ArcFaceCLIPLoss(args).to(device)
    elif args.loss == "amclip":
        loss_func = AdaptiveMarginCLIPLoss(args).to(device)
    elif args.loss == "circleclip":
        loss_func = CircleCLIPLoss(args).to(device)
    elif args.loss == "geomclip":
        loss_func = GeometricCLIPLoss(args).to(device)
    elif args.loss == "nnclip":
        loss_func = NearestNeighborCLIPLoss(args).to(device)
    elif args.loss == "clipclasscosface":
        loss_func = CLIPWithClassCosFaceLoss(
            args,
            dataset.num_categories,
            dataset.num_high_categories if args.use_high_categories else None,
        ).to(device)
    elif args.loss == "clipclasscircle":
        loss_func = CLIPWithClassCircleLoss(
            args,
            dataset.num_categories,
            dataset.num_high_categories if args.use_high_categories else None,
        ).to(device)
    else:
        raise ValueError(f"Invalid loss function: {args.loss}")

    return loss_func


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
    loss_func = build_loss(args, dataset, device)

    # ---------------------
    #        Models
    # ---------------------
    brain_encoder, vision_encoder, preprocess = build_models(args, dataset, device)

    trained_models = Models(
        brain_encoder,
        vision_encoder if (vision_encoder is not None and vision_encoder.training) else None,  # fmt: skip
        loss_func,
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

    optimizer_vq = None
    if isinstance(vision_encoder, LatentsQuantizer):
        optimizer_vq = torch.optim.SGD(vision_encoder.parameters(), lr=args.vision_quantize_lr)  # fmt: skip

    # -----------------------
    #     Strat training
    # -----------------------
    vq_brain = args.vq is not None
    cprint(f"vq_brain: {vq_brain}", "yellow")

    max_test_acc = 0.0
    no_best_counter = 0

    for epoch in range(args.epochs):
        train_clip_losses = []
        train_mse_losses = []
        train_vq_losses = []
        train_adv_losses = []
        test_clip_losses = []
        test_mse_losses = []
        test_vq_losses = []
        test_adv_losses = []
        train_topk_accs = []
        test_topk_accs = []
        train_perplexities = []
        test_perplexities = []

        # For plotting latents
        train_Y_list = []
        train_Z_list = []
        train_categories_list = []
        test_Y_list = []
        test_Z_list = []
        test_categories_list = []

        # -----------------------
        #       Train step
        # -----------------------
        trained_models.train()
        for batch in tqdm(train_loader, desc="Train"):
            X, Y, subject_idxs, y_idxs, classes, high_categories = *batch, *[None] * (6 - len(batch))  # fmt: skip
            # ( b, t, d )

            if preprocess is not None:
                Y = sequential_apply(Y.numpy(), preprocess, batch_size=1)

            X, Y = X.to(device), Y.to(device)

            vq_loss, perplexity = None, None
            adv_loss = None

            if vision_encoder is None:
                pass
            elif isinstance(vision_encoder, clip.model.CLIP):
                with torch.no_grad():
                    Y = vision_encoder.encode_image(Y).float()
            elif isinstance(vision_encoder, GumbelVectorQuantizer):
                ret_dict = vision_encoder(Y)
                Y, perplexity = ret_dict["Z"], ret_dict["perplexity"]
                # vq_loss = ret_dict["div_loss"]
            else:
                Y = vision_encoder(Y)

            if isinstance(brain_encoder, BrainEncoderBase):
                ret_dict = brain_encoder(X, subject_idxs)

                Z, Z_mse = ret_dict["Z_clip"], ret_dict["Z_mse"]

                if isinstance(brain_encoder, BrainEncoder):
                    if vq_brain:
                        vq_loss, perplexity = ret_dict["vq_loss"], ret_dict["perplexity"]  # fmt: skip

                    if args.dann:
                        adv_loss = ret_dict["adv_loss"]

            elif isinstance(brain_encoder, EEGNetDeep):
                assert not vq_brain, "EEGNetDeep doesn't support vector quantization."

                Z, Z_mse = brain_encoder(X), None
            else:
                raise NotImplementedError

            if isinstance(loss_func, CLIPWithClassCosFaceLoss):
                if args.use_high_categories:
                    clip_loss = loss_func(Z, Y, classes, high_categories)
                else:
                    clip_loss = loss_func(Z, Y, classes)
            else:
                clip_loss = loss_func(Z, Y)

            if Z_mse is not None:
                mse_loss = F.mse_loss(
                    rearrange(Y, "b d t -> b (d t)"),
                    rearrange(Z_mse, "b d t -> b (d t)"),
                    reduction=args.reduction,
                )

                loss = args.lambd * clip_loss + (1 - args.lambd) * mse_loss
            else:
                loss = clip_loss

            with torch.no_grad():
                if isinstance(train_classifier, DiagonalClassifier):
                    topk_accs, _ = train_classifier(Z, Y)
                elif isinstance(train_classifier, LabelClassifier):
                    topk_accs = train_classifier(
                        Z, y_idxs.to(device), trained_models.vision_encoder
                    )
                else:
                    raise NotImplementedError

            train_clip_losses.append(clip_loss.item())
            train_topk_accs.append(topk_accs)

            if Z_mse is not None:
                train_mse_losses.append(mse_loss.item())

            if vq_loss is not None:
                loss = loss + vq_loss
                train_vq_losses.append(vq_loss.item())

            if perplexity is not None:
                train_perplexities.append(perplexity.item())

            if adv_loss is not None:
                loss = loss + adv_loss
                train_adv_losses.append(adv_loss.item())

            optimizer.zero_grad()
            if optimizer_vq is not None:
                optimizer_vq.zero_grad()

            loss.backward()

            optimizer.step()
            if optimizer_vq is not None:
                optimizer_vq.step()

            if args.plot_latents or args.F == 2:
                train_Y_list.append(Y.detach().cpu().numpy())
                if Z_mse is not None:
                    train_Z_list.append(Z_mse.detach().cpu().numpy())
                else:
                    train_Z_list.append(Z.detach().cpu().numpy())

                if high_categories is not None:
                    train_categories_list.append(high_categories.numpy())
                elif classes is not None:
                    train_categories_list.append(classes.numpy())
                else:
                    raise ValueError("plot_latents is True but no classes are given.")

            # if isinstance(vision_encoder, GumbelVectorQuantizer):
            #     vision_encoder.update_gumbel_temp()

        loss_func.clamp_params()

        _ = trained_models.params_updated()

        # -----------------------
        #       Test step
        # -----------------------
        trained_models.eval()
        for batch in tqdm(test_loader, desc="Test"):
            X, Y, subject_idxs, y_idxs, classes, high_categories = *batch, *[None] * (6 - len(batch))  # fmt: skip

            if preprocess is not None:
                Y = sequential_apply(Y.numpy(), preprocess, batch_size=1)

            X, Y = X.to(device), Y.to(device)

            vq_loss, perplexity = None, None
            adv_loss = None

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

                    if isinstance(vision_encoder, GumbelVectorQuantizer):
                        Y, perplexity = Y["Z"], Y["perplexity"]

                if not isinstance(brain_encoder, BrainEncoderBase):
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

                if isinstance(brain_encoder, BrainEncoderBase):
                    ret_dict = Z

                    Z, Z_mse = ret_dict["Z_clip"], ret_dict["Z_mse"]

                    if isinstance(brain_encoder, BrainEncoder):
                        if vq_brain:
                            vq_loss, perplexity = ret_dict["vq_loss"], ret_dict["perplexity"]  # fmt: skip

                        if args.dann:
                            adv_loss = ret_dict["adv_loss"]

                elif isinstance(brain_encoder, EEGNetDeep):
                    Z_mse = None
                else:
                    raise NotImplementedError

                if isinstance(loss_func, CLIPWithClassCosFaceLoss):
                    if args.use_high_categories:
                        clip_loss = loss_func(Z, Y, classes, high_categories)
                    else:
                        clip_loss = loss_func(Z, Y, classes)
                else:
                    clip_loss = loss_func(Z, Y)

                if Z_mse is not None:
                    mse_loss = F.mse_loss(
                        rearrange(Y, "b d t -> b (d t)"),
                        rearrange(Z_mse, "b d t -> b (d t)"),
                        reduction=args.reduction,
                    )

                if isinstance(test_classifier, DiagonalClassifier):
                    topk_accs, _ = test_classifier(
                        Z, Y, sequential=args.test_with_whole
                    )
                elif isinstance(test_classifier, LabelClassifier):
                    topk_accs = test_classifier(
                        Z, y_idxs.to(device), trained_models.vision_encoder, sequential=args.test_with_whole  # fmt: skip
                    )
                else:
                    raise NotImplementedError

            # TODO: Reuse the train step code.

            test_clip_losses.append(clip_loss.item())
            test_mse_losses.append(mse_loss.item())
            test_topk_accs.append(topk_accs)

            if vq_loss is not None:
                test_vq_losses.append(vq_loss.item())

            if perplexity is not None:
                test_perplexities.append(perplexity.item())

            if adv_loss is not None:
                test_adv_losses.append(adv_loss.item())

            if args.plot_latents or args.F == 2:
                test_Y_list.append(Y.detach().cpu().numpy())
                if Z_mse is not None:
                    test_Z_list.append(Z_mse.detach().cpu().numpy())
                else:
                    test_Z_list.append(Z.detach().cpu().numpy())

                if high_categories is not None:
                    test_categories_list.append(high_categories.numpy())
                elif classes is not None:
                    test_categories_list.append(classes.numpy())
                else:
                    raise ValueError("plot_latents is True but no classes are given.")

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
            # fmt: off
            if len(train_mse_losses) > 0:
                performance_now.update({"train_mse_loss": np.mean(train_mse_losses), "test_mse_loss": np.mean(test_mse_losses)})
            if len(train_vq_losses) > 0:
                performance_now.update({"train_vq_loss": np.mean(train_vq_losses), "test_vq_loss": np.mean(test_vq_losses)})
            if len(train_perplexities) > 0:
                performance_now.update({"train_perplexity": np.mean(train_perplexities),"test_perplexity": np.mean(test_perplexities)})
            if len(train_adv_losses) > 0:
                performance_now.update({"train_adv_loss": np.mean(train_adv_losses), "test_adv_loss": np.mean(test_adv_losses)})
            if hasattr(loss_func, "temp"):
                performance_now.update({"temp": loss_func.temp.item()})
            if hasattr(loss_func, "margin"):
                performance_now.update({"margin": loss_func.margin.item()})
            if hasattr(vision_encoder, "gumbel_temp"):
                performance_now.update({"gumbel_temp": vision_encoder.gumbel_temp})
            # FIXME: This doesn't work when args.blocks is a list of strings.
            if args.blocks == "transformer" and args.pos_enc == "sine_abs":
                performance_now.update({"pos_scale": brain_encoder.blocks[0].pos_enc.scale.item()})
            # fmt: on

            if args.F == 2:
                plots = plot_2d_latents_with_sorted_categories(
                    np.concatenate(train_Z_list),
                    np.concatenate(train_Y_list),
                    np.concatenate(train_categories_list),
                    np.concatenate(test_Z_list),
                    np.concatenate(test_Y_list),
                    np.concatenate(test_categories_list),
                )
                performance_now.update({"latents": wandb.Image(plots)})

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

            if args.plot_latents:
                best_train_Z = np.concatenate(train_Z_list)
                best_train_categories = np.concatenate(train_categories_list)
                best_test_Z = np.concatenate(test_Z_list)
                best_test_categories = np.concatenate(test_categories_list)
        else:
            no_best_counter += 1

        # if args.plot_latents and epoch == 0:
        #     plot_latents_2d(np.concatenate(train_Y_list), np.concatenate(train_categories_list), save_path=os.path.join(run_dir, f"plots/image_latents/train_epoch0.png"))  # fmt: skip
        #     plot_latents_2d(np.concatenate(test_Y_list), np.concatenate(test_categories_list), save_path=os.path.join(run_dir, f"plots/image_latents/test_epoch0.png"))  # fmt: skip

        if no_best_counter > args.patience:
            cprint(f"Early stopping at epoch {epoch}", color="cyan")
            break

    if args.plot_latents:
        plot_latents_2d(best_train_Z, best_train_categories, save_path=os.path.join(run_dir, "plots/brain_latents/best_train.png"))  # fmt: skip
        plot_latents_2d(best_test_Z, best_test_categories, save_path=os.path.join(run_dir, "plots/brain_latents/best_test.png"))  # fmt: skip
        plot_latents_2d(np.concatenate(train_Z_list), np.concatenate(train_categories_list), save_path=os.path.join(run_dir, "plots/brain_latents/last_train.png"))  # fmt: skip
        plot_latents_2d(np.concatenate(test_Z_list), np.concatenate(test_categories_list), save_path=os.path.join(run_dir, "plots/brain_latents/last_test.png"))  # fmt: skip


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
