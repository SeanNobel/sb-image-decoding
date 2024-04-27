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
from typing import Union, Optional, List
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

import clip
from transformers import AutoProcessor, CLIPVisionModel

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
    BrainDecoder,
    # Wav2Vec2ConformerSpatialMixer,
    EEGNetDeep,
    ViT,
    ViViT,
    ViViTReduceTime,
    Unet3DEncoder,
    OpenFaceMapper,
    LatentsQuantizer,
    GumbelVectorQuantizer,
    GumbelVectorQuantizerV2,
    MLPTemporalReducer,
    MLP,
    SubspaceMapper,
)
from nd.utils.layout import ch_locations_2d, DynamicChanLoc2d
from nd.utils.loss import (
    build_clip,
    VariationalCLIPLoss,
    CLIPWithClassCosFaceLoss,
)
from nd.utils.loss import VariationalLowerBound
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
        brain_encoder = BrainEncoder(args, subjects).to(device)

    # elif args.brain_encoder == "wav2vec2":
    #     brain_encoder = Wav2Vec2ConformerSpatialMixer(args, subjects=subjects).to(device)  # fmt: skip

    elif args.brain_encoder == "eegnet":
        brain_encoder = EEGNetDeep(args, duration=dataset.X.shape[-1]).to(device)
    else:
        raise NotImplementedError

    brain_decoder = BrainDecoder(
        args.vae_dim, args.num_channels, int(args.seq_len * args.brain_resample_sfreq)
    ).to(device) if args.vae else None  # fmt: skip

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
            vision_encoder = MLPTemporalReducer(args.orig_clip_tokens, args.num_clip_tokens).to(device)  # fmt: skip

        elif args.loss == "subspaceclip":
            vision_encoder = SubspaceMapper(args.F, args.subspace_downs).to(device)

        elif args.F_mse != args.F:
            vision_encoder = MLP(args.F, args.F_mse).to(device)
    else:
        vision_encoder = eval(args.vision.model)(**args.vision_encoder).to(device)

    return brain_encoder, brain_decoder, vision_encoder, preprocess


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
    loss_func = build_clip(args, dataset, device)

    elbo_func = VariationalLowerBound(args, device) if args.vae else None

    # ---------------------
    #        Models
    # ---------------------
    brain_encoder, brain_decoder, vision_encoder, preprocess = build_models(args, dataset, device)  # fmt: skip

    trained_models = Models(
        brain_encoder,
        brain_decoder,
        vision_encoder if (vision_encoder is not None and vision_encoder.training) else None,  # fmt: skip
        loss_func,
    )

    if sweep:
        wandb.config.update({"brain_encoder_params": count_parameters(brain_encoder)})

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
    loss_temps_list = []  # For logging multiple-temperature losses

    for epoch in range(args.epochs):
        train_metrics = {"clip_loss": [], "mse_loss": [], "z_norms": [], "recon_loss": [], "kl_loss": [], "vq_loss": [], "adv_loss": [], "topk_accs": [], "perplexity": []}  # fmt: skip
        test_metrics = {"clip_loss": [], "mse_loss": [], "z_norms": [], "recon_loss": [], "kl_loss": [], "vq_loss": [], "adv_loss": [], "topk_accs": [], "perplexity": []}  # fmt: skip

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

            if preprocess is not None:
                Y = sequential_apply(Y.numpy(), preprocess, batch_size=1)

            X, Y = X.to(device), Y.to(device)

            # -----------------------
            #     Vision encoder
            # -----------------------
            Y_mse = None
            if vision_encoder is None:
                pass
            elif isinstance(vision_encoder, clip.model.CLIP):
                with torch.no_grad():
                    Y = vision_encoder.encode_image(Y).float()
            elif isinstance(vision_encoder, GumbelVectorQuantizer):
                ret_dict = vision_encoder(Y)
                Y, perplexity = ret_dict["Z"], ret_dict["perplexity"]
            elif isinstance(vision_encoder, MLP):
                Y_mse = vision_encoder(Y)
            else:
                Y = vision_encoder(Y)

            # -----------------------
            #     Brain encoder
            # -----------------------
            ret_dict = brain_encoder(X, subject_idxs)

            Z, Z_mse = ret_dict["Z_clip"], ret_dict["Z_mse"]

            q, Z_sample = None, None
            if "q" in ret_dict:
                q, Z_sample = ret_dict["q"], ret_dict["Z_sample"]

            # -----------------------
            #     Brain decoder
            # -----------------------
            if brain_decoder is not None:
                X_recon = brain_decoder(rearrange(Z_sample, "l b d -> (l b) d"))
                X_recon = rearrange(X_recon, "(l b) c t -> l b c t", b=X.shape[0])

            # -----------------------
            #       Loss step
            # -----------------------
            if isinstance(loss_func, CLIPWithClassCosFaceLoss):
                if args.use_high_categories:
                    clip_loss = loss_func(Z, Y, classes, high_categories)
                else:
                    clip_loss = loss_func(Z, Y, classes)
            elif isinstance(loss_func, VariationalCLIPLoss):
                assert q is not None, "You need the posterior for variational loss."

                clip_loss = loss_func(Z, Y, q)
            else:
                clip_loss = loss_func(Z, Y)

            mse_loss = F.mse_loss(
                rearrange(Z_mse, "b t d -> b (t d)"),
                Y_mse if Y_mse is not None else rearrange(Y, "b t d -> b (t d)"),
                reduction=args.reduction,
            )

            loss = args.lambd * clip_loss + (1 - args.lambd) * mse_loss

            recon_loss, kl_loss = None, None
            if brain_decoder is not None:
                elbo_loss, recon_loss, kl_loss = elbo_func(X_recon, X, q)
                loss += elbo_loss

            vq_loss, perplexity = None, None
            if "vq_loss" in ret_dict:
                vq_loss, perplexity = ret_dict["vq_loss"], ret_dict["perplexity"]  # fmt: skip
                loss += vq_loss

            adv_loss = None
            if "adv_loss" in ret_dict:
                adv_loss = ret_dict["adv_loss"]
                loss = loss + adv_loss

            optimizer.zero_grad()
            if optimizer_vq is not None:
                optimizer_vq.zero_grad()

            loss.backward()

            optimizer.step()
            if optimizer_vq is not None:
                optimizer_vq.step()

            loss_func.clamp_params()

            # -----------------------
            #        Accuracy
            # -----------------------
            topk_accs = loss_func.accuracy(Z, Y)

            # -----------------------
            #        Logging
            # -----------------------
            train_metrics["clip_loss"].append(clip_loss.item())
            train_metrics["mse_loss"].append(mse_loss.item())
            train_metrics["topk_accs"].append(topk_accs)
            train_metrics["z_norms"].append(Z.reshape(Z.shape[0], -1).norm(dim=-1).mean().item())

            if recon_loss is not None:
                train_metrics["recon_loss"].append(recon_loss.item())
                train_metrics["kl_loss"].append(kl_loss.item())

            if vq_loss is not None:
                train_metrics["vq_loss"].append(vq_loss.item())
                train_metrics["perplexity"].append(perplexity.item())

            if adv_loss is not None:
                train_metrics["adv_loss"].append(adv_loss.item())

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

            with torch.no_grad():
                # -----------------------
                #     Vision encoder
                # -----------------------
                Y_mse = None
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
                    ret = sequential_apply(Y, vision_encoder, args.batch_size, desc="VisionEncoder")

                    if isinstance(vision_encoder, GumbelVectorQuantizer):
                        Y, perplexity = ret["Z"], ret["perplexity"]
                    elif isinstance(vision_encoder, MLP):
                        Y_mse = ret

                # -----------------------
                #     Brain encoder
                # -----------------------
                if not isinstance(brain_encoder, BrainEncoderBase):
                    subject_idxs = None

                # NOTE: sequential_apply doesn't do sequential application if batch_size == X.shape[0].
                ret_dict = sequential_apply(
                    X,
                    brain_encoder,
                    args.batch_size,
                    subject_idxs=subject_idxs,
                    desc="BrainEncoder",
                    reduction=args.reduction,
                )

                Z, Z_mse = ret_dict["Z_clip"], ret_dict["Z_mse"]

                q, Z_sample = None, None
                if "q" in ret_dict:
                    q, Z_sample = ret_dict["q"], ret_dict["Z_sample"]

                # -----------------------
                #     Brain decoder
                # -----------------------
                if brain_decoder is not None:
                    X_recon = sequential_apply(
                        rearrange(Z_sample, "l b d -> (l b) d"),
                        brain_decoder,
                        args.batch_size,
                        desc="BrainDecoder",
                    )
                    X_recon = rearrange(X_recon, "(l b) c t -> l b c t", b=X.shape[0])

                # -----------------------
                #       Loss step
                # -----------------------
                if isinstance(loss_func, CLIPWithClassCosFaceLoss):
                    if args.use_high_categories:
                        clip_loss = loss_func(Z, Y, classes, high_categories)
                    else:
                        clip_loss = loss_func(Z, Y, classes)
                elif isinstance(loss_func, VariationalCLIPLoss):
                    clip_loss = loss_func(Z, Y, q)
                else:
                    clip_loss = loss_func(Z, Y)

                mse_loss = F.mse_loss(
                    rearrange(Z_mse, "b t d -> b (t d)"),
                    Y_mse if Y_mse is not None else rearrange(Y, "b t d -> b (t d)"),
                    reduction=args.reduction,
                )

                # -----------------------
                #     Classification
                # -----------------------
                if isinstance(dataset, ThingsMEGCLIPDataset):
                    topk_accs = loss_func.label_accuracy(
                        Z,
                        y_idxs.to(device),
                        None if isinstance(vision_encoder, MLP) else vision_encoder,
                        sequential=args.test_with_whole,
                    )
                else:
                    topk_accs = loss_func.accuracy(Z, Y, sequential=args.test_with_whole)

            # -----------------------
            #        Logging
            # -----------------------
            test_metrics["clip_loss"].append(clip_loss.item())
            test_metrics["mse_loss"].append(mse_loss.item())
            test_metrics["topk_accs"].append(topk_accs)
            test_metrics["z_norms"].append(Z.reshape(Z.shape[0], -1).norm(dim=-1).mean().item())

            if "vq_loss" in ret_dict:
                test_metrics["vq_loss"].append(ret_dict["vq_loss"].item())
            if "perplexity" in ret_dict:
                test_metrics["perplexity"].append(ret_dict["perplexity"].item())

            if "adv_loss" in ret_dict:
                test_metrics["adv_loss"].append(ret_dict["adv_loss"].item())

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

        train_topk_accs = np.stack(train_metrics["topk_accs"])
        test_topk_accs = np.stack(test_metrics["topk_accs"])

        print(
            f"Epoch {epoch}/{args.epochs} | ",
            f"avg train CLIP loss: {np.mean(train_metrics['clip_loss']):.3f} | ",
            f"avg test CLIP loss: {np.mean(test_metrics['clip_loss']):.3f} | ",
            f"lr: {optimizer.param_groups[0]['lr']:.5f}",
        )

        if sweep:
            performance_now = {
                "epoch": epoch,
                "train_clip_loss": np.mean(train_metrics["clip_loss"]),
                "test_clip_loss": np.mean(test_metrics["clip_loss"]),
                "train_mse_loss": np.mean(train_metrics["mse_loss"]),
                "test_mse_loss": np.mean(test_metrics["mse_loss"]),
                "lrate": optimizer.param_groups[0]["lr"],
                "train_z_norm": np.mean(train_metrics["z_norms"]),
                "test_z_norm": np.mean(test_metrics["z_norms"]),
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
            if len(train_metrics["recon_loss"]) > 0:
                performance_now.update({"train_recon_loss": np.mean(train_metrics["recon_loss"]), "test_recon_loss": np.mean(test_metrics["recon_loss"])})
                performance_now.update({"train_kl_loss": np.mean(train_metrics["kl_loss"]), "test_kl_loss": np.mean(test_metrics["kl_loss"])})
            if len(train_metrics["vq_loss"]) > 0:
                performance_now.update({"train_vq_loss": np.mean(train_metrics["vq_loss"]), "test_vq_loss": np.mean(test_metrics["vq_loss"])})
            if len(train_metrics["perplexity"]) > 0:
                performance_now.update({"train_perplexity": np.mean(train_metrics["perplexity"]), "test_perplexity": np.mean(test_metrics["perplexity"])})
            if len(train_metrics["adv_loss"]) > 0:
                performance_now.update({"train_adv_loss": np.mean(train_metrics["adv_loss"]), "test_adv_loss": np.mean(test_metrics["adv_loss"])})
            # fmt: on
            if hasattr(loss_func, "temp"):
                if len(loss_func.temp) == 1:
                    performance_now.update({"temp": loss_func.temp.item()})
                else:
                    loss_temps_list.append(loss_func.temp.detach().cpu().numpy())
                    loss_temps = np.stack(loss_temps_list).T  # ( n_temps, epoch )
                    performance_now.update(
                        {
                            "temp": wandb.plot.line_series(
                                xs=np.arange(epoch + 1),
                                ys=loss_temps,
                                keys=[f"temp_{i}" for i in range(len(loss_temps))],
                                title="Loss temperatures",
                                xname="Epoch",
                            )
                        }
                    )
            if hasattr(loss_func, "margin"):
                performance_now.update({"margin": loss_func.margin.item()})
            if hasattr(vision_encoder, "gumbel_temp"):
                performance_now.update({"gumbel_temp": vision_encoder.gumbel_temp})
            # FIXME: This doesn't work when args.blocks is a list of strings.
            if args.blocks == "transformer" and args.pos_enc == "sine_abs":
                performance_now.update({"pos_scale": brain_encoder.blocks[0].pos_enc.scale.item()})  # fmt: skip

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
