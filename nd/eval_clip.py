import os, sys
import numpy as np
import torch
import torch.nn as nn
from time import time
from tqdm import tqdm
from termcolor import cprint
import hydra
from omegaconf import DictConfig, OmegaConf

import clip

from nd.train_clip import build_dataloaders, build_models
from nd.datasets.datasets import (
    YLabGODCLIPDataset,
    YLabE0030CLIPDataset,
    UHDCLIPDataset,
    StyleGANCLIPDataset,
    CollateFunctionForVideoHDF5,
)
from nd.datasets.things_meg import ThingsMEGCLIPDataset
from nd.models.brain_encoder import BrainEncoder
from nd.models.vision_encoders import ViT, ViViT, ViViTReduceTime, Unet3DEncoder
from nd.utils.layout import ch_locations_2d, DynamicChanLoc2d
from nd.utils.train_utils import sequential_apply
from nd.utils.eval_utils import (
    VisionSaver,
    EmbeddingSaver,
    update_with_eval,
    get_run_dir,
)


@torch.no_grad()
def infer(args: DictConfig) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_dir = get_run_dir(args)
    cprint(f"Using model params in: {run_dir}", "cyan")

    save_dir = os.path.join(
        "data", "clip_embds", *run_dir.split("/")[1:], "unnormalized"
    )
    os.makedirs(save_dir, exist_ok=True)

    device = f"cuda:{args.cuda_id}"

    # -----------------------
    #       Dataloader
    # -----------------------
    dataloader, dataset = build_dataloaders(args, split=False)

    # ---------------------
    #        Models
    # ---------------------
    brain_encoder, vision_encoder, preprocess = build_models(args, dataset, device)

    brain_encoder.load_state_dict(
        torch.load(os.path.join(run_dir, "brain_encoder_best.pt"), map_location=device)
    )
    brain_encoder.eval()

    if not args.vision.pretrained:
        vision_encoder.load_state_dict(
            torch.load(
                os.path.join(run_dir, "vision_encoder_best.pt"), map_location=device
            )
        )
        vision_encoder.eval()

    # -----------------------
    #       Evaluation
    # -----------------------
    Z_list = []
    Z_mse_list = []
    Y_list = []
    vision_saver = VisionSaver(args, save_dir) if args.save_vision else None

    for batch in tqdm(dataloader, "Embedding whole dataset"):
        X, Y, subject_idxs, y_idxs, classes, high_categories = *batch, *[None] * (6 - len(batch))  # fmt: skip

        if vision_saver is not None:
            vision_saver.save(Y)

        if preprocess is not None:
            Y = sequential_apply(Y.numpy(), preprocess, batch_size=1)

        X, Y = X.to(device), Y.to(device)

        if vision_encoder is None:
            pass
        elif isinstance(vision_encoder, clip.model.CLIP):
            Y = vision_encoder.encode_image(Y).float()
        else:
            Y = vision_encoder(Y)

        Z = brain_encoder.encode(
            X, subject_idxs, normalize=False, swap_dims=True, return_mse=False
        )
        Z_mse = brain_encoder.encode(
            X, subject_idxs, normalize=False, swap_dims=True, return_mse=True
        )
        # Z, Z_mse, _, _ = *Z, *[None] * (4 - len(Z))

        assert Z.shape == Y.shape, f"Z.shape: {Z.shape}, Y.shape: {Y.shape}"

        # has_time = Z.ndim == 3
        # if has_time:
        #     b, d, t = Z.shape

        #     Z = Z.reshape(b, -1)
        #     Z_mse = Z_mse.reshape(b, -1)
        #     Y = Y.reshape(b, -1)
        # else:
        #     assert Z.ndim == 2, f"Z.ndim: {Z.ndim}"

        # Z /= Z.norm(dim=-1, keepdim=True)
        # Z_mse /= Z_mse.norm(dim=-1, keepdim=True)
        # Y /= Y.norm(dim=-1, keepdim=True)

        # if has_time:
        #     Z = Z.reshape(b, d, t)
        #     Z_mse = Z_mse.reshape(b, d, t)
        #     Y = Y.reshape(b, d, t)

        b = Z.shape[0]
        Z_list.append(Z.reshape(b, -1).cpu())
        Z_mse_list.append(Z_mse.reshape(b, -1).cpu())
        Y_list.append(Y.reshape(b, -1).cpu())

    torch.save(torch.cat(Z_list), os.path.join(save_dir, "brain_clip_embeds.pt"))
    torch.save(torch.cat(Z_mse_list), os.path.join(save_dir, "brain_mse_embeds.pt"))
    torch.save(torch.cat(Y_list), os.path.join(save_dir, "vision_embeds.pt"))

    if vision_saver is not None:
        vision_saver.close()


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def run(_args: DictConfig) -> None:
    args = OmegaConf.load(os.path.join("configs", _args.config_path))

    args = update_with_eval(args)
    # args.__dict__.update(args.eval)

    infer(args)


if __name__ == "__main__":
    run()

    # Copy of previous eval_clip.py

    # -----------------------
    #       Dataloader
    # -----------------------
    # if args.split in ["shallow", "mixed_shallow"]:
    #     dataset = eval(f"{args.dataset}CLIPDataset")(args)

    #     train_size = int(len(dataset.X) * args.train_ratio)
    #     test_size = len(dataset.X) - train_size
    #     train_set, test_set = torch.utils.data.random_split(
    #         dataset,
    #         lengths=[train_size, test_size],
    #         generator=torch.Generator().manual_seed(args.seed),
    #         # Must use the same seed as train
    #     )

    #     Y_ref = dataset.Y_ref

    #     subject_names = dataset.subject_names

    # else:
    #     train_set = eval(f"{args.dataset}CLIPDataset")(args)
    #     test_set = eval(f"{args.dataset}CLIPDataset")(args, train=False)

    #     test_size = len(test_set.X)

    #     assert len(train_set.Y_ref) == len(test_set.Y_ref), "train set Y_ref and test set Y_ref have different lengths."  # fmt: skip
    #     Y_ref = train_set.Y_ref

    #     subject_names = train_set.subject_names

    # if len(Y_ref) > 0:
    #     collate_fn = CollateFunctionForVideoHDF5(
    #         Y_ref,
    #         frame_size=args.vision_encoder.image_size,
    #     )
    # else:
    #     collate_fn = None

    # loader_args = {
    #     "collate_fn": collate_fn,
    #     "batch_size": args.batch_size,
    #     "shuffle": False,  # This must be False to keep consistency between image embds and image idxs.
    #     "drop_last": False,
    #     "num_workers": 4,
    #     "pin_memory": True,
    # }
    # train_loader = torch.utils.data.DataLoader(dataset=train_set, **loader_args)
    # test_loader = torch.utils.data.DataLoader(dataset=test_set, **loader_args)

    # ---------------------
    #        Models
    # ---------------------
    # Brain

    # if not args.reduce_time:
    #     brain_encoder = BrainEncoder(
    #         args,
    #         subject_names=subject_names,
    #         layout=eval(args.layout),
    #     ).to(device)

    # else:
    #     brain_encoder = BrainEncoderReduceTime(
    #         args,
    #         subject_names=subject_names,
    #         layout=eval(args.layout),
    #         time_multiplier=args.time_multiplier,
    #     ).to(device)

    # Vision

    # if args.vision.pretrained:
    #     vision_encoder, preprocess = clip.load(
    #         args.vision.pretrained_model, device=device
    #     )
    # else:
    #     vision_encoder = eval(args.vision.model)(**args.vision_encoder).to(device)

    #     vision_encoder.load_state_dict(
    #         torch.load(
    #             os.path.join(run_dir, "vision_encoder_best.pt"), map_location=device
    #         )
    #     )
    #     vision_encoder.eval()
