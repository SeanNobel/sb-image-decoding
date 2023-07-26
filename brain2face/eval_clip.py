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

from brain2face.datasets import (
    YLabGODCLIPDataset,
    YLabE0030CLIPDataset,
    UHDCLIPDataset,
    StyleGANCLIPDataset,
)
from brain2face.models.brain_encoder import BrainEncoder, BrainEncoderReduceTime
from brain2face.models.face_encoders import ViT, ViViT
from brain2face.utils.layout import ch_locations_2d, DynamicChanLoc2d
from brain2face.utils.train_utils import sequential_apply
from brain2face.utils.eval_utils import (
    ImageSaver,
    EmbeddingSaver,
    update_with_eval,
    get_run_dir,
)


@torch.no_grad()
def infer(args: DictConfig) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_dir = get_run_dir(args)

    save_dir = os.path.join("data/clip_embds", args.dataset.lower())
    os.makedirs(save_dir, exist_ok=True)

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
            # Must use the same seed as train
        )

        subject_names = dataset.subject_names
    else:
        train_set = eval(f"{args.dataset}CLIPDataset")(args)
        test_set = eval(f"{args.dataset}CLIPDataset")(args, train=False)

        num_subjects = train_set.num_subjects
        test_size = len(test_set.X)

    loader_args = {
        "batch_size": args.batch_size,
        "shuffle": False,  # This must be False to keep consistency between image embds and image idxs.
        "drop_last": False,
        "num_workers": 4,
        "pin_memory": True,
    }
    train_loader = torch.utils.data.DataLoader(dataset=train_set, **loader_args)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, **loader_args)

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

    brain_encoder.load_state_dict(
        torch.load(os.path.join(run_dir, "brain_encoder_best.pt"), map_location=device)
    )
    brain_encoder.eval()

    if args.face.pretrained:
        face_encoder, preprocess = clip.load(args.face.pretrained_model, device=device)
    else:
        face_encoder = eval(args.face.model)(**args.face_encoder).to(device)

        face_encoder.load_state_dict(
            torch.load(os.path.join(run_dir, "face_encoder_best.pt"), map_location=device)
        )
        face_encoder.eval()

    # -----------------------
    #       Evaluation
    # -----------------------
    for mode in ["train", "test"]:
        Z_list = []
        Y_list = []
        image_saver = ImageSaver(
            os.path.join(save_dir, mode),
            args.for_webdataset,
            to_tensored=not args.face.pretrained,  # Whether the image is divided by 255
        )
        emb_saver = EmbeddingSaver(os.path.join(save_dir, mode), args.for_webdataset)

        for X, Y, subject_idxs in tqdm(eval(f"{mode}_loader"), f"Embedding {mode} set"):
            if args.reduce_time:
                image_saver.save(Y)

            if args.face.pretrained:
                Y = sequential_apply(Y.numpy(), preprocess, batch_size=1)

            X, Y = X.to(device), Y.to(device)

            Z = brain_encoder(X, subject_idxs)

            if args.face.pretrained:
                Y = face_encoder.encode_image(Y).float()
            else:
                Y = face_encoder(Y)

            Z /= Z.norm(dim=-1, keepdim=True)
            Y /= Y.norm(dim=-1, keepdim=True)

            Z_list.append(Z.cpu())
            Y_list.append(Y.cpu())

        emb_saver.save(torch.cat(Z_list), torch.cat(Y_list))


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def run(_args: DictConfig) -> None:
    args = OmegaConf.load(os.path.join("configs", _args.config_path))

    args = update_with_eval(args)
    # args.__dict__.update(args.eval)

    infer(args)


if __name__ == "__main__":
    run()
