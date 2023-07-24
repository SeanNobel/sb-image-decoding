import os, sys
import numpy as np
import torch
import torch.nn as nn
from time import time
from tqdm import tqdm
from termcolor import cprint
import hydra
from omegaconf import DictConfig, OmegaConf

from brain2face.datasets import (
    YLabE0030CLIPDataset,
    YLabGODCLIPDataset,
    UHDCLIPDataset,
    StyleGANCLIPDataset,
)
from brain2face.models.brain_encoder import BrainEncoder, BrainEncoderReduceTime
from brain2face.models.face_encoders import ViT, ViViT
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

    loader_args = {"drop_last": False, "num_workers": 4, "pin_memory": True}
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=False,  # It must be False to keep consistency between face embds and face image idxs.
        **loader_args,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=test_size if args.test_with_whole else args.batch_size,
        shuffle=False,
        **loader_args,
    )

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

    if args.face.encoded:
        face_encoder = None
    else:
        face_encoder = eval(args.face.model)(**args.face_encoder).to(device)

    brain_encoder.load_state_dict(
        torch.load(os.path.join(run_dir, "brain_encoder_best.pt"), map_location=device)
    )
    brain_encoder.eval()

    if face_encoder is not None:
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
        image_saver = ImageSaver(os.path.join(save_dir, mode), args.for_webdataset)
        emb_saver = EmbeddingSaver(os.path.join(save_dir, mode), args.for_webdataset)

        for X, Y, subject_idxs in tqdm(eval(f"{mode}_loader")):
            X, Y = X.to(device), Y.to(device)

            if mode == "test" and args.test_with_whole:
                Z = sequential_apply(
                    X, brain_encoder, args.batch_size, subject_idxs=subject_idxs
                )

                if face_encoder is not None:
                    if args.face.type == "static":
                        image_saver.save(Y)

                    Y = sequential_apply(Y, face_encoder, args.batch_size)

            else:
                Z = brain_encoder(X, subject_idxs)

        if face_encoder is not None:
            if args.reduce_time:
                image_saver.save(Y)

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
