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
    Brain2FaceUHDDataset,
    Brain2FaceStyleGANDataset,
    Brain2FaceYLabECoGDataset,
)
from brain2face.models.brain_encoder import BrainEncoder, BrainEncoderReduceTime
from brain2face.models.face_encoders import ViT, ViViT
from brain2face.utils.loss import CLIPLoss
from brain2face.utils.layout import ch_locations_2d


@torch.no_grad()
def eval(args: DictConfig):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_name = "".join(
        [k + "-" + str(v) + "_" for k, v in sorted(args.eval.__dict__.items())]
    )
    run_dir = os.path.join("runs", args.dataset.lower(), run_name)
    assert os.path.exists(run_dir), "run_dir doesn't exist."

    device = f"cuda:{args.cuda_id}"

    # -----------------------
    #       Dataloader
    # -----------------------
    """
    FIXME: I'm going to train DALLE2 prior & decoder with the train set of CLIP encoders training
        for now. This setting must be considered carefully later.
    """
    if args.split == "shallow":
        dataset = eval(f"Brain2Face{args.dataset}Dataset")(args)

        train_size = int(dataset.X.shape[0] * args.train_ratio)
        test_size = dataset.X.shape[0] - train_size
        train_set, _ = torch.utils.data.random_split(
            dataset,
            lengths=[train_size, test_size],
            generator=torch.Generator().manual_seed(args.seed),
        )

        num_subjects = dataset.num_subjects
    else:
        train_set = eval(f"Brain2Face{args.dataset}Dataset")(args)

        num_subjects = train_set.num_subjects

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_mamoery=True,
    )

    # ---------------------
    #        Models
    # ---------------------
    if args.face.type == "dynamic":
        brain_encoder = BrainEncoder(
            args, num_subjects=num_subjects, layout_fn=ch_locations_2d
        ).to(device)

        if args.face.encoded:
            face_encoder = None
        else:
            face_encoder = ViViT(
                num_frames=args.seq_len * args.fps, dim=args.F, **args.vivit
            ).to(device)

    elif args.face.type == "static":
        brain_encoder = BrainEncoderReduceTime(
            args, num_subjects=num_subjects, layout_fn=ch_locations_2d
        ).to(device)

        if args.face.encoded:
            face_encoder = None
        else:
            face_encoder = ViT(dim=args.F, **args.vit).to(device)

    brain_encoder.load_state_dict(torch.load(run_dir + "brain_encoder_best.pt"))
    brain_encoder.eval()

    if face_encoder is not None:
        face_encoder.load_state_dict(torch.load(run_dir + "face_encoder_best.pt"))
        face_encoder.eval()

    # -----------------------
    #       Evaluation
    # -----------------------
    i = 0
    for X, Y, subject_idxs in tqdm(train_loader):
        X, Y = X.to(device), Y.to(device)

        Z = brain_encoder(X, subject_idxs)

        if face_encoder is not None:
            Y = face_encoder(Y)

        for z, y in zip(Z, Y):
            np.save(z, os.path.join("data/clip_embds/uhd", f"{str(i).zfill(5)}.npy"))
            np.save(y, os.path.join("data/clip_embds/uhd", f"{str(i).zfill(5)}.npy"))

            i += 1


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def run(_args: DictConfig) -> None:
    args = OmegaConf.load(os.path.join("configs", _args.config_path))

    eval(args)


if __name__ == "__main__":
    run()
