import os, sys
import torch
import numpy as np
import libs.autoencoder
import libs.clip
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from termcolor import cprint

import clip

from datasets import ThingsMEGDatabase

from nd.models.brain_encoder import BrainEncoder
from nd.utils.layout import ch_locations_2d
from nd.utils.eval_utils import get_run_dir

TO_REPORTED_STD = True


@torch.no_grad()
@hydra.main(
    version_base=None, config_path="../../configs/thingsmeg", config_name="clip"
)
def main(args):
    device = "cuda"
    save_root = os.path.join(args.root, "data/uvit/thingsmeg", get_run_dir(args).split("/")[-1])  # fmt: skip
    os.makedirs(save_root, exist_ok=True)

    # -----------------
    #      Dataset
    # -----------------
    dataset = ThingsMEGDatabase(args)
    train_set = torch.utils.data.Subset(dataset, dataset.train_idxs)
    test_set = torch.utils.data.Subset(dataset, dataset.test_idxs)

    # Saving filenames
    np.save(os.path.join(save_root, "train_filenames.npy"), dataset.Y_paths[dataset.train_idxs])  # fmt: skip
    np.save(os.path.join(save_root, "test_filenames.npy"), dataset.Y_paths[dataset.test_idxs])  # fmt: skip

    # -----------------
    #       Models
    # -----------------
    # Stable Diffusion
    autoencoder = libs.autoencoder.get_model(
        "assets/stable-diffusion/autoencoder_kl.pth", scale_factor=0.23010
    )
    autoencoder.to(device)

    # CLIP-MEG
    subjects = dataset.subject_names if hasattr(dataset, "subject_names") else dataset.num_subjects  # fmt: skip
    brain_encoder = BrainEncoder(args, subjects=subjects).to(device)

    weights_path = os.path.join(args.root, get_run_dir(args), "brain_encoder_best.pt")
    cprint(f"Loading weights from {weights_path}", "cyan")
    brain_encoder.load_state_dict(torch.load(weights_path, map_location=device))
    brain_encoder.eval()

    # CLIP-Vision ViT-B/32
    # clip_model, preprocess = clip.load("ViT-B/32")
    # clip_model = clip_model.eval().to(device)

    # -----------------
    #   Empty context
    # -----------------
    for source in ["zeros", "randn"]:
        X_null = eval(f"torch.{source}_like")(dataset.X[0], device=device)
        Z_null = brain_encoder.encode(
            X_null,
            torch.arange(len(X_null), device=device),
            normalize=False,
            swap_dims=True,
        )
        # ( 4, 768 )
        Z_null = Z_null.mean(dim=0).detach().cpu().numpy()
        # ( 1, 768 )
        np.save(os.path.join(save_root, f"empty_context_from_{source}.npy"), Z_null)

    # -----------------
    # Extract Embeddings
    # -----------------
    for split, datas in zip(["train", "test"], [train_set, test_set]):
        save_dir = os.path.join(save_root, split)
        os.makedirs(save_dir, exist_ok=True)

        for idx, (X, image, subject_idxs) in tqdm(enumerate(datas), total=len(datas)):
            """
            NOTE: We don't need normalization for OpenAI pretrained CLIP. MEG embeddings are
            normalized within encode method.
            NOTE: There are 5 captions per image in MSCOCO, while we have 4 subjects per image in THINGS-MEG.

            X ( subjects=4, channels=271, timesteps=169 ): MEG sample from all subjects for the image
            image ( c=3, h=256, w=256 ): One image sample
            subject_idxs ( subjects=4, ): torch.arange(4)
            """
            moment = autoencoder.encode_moments(image.to(device).unsqueeze(0)).squeeze(0)  # fmt: skip
            # ( 8, 32, 32 )
            np.save(os.path.join(save_dir, f"{idx}.npy"), moment.detach().cpu().numpy())

            Z = brain_encoder.encode(
                X.to(device), subject_idxs.to(device), normalize=False, swap_dims=True
            )
            # ( 4, 768 )

            for i, Z_ in enumerate(Z):
                # NOTE: Unsqueezing corresponds to making the dimension of 77 in CLIP-Text.
                Z_ = Z_.detach().cpu().numpy()
                np.save(os.path.join(save_dir, f"{idx}_{i}.npy"), Z_)

            # image_clip = preprocess(image_clip).to(device)
            # Y_clip = clip_model.encode_image(image_clip.unsqueeze(0)).float()
            # Y_clip = Y_clip.squeeze(0).detach().cpu().numpy()
            # cprint(f"Y_clip (image): {Y_clip.shape}, {Y_clip.dtype}, Mean: {Y_clip.mean()}, Std: {Y_clip.std()}", "cyan")  # fmt: skip


if __name__ == "__main__":
    main()
