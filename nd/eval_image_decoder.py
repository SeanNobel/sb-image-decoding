import os, sys
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import cv2
from PIL import Image
from time import time
from tqdm import tqdm
from termcolor import cprint
from typing import Union, Optional
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from dalle2_pytorch import Unet, Decoder

from nd.datasets import NeuroDiffusionCLIPEmbImageDataset, ThingsMEGDecoderDataset


@torch.no_grad()
@hydra.main(version_base=None, config_path="../configs", config_name="default")
def run(_args: DictConfig) -> None:
    args = OmegaConf.load(os.path.join("configs", _args.config_path))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    gen_dir = os.path.join("generated", args.dataset.lower(), "decoder")
    os.makedirs(os.path.join(gen_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(gen_dir, "test"), exist_ok=True)

    device = "cuda:0"

    # -----------------
    #    Dataloader
    # -----------------
    batch_size = 32

    if args.dataset == "ThingsMEG":
        dataset = ThingsMEGDecoderDataset(args)

        train_set = torch.utils.data.Subset(dataset, dataset.train_idxs)
        test_set = torch.utils.data.Subset(dataset, dataset.test_idxs)
    else:
        train_set = NeuroDiffusionCLIPEmbImageDataset(
            args.dataset, args.clip_train_name
        )
        test_set = NeuroDiffusionCLIPEmbImageDataset(
            args.dataset, args.clip_train_name, train=False
        )

    loader_args = {
        "batch_size": batch_size,
        "drop_last": True,
        "shuffle": False,
    }
    train_loader = torch.utils.data.DataLoader(dataset=train_set, **loader_args)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, **loader_args)

    # -----------------
    #      Decoder
    # -----------------
    unet1 = Unet(
        dim=args.unet1.dim,
        image_embed_dim=args.embed_dim,
        cond_dim=128,
        channels=3,
        dim_mults=tuple(args.unet1.dim_mults),
    ).to(device)

    unet2 = Unet(
        dim=args.unet2.dim,
        image_embed_dim=args.embed_dim,
        cond_dim=128,
        channels=3,
        dim_mults=tuple(args.unet2.dim_mults),
    ).to(device)

    decoder = Decoder(
        unet=(unet1, unet2),
        image_sizes=tuple(args.image_sizes),
        timesteps=args.timesteps,
    ).to(device)

    decoder_path = os.path.join(
        "runs/decoder", args.dataset.lower(), args.train_name, "decoder_best.pt"
    )
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    cprint(f"Loaded decoder from {decoder_path}", "cyan")

    # -----------------
    # Generate samples
    # -----------------
    for dataloader, split in zip([train_loader, test_loader], ["train", "test"]):
        image_embeds, images_gt = next(iter(dataloader))
        image_embeds = image_embeds.to(device)

        images = decoder.sample(image_embed=image_embeds, text=None, cond_scale=1.0)
        images = list(map(T.ToPILImage(), images.unbind(dim=0)))

        images_gt = (images_gt.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
        # images_gt = Image.fromarray(images_gt)

        for i, (image, image_gt) in enumerate(zip(images, images_gt)):
            # for i, image_gt in enumerate(images_gt):
            image.save(os.path.join(gen_dir, f"{split}_{i}.jpg"))
            cv2.imwrite(os.path.join(gen_dir, f"{split}_{i}_gt.jpg"), image_gt)


if __name__ == "__main__":
    run()
