import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import dlib
from argparse import Namespace
from tqdm import tqdm
from termcolor import cprint
from typing import Optional, Tuple, List
from pathlib import Path

from brain2face.constants import DLIB_PREDICTOR_PATH

from encoder4editing.models.psp import pSp


class StyleGANEncoder:
    def __init__(self, model_path: str) -> None:
        self.predictor = dlib.shape_predictor(DLIB_PREDICTOR_PATH)

        ckpt = torch.load(model_path, map_location="cpu")
        opts = ckpt["opts"]
        opts["checkpoint_path"] = model_path
        self.net = pSp(Namespace(**opts))
        self.net.to("cuda")
        # self.net = nn.DataParallel(self.net, device_ids=[0, 1, 2, 3])
        self.net.eval()
        # self.first_device = self.net.device_ids[0]
        # cprint(f"Model first device: {self.first_device}", "cyan")

        self.img_transforms = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def run_on_batch(
        self, transformed_images: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # cprint("Aligning images...", "cyan")

        # transformed_images = []

        # for input_image in input_images:
        # NOTE: saving to disk to use the align_face function without any modification
        # cv2.imwrite(TMP_FACE_PATH, input_image)

        # aligned_image = self.run_alignment(input_image)

        # transformed_image = self.img_transforms(aligned_image)

        # transformed_images.append(transformed_image)

        # aligned_images = ctx.Pool(32).map(self.run_alignment, input_images)
        # cprint("Images aligned.", "cyan")

        # transformed_images = [
        #     self.img_transforms(aligned_image) for aligned_image in aligned_images
        # ]
        # p = ctx.Pool(32)
        # transformed_images = p.map(self.img_transforms, aligned_images)
        # p.close()
        # cprint("Images transformed.", "cyan")

        # first_device = self.net.device_ids[0]
        transformed_images = torch.stack(transformed_images).to("cuda").float()

        # cprint(f"Transformed images have shape: {transformed_images.shape}", "cyan")

        with torch.no_grad():
            images, latents = self.net(
                transformed_images, randomize_noise=False, return_latents=True
            )
        return images.cpu(), latents.cpu()

    # def run_alignment(self, input_image: np.ndarray) -> Image:
    #     aligned_image = align_face(input_image, self.predictor)
    #     # cprint("Aligned image has shape: {}".format(aligned_image.size), "cyan")

    #     return aligned_image
