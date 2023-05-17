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

from brain2face.preproc.extractor import FaceExtractor
from constants import DLIB_PREDICTOR_PATH, EXTRACTED_VIDEO_ROOT

from encoder4editing.models.psp import pSp
from encoder4editing.utils.alignment import align_face

# import multiprocessing

# ctx = multiprocessing.get_context("spawn")


def face_preproc(args, video_path, video_times_path) -> Tuple[torch.Tensor, np.ndarray]:
    # -------------------------
    #        Input Video
    # -------------------------
    cap = cv2.VideoCapture(video_path)
    input_size = np.array(
        [int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))]
    )
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # -------------------------
    #      Face Extractor
    # -------------------------
    face_extractor = FaceExtractor(args.face_extractor, input_size, "center", fps)

    # -------------------------
    #         StyleGAN
    # -------------------------
    stylegan_encoder = StyleGANEncoder(args.stylegan.model_path)

    # -------------------------
    #         Outputs
    # -------------------------
    fmt = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    writer = cv2.VideoWriter(
        f"{EXTRACTED_VIDEO_ROOT}/{Path(video_path).stem}.mp4",
        fmt,
        fps,
        tuple(args.face_extractor.output_size),
    )

    y_times = np.loadtxt(video_times_path, delimiter=",")  # ( 109800, )

    y_list = []
    no_face_idxs = []
    i = 0
    batch_size = 32
    transformed_list = []

    segment_len = args.seq_len * fps  # 90

    if not args.debug:
        pbar = tqdm(total=num_frames)
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                cprint("Reached to the final frame", color="yellow")
                break

            extracted = face_extractor.run(i, frame)

            if extracted is not None:
                writer.write(extracted)

                aligned_image = align_face(extracted, stylegan_encoder.predictor)

                if aligned_image is not None:
                    transformed_image = stylegan_encoder.img_transforms(aligned_image)

                    transformed_list.append(transformed_image)

                else:
                    if i < len(y_times):
                        no_face_idxs.append(i)
            else:
                if i < len(y_times):
                    no_face_idxs.append(i)

                writer.write(
                    np.zeros((*args.face_extractor.output_size, 3), dtype=np.uint8)
                )

            i += 1
            if i % 100 == 0:
                pbar.update(100)

            if len(transformed_list) == batch_size:
                _, latents = stylegan_encoder.run_on_batch(transformed_list)
                y_list.append(latents)
                transformed_list = []

            # if i == 5000:
            #     break

            assert len(transformed_list) <= batch_size

        if not len(transformed_list) == 0:
            _, latents = stylegan_encoder.run_on_batch(transformed_list)
            y_list.append(latents)

        cap.release()

        y = torch.cat(y_list).numpy()  # ( 109797, 18, 512 )

    else:
        y = np.random.rand(100000, 18, 512)

    y = y[: -(y.shape[0] % segment_len)]  # ( 109710, 18, 512 )
    y = y.reshape(-1, segment_len, *y.shape[-2:])  # ( 1219, 90, 18, 512 )

    y_times = np.delete(y_times, no_face_idxs)
    y_times = y_times[::segment_len][: y.shape[0]]  # ( 1219, )

    assert len(y) == len(y_times)

    return y, y_times


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
