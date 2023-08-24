import os, sys
import numpy as np
import cv2
import torch
import scipy.signal as signal
from termcolor import cprint

import hydra
from omegaconf import DictConfig, OmegaConf

from dalle2_pytorch import DiffusionPriorNetwork, DiffusionPrior
from dalle2_video.dalle2_video import DALLE2Video, Unet3D, UnetTemporalConv, VideoDecoder

from brain2face.datasets import UHDPipelineDataset
from brain2face.models.brain_encoder import BrainEncoder, BrainEncoderReduceTime
from brain2face.utils.layout import ch_locations_2d, DynamicChanLoc2d
from brain2face.utils.train_utils import sequential_apply
from brain2face.utils.eval_utils import update_with_eval, get_run_dir


@torch.no_grad()
@hydra.main(version_base=None, config_path="../configs", config_name="default")
def pipeline(_args: DictConfig) -> None:
    args_clip = OmegaConf.load(os.path.join("configs", _args.config_path))
    args_clip = update_with_eval(args_clip)

    args_prior, args_decoder = (
        OmegaConf.load(
            os.path.join(
                "configs",
                "/".join(_args.config_path.split("/")[:-2]),
                model,
                f"{'static' if args_clip.reduce_time else 'temporal'}-emb.yaml",
            )
        )
        for model in ["prior", "decoder"]
    )

    run_dir_clip = get_run_dir(args_clip)

    # NOTE: devices need to be hard-coded, as I cannot figure out how the device
    #       that pytorch-dalle2 uses
    device_clip = "cuda:1"
    device_prior = "cuda:1"
    device_decoder = "cuda:2"

    brain_enc_path = os.path.join(run_dir_clip, "brain_encoder_best.pt")
    prior_path = os.path.join(
        "runs/prior",
        args_clip.dataset.lower(),
        args_prior.type,
        args_prior.train_name,
        "prior_best.pt",
    )
    decoder_path = os.path.join(
        "runs/decoder",
        args_clip.dataset.lower(),
        args_decoder.type,
        args_decoder.train_name,
        "decoder_best.pt",
    )
    cprint(f"Loading brain encoder weights from {brain_enc_path}", "cyan")
    cprint(f"Loading prior weights from {prior_path}", "cyan")
    cprint(f"Loading decoder weights from {decoder_path}", "cyan")

    # ----------------
    #    Dataloader
    # ----------------
    session_id = 0

    train_set = UHDPipelineDataset(
        args_clip,
        session_id,  # Which session to run pipeline on
        # start=0,
        # end=10, # Which part of the session to make video of (in seconds)
    )
    # test_set = UHDPipelineDataset(args_clip, session_id, train=False)

    subject_names = train_set.subject_names

    loader_args = {"shuffle": False, "drop_last": False, "num_workers": 4, "pin_memory": True}  # fmt: skip
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=1, **loader_args
    )
    # test_loader = torch.utils.data.DataLoader(
    #     dataset=test_set, batch_size=len(test_set), **loader_args
    # )

    # ----------------
    #      Models
    # ----------------
    # ---- Brain Encoder ---- #
    if not args_clip.reduce_time:
        brain_encoder = BrainEncoder(
            args_clip,
            subject_names=subject_names,
            layout=eval(args_clip.layout),
        ).to(device_clip)

    else:
        brain_encoder = BrainEncoderReduceTime(
            args_clip,
            subject_names=subject_names,
            layout=eval(args_clip.layout),
            time_multiplier=args_clip.time_multiplier,
        ).to(device_clip)

    brain_encoder.load_state_dict(torch.load(brain_enc_path, map_location=device_clip))
    brain_encoder.eval()

    # NOTE: Taking num_subjects from the saved weights.
    # num_subjects = len(
    #     [key for key in weights_brain_enc.keys() if "subject_layer" in key]
    # )

    # ---- Diffusion Prior ---- #
    prior_network = DiffusionPriorNetwork(
        dim=args_prior.dim,
        depth=args_prior.depth,
        dim_head=args_prior.dim_head,
        heads=args_prior.heads,
    ).to(device_prior)

    diffusion_prior = DiffusionPrior(
        net=prior_network,
        image_embed_dim=args_prior.image_embed_dim,
        timesteps=args_prior.timesteps,
        cond_drop_prob=args_prior.cond_drop_prob,
        condition_on_text_encodings=False,
    ).to(device_prior)

    diffusion_prior.load_state_dict(torch.load(prior_path, map_location=device_prior))

    # ---- Decoder ---- #
    unet1 = Unet3D(
        dim=args_decoder.unet1.dim,
        video_embed_dim=args_decoder.video_embed_dim,
        channels=args_decoder.channels,
        dim_mults=tuple(args_decoder.unet1.dim_mults),
        cond_on_text_encodings=False,
    ).to(device_decoder)

    unet2 = Unet3D(
        dim=args_decoder.unet2.dim,
        video_embed_dim=args_decoder.video_embed_dim,
        channels=args_decoder.channels,
        dim_mults=tuple(args_decoder.unet2.dim_mults),
        cond_on_text_encodings=False,
    ).to(device_decoder)

    decoder = VideoDecoder(
        unet=(unet1, unet2),
        frame_sizes=tuple(args_decoder.frame_sizes),
        frame_numbers=tuple(args_decoder.frame_numbers),
        timesteps=args_decoder.timesteps,
        learned_variance=False,
    ).to(device_decoder)

    decoder.load_state_dict(torch.load(decoder_path, map_location=device_decoder))

    dalle2 = DALLE2Video(
        prior=diffusion_prior,
        decoder=decoder,
        temporal_emb=args_prior.temporal_emb,
        prior_num_samples=1,  # NOTE: Somehow setting this number larger reduces batch size. Why?
        decoder_cuda=device_decoder,
    )

    # ----------------
    #     Pipeline
    # ----------------
    X, subject_idxs = next(iter(train_loader))

    X = X.to(device_clip)

    Z = brain_encoder(X, subject_idxs)

    # Normalization
    if not args_clip.reduce_time:
        b, d, t = Z.shape
        Z = Z.reshape(b, -1)

    Z /= Z.norm(dim=-1, keepdim=True)

    if not args_clip.reduce_time:
        Z = Z.reshape(b, d, t)

    # FIXME: Resampling from 90 to 16 frames as I cannot train video decoder with 90 frames.
    # Z = signal.resample(Z.cpu().numpy(), num=16, axis=-1)
    # Z = torch.from_numpy(Z).to(device_prior)
    # brain_embed = brain_embed.mean(dim=-1)

    cprint(Z.shape, "yellow")

    videos = dalle2(text_embed=Z)
    # , return_pil_images=True)

    video = videos[0]

    # video = (video - video.min()) / (video.max() - video.min())

    video = (video.permute(1, 2, 3, 0).cpu().numpy() * 255).astype(np.uint8)
    # ( t=90, 256, 256, 3 )

    cprint(video, "yellow")
    cprint(video.mean(), "yellow")
    cprint(video.shape, "yellow")

    fmt = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    writer = cv2.VideoWriter(
        "assets/generated_videos/unet3d.mp4", fmt, 5, tuple(video.shape[1:3])
    )

    for frame in video:
        writer.write(frame)
        # cv2.imwrite(f"assets/generated_images/{i}.jpg", np.array(image, dtype=np.uint8))

    writer.release()


if __name__ == "__main__":
    pipeline()
