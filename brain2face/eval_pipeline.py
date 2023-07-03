import os
import numpy as np
import cv2
import torch
from termcolor import cprint

import hydra
from omegaconf import DictConfig, OmegaConf

from dalle2_pytorch import DALLE2, DiffusionPriorNetwork, DiffusionPrior, Unet, Decoder

from brain2face.models.brain_encoder import BrainEncoderReduceTime
from brain2face.utils.eval_utils import update_with_eval, get_run_dir


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def pipeline(_args: DictConfig) -> None:
    args_clip = OmegaConf.load(os.path.join("configs", _args.config_path))
    args_clip = update_with_eval(args_clip)

    args_prior = OmegaConf.load(os.path.join("configs", "diffusion_prior.yaml"))
    args_decoder = OmegaConf.load(os.path.join("configs", "decoder.yaml"))

    run_dir_clip = get_run_dir(args_clip)

    # NOTE: devices need to be hard-coded, as I cannot figure out how the device
    #       that pytorch-dalle2 uses
    device_clip = "cuda:1"
    device_dalle2 = "cuda:0"

    # ----------------
    #      Models
    # ----------------
    weights_brain_enc = torch.load(
        os.path.join(run_dir_clip, "brain_encoder_best.pt"), map_location=device_clip
    )

    num_subjects = len(
        [key for key in weights_brain_enc.keys() if "subject_layer" in key]
    )

    brain_encoder = BrainEncoderReduceTime(
        args_clip, num_subjects=num_subjects, unknown_subject=True
    ).to(device_clip)

    brain_encoder.load_state_dict(weights_brain_enc)
    brain_encoder.eval()

    prior_network = DiffusionPriorNetwork(
        dim=args_prior.dim,
        depth=args_prior.depth,
        dim_head=args_prior.dim_head,
        heads=args_prior.heads,
    ).to(device_dalle2)

    diffusion_prior = DiffusionPrior(
        net=prior_network,
        image_embed_dim=args_prior.image_embed_dim,
        timesteps=args_prior.timesteps,
        cond_drop_prob=args_prior.cond_drop_prob,
        condition_on_text_encodings=False,
    ).to(device_dalle2)

    unet1 = Unet(
        dim=args_decoder.unet1.dim,
        image_embed_dim=args_decoder.image_embed_dim,
        # text_embed_dim=args.text_embed_dim,
        channels=args_decoder.channels,
        dim_mults=tuple(args_decoder.unet1.dim_mults),
        cond_on_text_encodings=False,
    ).to(device_dalle2)

    unet2 = Unet(
        dim=args_decoder.unet2.dim,
        image_embed_dim=args_decoder.image_embed_dim,
        # text_embed_dim=args.text_embed_dim,
        channels=args_decoder.channels,
        dim_mults=tuple(args_decoder.unet2.dim_mults),
        cond_on_text_encodings=False,
    ).to(device_dalle2)

    decoder = Decoder(
        unet=(unet1, unet2),
        image_sizes=tuple(args_decoder.image_sizes),
        timesteps=args_decoder.timesteps,
    ).to(device_dalle2)

    decoder.load_state_dict(
        torch.load(
            os.path.join(
                "runs/decoder",
                args_clip.dataset.lower(),
                args_decoder.run_name,
                "decoder_last.pt",
            ),
            map_location=device_dalle2,
        )
    )

    dalle2 = DALLE2(
        prior=diffusion_prior,
        decoder=decoder,
    )

    # ----------------
    #     Pipeline
    # ----------------
    mock_batch = torch.randn(
        4, args_clip.num_channels, args_clip.seq_len * args_clip.brain_resample_sfreq
    ).to(device_clip)

    brain_emb = brain_encoder(mock_batch, None).to(device_dalle2)
    cprint(brain_emb.shape, "cyan")

    images = dalle2(brain_emb, return_pil_images=True)

    for i, image in enumerate(images):
        cv2.imwrite(f"assets/generated_images/{i}.jpg", np.array(image, dtype=np.uint8))


if __name__ == "__main__":
    pipeline()
