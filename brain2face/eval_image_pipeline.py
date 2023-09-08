import os, sys
import numpy as np
import cv2
import torch
from termcolor import cprint

import hydra
from omegaconf import DictConfig, OmegaConf

from dalle2_pytorch import DALLE2, DiffusionPriorNetwork, DiffusionPrior, Unet, Decoder

from brain2face.datasets import UHDPipelineDataset, YLabGODPipelineDataset
from brain2face.models.brain_encoder import BrainEncoderReduceTime
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
                "/".join(_args.config_path.split("/")[:-1]),
                f"{model}.yaml"
            )
        )
        for model in ["prior", "decoder"]
    )

    run_dir_clip = get_run_dir(args_clip)
    
    gen_dir = os.path.join("generated", args_clip.dataset.lower())

    # NOTE: devices need to be hard-coded, as I cannot figure out how the device
    #       that pytorch-dalle2 uses
    device_clip = "cuda:1"
    device_dalle2 = "cuda:0"
    
    batch_size = 32

    # ----------------
    #    Dataloader
    # ----------------
    session_id = 0
    train_set = eval(f"{args_clip.dataset}PipelineDataset")(args_clip, session_id)
    # test_set = UHDPipelineDataset(args_clip, session_id=0, train=False)

    loader_args = {"shuffle": False, "drop_last": False, "num_workers": 4, "pin_memory": True}  # fmt: skip
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=batch_size, **loader_args
    )
    # test_loader = torch.utils.data.DataLoader(
    #     dataset=test_set, batch_size=len(test_set), **loader_args
    # )

    # ----------------
    #      Models
    # ----------------
    weights_brain_enc = torch.load(
        os.path.join(run_dir_clip, "brain_encoder_best.pt"), map_location=device_clip
    )

    subject_names = train_set.subject_names

    brain_encoder = BrainEncoderReduceTime(
        args_clip,
        subject_names=subject_names,
        layout=eval(args_clip.layout),
        time_multiplier=args_clip.time_multiplier,
    ).to(device_clip)  # fmt: skip

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

    diffusion_prior.load_state_dict(
        torch.load(
            os.path.join(
                "runs/prior",
                args_clip.dataset.lower(),
                args_prior.train_name,
                "prior_best.pt",
            ),
            map_location=device_dalle2,
        )
    )

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
                args_decoder.train_name,
                "decoder_best.pt",
            ),
            map_location=device_dalle2,
        )
    )

    dalle2 = DALLE2(
        prior=diffusion_prior,
        decoder=decoder,
        prior_num_samples=1,
    )

    # ----------------
    #     Pipeline
    # ----------------
    for i, (X, Y, subject_idxs) in enumerate(train_loader):
        X = X.to(device_clip)

        brain_embed = brain_encoder(X, subject_idxs).to(device_dalle2)

        images = dalle2(brain_embed, return_pil_images=True)

        # images = (images.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        # ( samples, 256, 256, 3 )
                
        for j, (image, image_gt) in enumerate(zip(images, Y.numpy())):
            image.save(
                os.path.join(gen_dir, f"{i * batch_size + j}.jpg")
            )            
            cv2.imwrite(
                os.path.join(gen_dir, f"{i * batch_size + j}_gt.jpg"), image_gt
            )


if __name__ == "__main__":
    pipeline()
