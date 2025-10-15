import os, sys
import random
import numpy as np
import torch
import ml_collections
import clip
import einops
import time
from glob import glob
from natsort import natsorted
from termcolor import cprint
from contextlib import contextmanager
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from functools import partial
from hydra import initialize, compose
import omegaconf
from typing import Tuple, Any, Union, List

from absl import flags, app, logging
from ml_collections import config_flags

import uvit.utils as utils
import uvit.libs as libs
from uvit.dpm_solver_pp import NoiseScheduleVP, DPM_Solver
from uvit.datasets import ThingsMEGDatabase

from sbid.datasets import ThingsMEGCLIPDataset
from sbid.models.brain_encoder import BrainEncoder
from sbid.utils.layout import ch_locations_2d
from sbid.utils.eval_utils import get_run_dir


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@contextmanager
def timer(prefix: str):
    stime = time.time()
    yield
    print(f"{prefix} | took {time.time() - stime:.2f}s")


def stable_diffusion_beta_schedule(
    linear_start=0.00085, linear_end=0.0120, n_timestep=1000
):
    _betas = torch.linspace(
        linear_start**0.5, linear_end**0.5, n_timestep, dtype=torch.float64
    )
    _betas **= 2

    return _betas.numpy()


class Brain2ImageSampler:
    def __init__(
        self,
        config: ml_collections.ConfigDict,
        args: omegaconf.DictConfig,
        num_subjects: int,
    ):
        """
        Args:
            config: Configs related to U-ViT
            args: Configs related to CLIP
        """
        if config.get("benchmark", False):
            """cuDNN selects the fastest convolution algorithm after benchmarking them.
            This leads to non-deterministic results if the input size changes."""
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        set_seed(config.seed)

        self.config = ml_collections.FrozenConfigDict(config)
        utils.set_logger(log_level="info")

        self._betas = stable_diffusion_beta_schedule()  # ( 1000, )
        self.N = len(self._betas)

        # U-ViT
        logging.info(f"Loading nnet from {self.config.nnet_path}")
        self.nnet = utils.get_nnet(**self.config.nnet)
        self.nnet.load_state_dict(torch.load(self.config.nnet_path, map_location="cpu"))
        self.nnet.to(self.device)
        self.nnet.eval()

        # TODO: load CLIP Brain Encoder
        self.brain_encoder = BrainEncoder(
            args,
            subjects=num_subjects,
            layout=eval(args.layout),
            vq=args.vq,
            blocks=args.blocks,
            downsample=args.downsample,
            temporal_aggregation=args.temporal_aggregation,
        ).to(self.device)
        self.brain_encoder.load_state_dict(
            torch.load(
                os.path.join(args.root, get_run_dir(args), "brain_encoder_best.pt"),
                map_location=self.device,
            )
        )
        self.brain_encoder.eval()

        # TODO: What is the null context for MEG?
        self.empty_context = torch.tensor(
            np.load(self.config.empty_context_path), device=self.device
        )

        # Stable Diffusion
        self.autoencoder = libs.autoencoder.get_model(**config.autoencoder)
        self.autoencoder.to(self.device)

    @staticmethod
    def _unpreprocess(v: torch.Tensor):
        return (0.5 * (v + 1.0)).clamp(0.0, 1.0)

    @torch.cuda.amp.autocast()
    def _decode(self, _batch):
        return self.autoencoder.decode(_batch)

    def _b2i_nnet(
        self, x: torch.Tensor, t_continuous: torch.Tensor, brain: torch.Tensor
    ):
        """Classifier free guidance.
        Args:
            x ( n, 4, 32, 32 ): _description_
            timesteps ( n, ): _description_
            brain ( n, 1, 768 ): _description_
        """
        timesteps = t_continuous * self.N

        x_out = self.nnet(x, timesteps, context=brain)

        if self.config.sample.scale == 0.0:
            return x_out

        empty_context = einops.repeat(self.empty_context, "l d -> n l d", n=x.shape[0])
        x_out_uncond = self.nnet(x, timesteps, context=empty_context)

        return x_out + self.config.sample.scale * (x_out - x_out_uncond)

    def _sample_fn(self, contexts: torch.Tensor):
        """_summary_
        Args:
            contexts ( n, 1?, F ): _description_
        """
        z_init = torch.randn(self.n_samples, *self.config.z_shape, device=self.device)
        # ( n, 4, 32, 32 )

        noise_schedule = NoiseScheduleVP(
            schedule="discrete",
            betas=torch.tensor(self._betas, device=self.device).float(),
        )

        dpm_solver = DPM_Solver(
            partial(self._b2i_nnet, brain=contexts.to(self.device)),
            noise_schedule,
            predict_x0=True,
            thresholding=False,
        )

        with torch.no_grad(), torch.autocast(device_type=self.device), timer(
            f"\ngenerate {self.n_samples} samples with {self.config.sample.sample_steps} steps"
        ):
            z = dpm_solver.sample(
                z_init, steps=self.config.sample.sample_steps, eps=1.0 / self.N, T=1.0
            )

        return z

    def sample(self, prompts: Tuple[torch.Tensor], filename: str):
        logging.info(self.config.sample)
        logging.info(f"N={self.N}")

        self.n_samples = prompts[0].shape[0]

        contexts = self.brain_encoder.encode(
            *prompts, device=self.device, normalize=False
        )
        # FIXME: Brain encoder currently reduces temporal dimension, but U-ViT expects it.
        contexts = contexts.unsqueeze(1)  # ( n, 1, F )

        z = self._sample_fn(contexts)

        samples = self._unpreprocess(self._decode(z))

        save_dir = os.path.join(self.config.output_path, self.config.mode)
        os.makedirs(save_dir, exist_ok=True)

        save_image(
            make_grid(samples, nrow=self.config.nrow),
            os.path.join(save_dir, f"{filename}.png"),
        )


# fmt: off
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", "uvit/configs/thingsmeg_uvit_small.py", "Configuration.", lock_config=False
)
nnet_dir = "uvit/workdir/thingsmeg_uvit_small/default/ckpts"
nnet_path = natsorted(glob(os.path.join(nnet_dir, "*.ckpt")))[-1]
nnet_path = os.path.join(nnet_path, "nnet.pth")
cprint(nnet_path, "cyan")
flags.DEFINE_string("nnet_path", nnet_path, "The nnet to evaluate.")

empty_context_path = "data/uvit/thingsmeg_features/empty_context_from_zeros.npy"
flags.DEFINE_string("empty_context_path", empty_context_path, "The path to empty context.")

flags.DEFINE_string("output_path", "out", "dir to write results to")
flags.DEFINE_integer("n_samples", 4, "the number of samples to generate")
flags.DEFINE_integer("nrow", 4, "number of images displayed in each row of the grid")
flags.DEFINE_string("mode", "b2i", "mode of sampling. this script is fixed to brain2image.")

SAMPLE_IDXS = np.arange(0, 100, 10)

def main(argv):
    config = FLAGS.config
    config.nnet_path = FLAGS.nnet_path
    config.empty_context_path = FLAGS.empty_context_path
    config.output_path = FLAGS.output_path
    config.nrow = min(FLAGS.nrow, FLAGS.n_samples)
    config.n_samples = FLAGS.n_samples
    config.mode = FLAGS.mode
    
    config.autoencoder.pretrained_path = os.path.join("uvit", config.autoencoder.pretrained_path)
    
    # Configs related to CLIP
    with initialize(version_base=None, config_path="../configs/thingsmeg"):
        args = compose(config_name="clip.yaml")
        
    dataset = ThingsMEGDatabase(args)
    num_subjects = dataset.num_subjects if hasattr(dataset, "num_subjects") else len(dataset.subject_names) # fmt: skip
            
    sampler = Brain2ImageSampler(config, args, num_subjects)
    
    for sample_idx in SAMPLE_IDXS:
        train_prompts = (
            dataset.X[dataset.train_idxs][sample_idx], # ( num_subjects, 271, 169 )
            torch.arange(num_subjects)
        )
        
        filename = dataset.Y_paths[dataset.train_idxs][sample_idx]
        filename = os.path.splitext(os.path.basename(filename))[0]
        
        sampler.sample(train_prompts, filename)

if __name__ == "__main__":
    app.run(main)
