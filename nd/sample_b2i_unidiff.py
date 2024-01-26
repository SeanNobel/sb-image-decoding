import os
import random
import numpy as np
import torch
import ml_collections
import clip
import einops
import time
from contextlib import contextmanager
from torchvision import transforms
from torchvision.utils import save_image, make_grid

from absl import flags, app, logging
from ml_collections import config_flags

import unidiffuser.utils as utils
import unidiffuser.libs as libs
from unidiffuser.dpm_solver_pp import NoiseScheduleVP, DPM_Solver

from nd.models.brain_encoder import BrainEncoder


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


class SamplerBase:
    def _split(self, x: torch.Tensor):
        """_summary_
        Args:
            x ( n, c' * h' * w' + l * d ): _description_
        Returns:
            z ( n, c', h', w' ): _description_
            clip_img ( n, l, d ): _description_
        """
        c, h, w = self.config.z_shape
        z_dim = c * h * w

        z, clip_img = x.split([z_dim, self.config.clip_img_dim], dim=1)

        z = einops.rearrange(z, "n (c h w) -> n c h w", c=c, h=h, w=w)
        clip_img = einops.rearrange(
            clip_img, "n (l d) -> n l d", l=1, d=self.config.clip_img_dim
        )

        return z, clip_img

    def _combine(self, z: torch.Tensor, clip_img: torch.Tensor):
        """_summary_
        Args:
            z ( n, c', h', w' ): _description_
            clip_img ( n, l, d ): _description_
        Returns:
            x ( n, c' * h' * w' + l * d ): _description_
        """
        z = einops.rearrange(z, "n c h w -> n (c h w)")
        clip_img = einops.rearrange(clip_img, "n l d -> n (l d)")

        return torch.cat([z, clip_img], dim=-1)

    @staticmethod
    def _unpreprocess(v: torch.Tensor):
        return (0.5 * (v + 1.0)).clamp(0.0, 1.0)

    @torch.cuda.amp.autocast()
    def _decode(self, _batch):
        return self.autoencoder.decode(_batch)


class Brain2ImageSampler(SamplerBase):
    def __init__(self, config):
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
        self.brain_encoder = None
        # TODO: What is the null context for MEG?
        self.empty_context = None

        # Stable Diffusion
        self.autoencoder = libs.autoencoder.get_model(**config.autoencoder)
        self.autoencoder.to(self.device)

        # CLIP image encoder
        self.clip_img_model, self.clip_img_preproc = clip.load(
            "ViT-B/32", device=self.device, jit=False
        )

    def _prepare_contexts(self, prompts: torch.Tensor):
        """_summary_
        Args:
            prompts ( n, c, t ): MEG samples to sample images from.
        Returns:
            contexts ( n, 1?, F ): MEG embeddings.
                TODO: Current brain encoder reduces temporal dimension, but that that could be restored.
            img_contexts ( n, 2c', h', w' ): Stable Diffusion image embeddings. Moments?
            clip_imgs ( n, 1, 512 ): CLIP image embeddings.
        """
        resolution = self.config.z_shape[-1] * 8  # 512

        contexts = self.brain_encoder.encode(prompts)
        img_contexts = torch.randn(
            self.n_samples, 2 * self.config.z_shape[0], *self.config.z_shape[1:]
        )
        clip_imgs = torch.randn(self.n_samples, 1, self.config.clip_img_dim)

        return contexts, img_contexts, clip_imgs

    def _b2i_nnet(
        self, x: torch.Tensor, t_continuous: torch.Tensor, brain: torch.Tensor
    ):
        """Classifier free guidance.
        Args:
            x ( n, ): _description_
            timesteps (_type_): _description_
            brain (_type_): _description_
        """
        timesteps = t_continuous * self.N

        z, clip_img = self._split(x)

        t_brain = torch.zeros(timesteps.shape[0], dtype=torch.int, device=self.device)

        z_out, clip_img_out, brain_out = self.nnet(
            z,
            clip_img,
            text=brain,
            t_img=timesteps,
            t_text=t_brain,
            data_type=torch.zeros_like(t_brain, device=self.device, dtype=torch.int)
            + self.config.data_type,
        )

        x_out = self._combine(z_out, clip_img_out)

        if self.config.sample.scale == 0.0:
            return x_out

        if self.config.sample.b2i_cfg_mode == "true_uncond":
            z_out_uncond, clip_img_out_uncond, _ = self.nnet(
                z,
                clip_img,
                text=torch.randn_like(brain),
                t_img=timesteps,
                t_text=torch.ones_like(timesteps) * self.N,
                data_type=torch.zeros_like(t_brain, device=self.device, dtype=torch.int)
                + self.config.data_type,
            )

            x_out_uncond = self._combine(z_out_uncond, clip_img_out_uncond)
        else:
            raise NotImplementedError

        return x_out + self.config.sample.scale * (x_out - x_out_uncond)

    def _sample_fn(self, contexts: torch.Tensor):
        """_summary_
        Args:
            contexts ( n, 1?, F ): _description_
        """
        z_init = torch.randn(self.n_samples, *self.config.z_shape, device=self.device)
        clip_img_init = torch.randn(
            self.n_samples, 1, self.config.clip_img_dim, device=self.device
        )
        brain_init = torch.randn_like(contexts, device=self.device)

        x_init = self.combine(z_init, clip_img_init)

        noise_schedule = NoiseScheduleVP(
            schedule="discrete",
            betas=torch.tensor(self._betas, device=self.device).float(),
        )

        dpm_solver = DPM_Solver(
            self._b2i_nnet, noise_schedule, predict_x0=True, thresholding=False
        )

        with torch.no_grad(), torch.autocast(device_type=self.device), timer(
            f"\ngenerate {self.n_samples} samples with {self.config.sample.sample_steps} steps"
        ):
            x = dpm_solver.sample(
                x_init, steps=self.config.sample.sample_steps, eps=1.0 / self.N, T=1.0
            )

        # os.makedirs(self.config.output_path, exist_ok=True)

        z, clip_img = self._split(x)

        return z, clip_img

    def sample(self, prompts: torch.Tensor):
        logging.info(self.config.sample)
        logging.info(f"N={self.N}")

        self.n_samples = prompts.shape[0]

        contexts, img_contexts, clip_imgs = self._prepare_contexts(prompts)

        z_img = self.autoencoder.sample(img_contexts)  # ( n, c', h', w' )

        z, clip_img = self._sample_fn(contexts)

        samples = self._unpreprocess(self._decode(z))

        save_dir = os.path.join(self.config.output_path, self.config.mode)
        os.makedirs(save_dir, exist_ok=True)

        for idx, sample in enumerate(samples):
            save_image(sample, os.path.join(save_dir, f"{idx}.png"))

        save_image(
            make_grid(samples, nrow=self.config.nrow),
            os.path.join(save_dir, "grid.png"),
        )


# fmt: off
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", "configs/sample_unidiffuser.py", "Configuration.", lock_config=False)
flags.DEFINE_string("nnet_path", "models/uvit_v1.pth", "The nnet to evaluate.")
flags.DEFINE_string("output_path", "out", "dir to write results to")
flags.DEFINE_string("prompt", "an elephant under the sea", "the prompt for text-to-image generation and text variation")
flags.DEFINE_string("img", "assets/space.jpg", "the image path for image-to-text generation and image variation")
flags.DEFINE_integer("n_samples", 1, "the number of samples to generate")
flags.DEFINE_integer("nrow", 4, "number of images displayed in each row of the grid")
flags.DEFINE_string("mode", None,
                    "type of generation, one of t2i / i2t / joint / i / t / i2t2i/ t2i2t\n"
                    "t2i: text to image\n"
                    "i2t: image to text\n"
                    "joint: joint generation of text and image\n"
                    "i: only generate image\n"
                    "t: only generate text\n"
                    "i2t2i: image variation, first image to text, then text to image\n"
                    "t2i2t: text variation, first text to image, the image to text\n"
                    )


def main(argv):
    config = FLAGS.config
    config.nnet_path = FLAGS.nnet_path
    config.output_path = FLAGS.output_path
    config.prompt = FLAGS.prompt
    config.nrow = min(FLAGS.nrow, FLAGS.n_samples)
    config.img = FLAGS.img
    config.n_samples = FLAGS.n_samples
    config.mode = FLAGS.mode
    
    sampler = Brain2ImageSampler(config)
    sampler.sample()

if __name__ == "__main__":
    app.run(main)
