import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.utils import save_image, make_grid
from torchvision.io import read_image
from torchmetrics.functional.image import structural_similarity_index_measure as ssim_fn
from torchmetrics.image.inception import InceptionScore
from pathlib import Path
from absl import logging
from typing import Tuple, Dict, Optional
from termcolor import cprint
from tqdm import tqdm
from omegaconf import OmegaConf

import clip
from uvit import utils

from sbid.models import BrainEncoder, BrainAutoencoder, BrainMAE
from sbid.utils.eval_utils import update_with_eval, get_run_dir


def get_config_name():
    argv = sys.argv
    for i in range(1, len(argv)):
        if argv[i].startswith("--config="):
            return Path(argv[i].split("=")[-1]).stem


def get_hparams():
    argv = sys.argv
    lst = []
    for i in range(1, len(argv)):
        assert "=" in argv[i]
        if argv[i].startswith("--config.") and not argv[i].startswith("--config.dataset.path"):
            hparam, val = argv[i].split("=")
            hparam = hparam.split(".")[-1]
            if hparam.endswith("path"):
                val = Path(val).stem
            lst.append(f"{hparam}={val}")
    hparams = "-".join(lst)
    if hparams == "":
        hparams = "default"
    return hparams


class Schedule(object):
    def __init__(
        self,
        linear_start: float = 0.00085,
        linear_end: float = 0.0120,
        T: int = 1000,
    ):
        # self, linear_start: float = 0.00085, linear_end: float = 0.012, T: int = 1000
        """_betas[0...999] = betas[1...1000]
        for n>=1, betas[n] is the variance of q(xn|xn-1)
        for n=0,  betas[0]=0
        """
        self._betas = self._beta_schedule(linear_start, linear_end, T)  # ( 1000, )
        self.betas = np.append(0.0, self._betas)
        self.alphas = 1.0 - self.betas
        self.T = len(self._betas)

        assert isinstance(self.betas, np.ndarray) and self.betas[0] == 0
        assert isinstance(self.alphas, np.ndarray) and self.alphas[0] == 1
        assert len(self.betas) == len(self.alphas)

        # skip_alphas[s, t] = alphas[s + 1: t + 1].prod()
        self.skip_alphas, self.skip_betas = self._get_skip(self.alphas, self.betas)
        self.cum_alphas = self.skip_alphas[0]  # ( 1001, )
        self.cum_betas = self.skip_betas[0]
        self.snr = self.cum_alphas / self.cum_betas

    def sample(self, x_0: torch.Tensor):
        t = np.random.choice(list(range(1, self.T + 1)), (len(x_0),))

        eps = torch.randn_like(x_0)

        x_t = self._sample(t, x_0, eps)

        return torch.tensor(t, device=x_0.device), eps, x_t

    def p_sample(self, t_prev: int, t: int, x_t, eps):
        coef = self._betas[t - 1] / (1 - self.cum_alphas[t]) ** 0.5
        x_t = x_t - coef * eps

        x_t = x_t / (1 - self._betas[t - 1]) ** 0.5

        if t_prev > 1:
            # posterior_var = self._betas[t - 1] * (1 - self.cum_alphas[t_prev]) / (1 - self.cum_alphas[t])  # fmt: skip
            x_t = x_t + (self.tilde_beta(t_prev, t) ** 0.5) * torch.randn_like(x_t)

        return x_t

    def _sample(self, t: int, x_0: torch.Tensor, eps: torch.Tensor):
        return self._stp(self.cum_alphas[t] ** 0.5, x_0) + self._stp(self.cum_betas[t] ** 0.5, eps)

    @staticmethod
    def _beta_schedule(linear_start, linear_end, n_timestep) -> np.ndarray:
        """Stable Diffusion beta schedule"""
        _betas = torch.linspace(linear_start**0.5, linear_end**0.5, n_timestep, dtype=torch.float64)
        _betas **= 2

        return _betas.numpy()

    @staticmethod
    def _stp(s, ts: torch.Tensor):  # scalar tensor product
        if isinstance(s, np.ndarray):
            s = torch.from_numpy(s).type_as(ts)
        extra_dims = (1,) * (ts.dim() - 1)
        return s.view(-1, *extra_dims) * ts

    @staticmethod
    def _get_skip(alphas, betas):
        N = len(betas) - 1

        skip_alphas = np.ones([N + 1, N + 1], dtype=betas.dtype)
        for s in range(N + 1):
            skip_alphas[s, s + 1 :] = alphas[s + 1 :].cumprod()

        skip_betas = np.zeros([N + 1, N + 1], dtype=betas.dtype)
        for t in range(N + 1):
            prod = betas[1 : t + 1] * skip_alphas[1 : t + 1, t]
            skip_betas[:t, t] = (prod[::-1].cumsum())[::-1]

        return skip_alphas, skip_betas

    def tilde_beta(self, s, t):
        return self.skip_betas[s, t] * self.cum_betas[s] / self.cum_betas[t]

    def __repr__(self):
        return f"Schedule({self.betas[:10]}..., {self.T})"


class Bridge(Schedule):
    def __init__(self, linear_start: float = 1e-4, linear_end: float = 2e-2, T: int = 1000):
        super().__init__(linear_start, linear_end, T)

        self._var_fwd = self._betas.cumsum()
        self._var_bwd = np.flip(np.flip(self._betas).cumsum())

    @staticmethod
    def _beta_schedule(linear_start, linear_end, n_timestep) -> np.ndarray:
        """symmetric beta schedule"""
        assert n_timestep % 2 == 0
        _betas = torch.linspace(linear_start**0.5, linear_end**0.5, n_timestep, dtype=torch.float64)
        _betas = (_betas**2).numpy()[: n_timestep // 2]

        return np.concatenate([_betas, np.flip(_betas)])

    @staticmethod
    def _gaussian_product_coef(var1, var2):
        denom = var1 + var2
        mu1 = var2 / denom
        mu2 = var1 / denom
        var = var1 * var2 / denom

        return mu1, mu2, var**0.5

    def sample(self, x_0: torch.Tensor, x_T: torch.Tensor):
        t = np.random.choice(list(range(1, self.T + 1)), (len(x_0),))

        var_fwd = self._var_fwd[t - 1]
        var_bwd = self._var_bwd[t - 1]

        mu_0, mu_T, sigma = self._gaussian_product_coef(var_fwd, var_bwd)
        mu = self._stp(mu_0, x_0) + self._stp(mu_T, x_T)

        x_t = mu + self._stp(sigma, torch.randn_like(mu))

        eps = self._stp(1 / np.sqrt(var_fwd), x_t - x_0)

        return torch.tensor(t, device=x_0.device), eps.detach(), x_t.detach()

    def p_posterior(self, t_prev: int, t: int, x_t, x_0):
        """_summary_
        Args:
            t_prev (int): [999, ..., 1]
            t (int): [1000, ..., 2]
            x_t (_type_): _description_
            x_0 (_type_): _description_
        Returns:
            _type_: _description_
        """
        var = self._var_fwd[t - 1]
        var_prev = self._var_fwd[t_prev - 1]
        var_delta = var - var_prev

        mu_0, mu_t, sigma = self._gaussian_product_coef(var_prev, var_delta)

        x_t_prev = mu_0 * x_0 + mu_t * x_t

        if t_prev > 1:
            x_t_prev += sigma * torch.randn_like(x_t_prev)

        return x_t_prev


class TrainState(object):
    def __init__(
        self,
        optimizer,
        lr_scheduler,
        step,
        nnet=None,
        nnet_ema=None,
        brain_encoder=None,
    ):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.step = step
        self.nnet = nnet
        self.nnet_ema = nnet_ema
        self.brain_encoder = brain_encoder

    def ema_update(self, rate=0.9999):
        if self.nnet_ema is not None:
            utils.ema(self.nnet_ema, self.nnet, rate)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.step, os.path.join(path, "step.pth"))
        for key, val in self.__dict__.items():
            if key != "step" and val is not None:
                torch.save(val.state_dict(), os.path.join(path, f"{key}.pth"))

    def load(self, path):
        logging.info(f"load from {path}")
        self.step = torch.load(os.path.join(path, "step.pth"))
        for key, val in self.__dict__.items():
            if key != "step" and val is not None:
                val.load_state_dict(
                    torch.load(os.path.join(path, f"{key}.pth"), map_location="cpu")
                )

    def resume(self, ckpt_root, step=None):
        if not os.path.exists(ckpt_root):
            return
        if step is None:
            ckpts = list(filter(lambda x: ".ckpt" in x, os.listdir(ckpt_root)))
            if not ckpts:
                return
            steps = map(lambda x: int(x.split(".")[0]), ckpts)
            step = max(steps)
        ckpt_path = os.path.join(ckpt_root, f"{step}.ckpt")
        logging.info(f"resume from {ckpt_path}")
        self.load(ckpt_path)

    def to(self, device):
        for key, val in self.__dict__.items():
            if isinstance(val, nn.Module):
                val.to(device)


def get_brain_encoder(config, dataset):
    args = OmegaConf.load(config.brain_encoder.config_path)
    # FIXME
    # args = update_with_eval(args)

    subjects = dataset.subject_names if hasattr(dataset, "subject_names") else dataset.num_subjects  # fmt: skip

    if args.dataset.endswith("CLIP"):
        model = BrainEncoder(args, subjects)
    else:
        if args.masked:
            model = BrainMAE(args, subjects, mask_ratio=0)
        else:
            model = BrainAutoencoder(args, subjects)

    # prefix = "brain_encoder" if args.dataset.endswith("CLIP") else "autoencoder"
    # model.load_state_dict(
    #     torch.load(os.path.join(get_run_dir(args), f"{prefix}_best.pt"), map_location="cpu")
    # )
    model.load_state_dict(torch.load(config.brain_encoder.pretrained_path, map_location="cpu"))

    return model


# def get_brain_encoder(config):
#     loc = min_max_norm(np.load(config.dataset.montage_path))
#     loc = torch.from_numpy(loc.astype(np.float32))

#     use_fp16 = config.mixed_precision == "fp16"

#     return BrainEncoder(config.z_shape, loc, use_fp16, **config.brain_encoder.arch)


def initialize_train_state(config, device):
    params = []

    nnet = utils.get_nnet(**config.nnet)
    params += nnet.parameters()
    logging.info(f"nnet has {utils.cnt_params(nnet)} parameters")

    nnet_ema = utils.get_nnet(**config.nnet).eval() if config.train.use_ema else None

    # Build brain encoder
    if config.joint:
        brain_encoder = get_brain_encoder(config)
        params += brain_encoder.parameters()
        logging.info(f"brain_encoder has {utils.cnt_params(brain_encoder)} parameters")
    else:
        brain_encoder = None
        logging.info(f"Not learning brain encoder jointly.")

    optimizer = utils.get_optimizer(params, **config.optimizer)
    lr_scheduler = utils.get_lr_scheduler(optimizer, **config.lr_scheduler)

    train_state = TrainState(
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        step=0,
        nnet=nnet,
        nnet_ema=nnet_ema,
        brain_encoder=brain_encoder,
    )
    train_state.ema_update(0)
    train_state.to(device)

    return train_state


class CLIPMetric:
    def __init__(self, device):
        model, preproc = clip.load("ViT-L/14")

        self.model = model.eval().requires_grad_(False).to(device)
        self.normalize = preproc.transforms.pop()

    def preproc(self, samples):
        samples = TF.resize(samples, 224, interpolation=TF.InterpolationMode.BICUBIC)
        samples = self.normalize(samples)

        return samples

    @torch.no_grad()
    def __call__(self, samples, samples_gt):
        samples = self.model.encode_image(self.preproc(samples))
        samples_gt = self.model.encode_image(self.preproc(samples_gt))

        samples /= samples.norm(dim=-1, keepdim=True)
        samples_gt /= samples_gt.norm(dim=-1, keepdim=True)

        return torch.diagonal(samples @ samples_gt.T)


def sample2dir(accelerator, path, dataloader, config, sample_fn, unpreprocess_fn=None, eval=False):
    if eval:
        os.makedirs(os.path.join(path, "with_gt"), exist_ok=True)
        os.makedirs(os.path.join(path, "grid"), exist_ok=True)

        ssim, clip = [], []
        clip_fn = CLIPMetric(accelerator.device)

    if dataloader is None:
        dataloader = utils.amortize(config.n_samples, config.mini_batch_size)

    idx = 0
    for batch in tqdm(
        dataloader,
        disable=not accelerator.is_main_process,
        desc="sample2dir",
    ):
        samples = unpreprocess_fn(sample_fn(batch))  # ( b=32, c=3, h=256, w=256 )
        _batch_size = len(samples)

        if eval:
            samples_gt = batch[1]

            ssim.append(ssim_fn(samples, samples_gt).item())  # averaged
            clip.append(clip_fn(samples, samples_gt))  # not averaged

            samples_gt = torch.cat([samples_gt, samples], dim=2)  # ( b=32, c=3, h=512, w=256 )
            samples_gt = accelerator.gather(samples_gt.contiguous())[:_batch_size]

        samples = accelerator.gather(samples.contiguous())[:_batch_size]

        if accelerator.is_main_process:
            for i in range(_batch_size):
                save_image(samples[i], os.path.join(path, f"{idx}.png"))
                if eval:
                    save_image(samples_gt[i], os.path.join(path, "with_gt", f"{idx}.png"))

                idx += 1

    def save_grid(idxs, tag: str):
        images = [read_image(os.path.join(path, "with_gt", f"{idx}.png")) for idx in idxs]
        images = torch.cat(images, dim=-1).to(torch.float32) / 255

        save_image(images, os.path.join(path, "grid", f"{tag}.png"))

    if eval:
        clip = torch.cat(clip)

        if accelerator.is_main_process:
            _, clip_idxs = clip.sort(descending=True)

            save_grid(clip_idxs[:12], "best")
            save_grid(clip_idxs[len(clip_idxs) // 2 - 6 : len(clip_idxs) // 2 + 6], "average")
            save_grid(clip_idxs[-12:], "worst")

        return np.mean(ssim), clip.mean().item()


class Observe(Schedule):
    def __init__(
        self,
        t_obs: int,
        linear_start: float = 0.00085,
        linear_end: float = 0.012,
        T: int = 1000,
    ):
        super().__init__(linear_start, linear_end, T)

        self.t_obs = t_obs
        self._betas_obs = self._betas[:t_obs]  # + 1]

    def sample_obs(self, x_0: torch.Tensor):
        t = np.full(len(x_0), self.t_obs)

        eps = torch.randn_like(x_0)

        return self._sample(t, x_0, eps)


class Interpolate(Schedule):
    def __init__(
        self,
        t_obs: int,
        linear_start: float = 0.00085,
        linear_end: float = 0.012,
        T: int = 1000,
    ):
        super().__init__(linear_start, linear_end, T)

        self.t_obs = t_obs
        self._betas_obs = self._betas[:t_obs]  # + 1]

        assert np.array_equal(self.alphas.cumprod(), self.cum_alphas)
        self.cum_alphas_end = np.flip(np.flip(self.alphas).cumprod())

    def sample(self, x_0: torch.Tensor, x_obs: Optional[torch.Tensor] = None):
        """sample from q(x_t|x_0, x_obs), where t is uniform
        Args:
            x_0 ( b, c, h, w ): _description_
            x_obs ( b * obs_ratio, c, h, w ): _description_
        Returns:
            _type_: _description_
        """
        t = np.random.choice(list(range(1, self.T + 1)), (x_0.shape[0],))
        eps = torch.randn_like(x_0)

        if x_obs is not None:
            b_obs = x_obs.shape[0]
            x_obs = self.rescale(x_obs, x_0[:b_obs])
            t_obs = np.full(b_obs, self.t_obs)

            eps_obs = self._stp(
                1 / self.cum_betas[t_obs] ** 0.5,
                x_obs - self._stp(self.cum_alphas[t_obs] ** 0.5, x_0[:b_obs]),
            )
            # cprint(f"Mean: {eps_obs.mean()}, std: {eps_obs.std()}", "yellow")
            eps[:b_obs], t[:b_obs] = eps_obs, t_obs

            x_t = torch.cat([x_obs, self._sample(t[b_obs:], x_0[b_obs:], eps[b_obs:])])
        else:
            x_t = self._sample(t, x_0, eps)

        return torch.tensor(t, device=x_0.device), eps, x_t

    def rescale(self, x_obs, x_0):
        b, c, h, w = x_obs.shape
        t_obs = np.full(b, self.t_obs)
        x_obs, x_0 = x_obs.view(b, -1), x_0.view(b, -1)

        # normalize
        x_obs = (x_obs - x_obs.mean(dim=1, keepdim=True)) / x_obs.std(dim=1, keepdim=True)

        # inverse normalize
        x_obs = self._stp(self.cum_betas[t_obs] ** 0.5, x_obs)
        x_obs += self._stp(self.cum_alphas[t_obs] ** 0.5, x_0.mean(dim=1, keepdim=True))

        return x_obs.view(b, c, h, w)


# class BrainEncoder(nn.Module):
#     def __init__(
#         self,
#         out_dims: Tuple[int],
#         loc: torch.Tensor,
#         use_fp16: bool,
#         seq_len: int,
#         depth: int = 2,
#         D1: int = 270,
#         D2: int = 320,
#         D3: int = 2048,
#         K: int = 32,
#         n_heads: int = 4,
#         depthwise_ksize: int = 31,
#         pos_enc_type: str = "abs",
#         d_drop: float = 0.1,
#         p_drop: float = 0.1,
#     ):
#         super().__init__()

#         self.out_dims = out_dims  # ( 4, 32, 32 )

#         self.spatial_attention = nn.Sequential(
#             SpatialAttention(loc, D1, K, d_drop),
#             nn.Conv1d(D1, D1, kernel_size=1, stride=1),
#         )

#         self.blocks = nn.Sequential()
#         for k in range(depth):
#             self.blocks.add_module(
#                 f"block{k}",
#                 ConformerBlock(
#                     k, D1, D2, n_heads, depthwise_ksize, pos_enc_type=pos_enc_type, p_drop=p_drop, use_fp16=use_fp16  # fmt: skip
#                 ),
#                 # TransformerBlock(k, D1, D2, n_heads, seq_len, "sine_abs"),
#             )

#         self.conv_final = nn.Conv1d(D2, D3, kernel_size=1, stride=1)

#         self.temporal_aggregation = TemporalAggregation(seq_len, D3)

#         self.clip_head = MLPHead(in_dim=D3, out_dim=768)
#         self.mse_head = MLPHead(in_dim=D3, out_dim=np.prod(out_dims))

#     def forward(self, X: torch.Tensor):
#         X = self.spatial_attention(X)

#         X = self.blocks(X)  # ( b, D2, t )
#         X = F.gelu(self.conv_final(X))  # ( b, D3, t )

#         X = self.temporal_aggregation(X)  # ( b, D3, 1 )

#         X = self.mse_head(X)  # ( b, 1, 4096 )

#         return X.reshape(X.shape[0], *self.out_dims)
