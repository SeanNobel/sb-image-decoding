import os, sys
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
import ml_collections
from pathlib import Path
from absl import logging
from typing import Tuple, Dict, Optional
from termcolor import cprint
from tqdm import tqdm
import gc

from uvit import utils

from nd.models.brain_encoder import SpatialAttention, ConformerBlock, TransformerBlock
from nd.datasets.things_meg import ThingsCLIPDatasetBase
from nd.utils.layout import min_max_norm


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


def stable_diffusion_beta_schedule(linear_start, linear_end, n_timestep) -> np.ndarray:
    _betas = torch.linspace(linear_start**0.5, linear_end**0.5, n_timestep, dtype=torch.float64)
    _betas **= 2

    return _betas.numpy()


def stp(s, ts: torch.Tensor):  # scalar tensor product
    if isinstance(s, np.ndarray):
        s = torch.from_numpy(s).type_as(ts)
    extra_dims = (1,) * (ts.dim() - 1)
    return s.view(-1, *extra_dims) * ts


def get_skip(alphas, betas):
    N = len(betas) - 1

    skip_alphas = np.ones([N + 1, N + 1], dtype=betas.dtype)
    for s in range(N + 1):
        skip_alphas[s, s + 1 :] = alphas[s + 1 :].cumprod()

    skip_betas = np.zeros([N + 1, N + 1], dtype=betas.dtype)
    for t in range(N + 1):
        prod = betas[1 : t + 1] * skip_alphas[1 : t + 1, t]
        skip_betas[:t, t] = (prod[::-1].cumsum())[::-1]

    return skip_alphas, skip_betas


class Schedule(object):  # discrete time
    def __init__(
        self,
        t_obs: int,
        linear_start: float = 0.00085,
        linear_end: float = 0.012,
        T: int = 1000,
    ):
        r"""_betas[0...999] = betas[1...1000]
        for n>=1, betas[n] is the variance of q(xn|xn-1)
        for n=0,  betas[0]=0
        """
        self._betas = stable_diffusion_beta_schedule(linear_start, linear_end, T)
        self.betas = np.append(0.0, self._betas)
        self.alphas = 1.0 - self.betas
        self.T = len(self._betas)

        self.t_obs = t_obs
        self._betas_obs = self._betas[:t_obs]  # + 1]

        assert isinstance(self.betas, np.ndarray) and self.betas[0] == 0
        assert isinstance(self.alphas, np.ndarray) and self.alphas[0] == 1
        assert len(self.betas) == len(self.alphas)

        # skip_alphas[s, t] = alphas[s + 1: t + 1].prod()
        self.skip_alphas, self.skip_betas = get_skip(self.alphas, self.betas)
        self.cum_alphas = self.skip_alphas[0]  # cum_alphas = alphas.cumprod()
        self.cum_betas = self.skip_betas[0]
        self.snr = self.cum_alphas / self.cum_betas

    def tilde_beta(self, s, t):
        return self.skip_betas[s, t] * self.cum_betas[s] / self.cum_betas[t]

    def sample(self, x_0: torch.Tensor):
        t = np.random.choice(list(range(1, self.T + 1)), (len(x_0),))

        eps = torch.randn_like(x_0)

        x_t = stp(self.cum_alphas[t] ** 0.5, x_0) + stp(self.cum_betas[t] ** 0.5, eps)

        return torch.tensor(t, device=x_0.device), eps, x_t

    def sample_obs(self, x_0: torch.Tensor):
        t = np.full(len(x_0), self.t_obs)

        eps = torch.randn_like(x_0)

        return stp(self.cum_alphas[t] ** 0.5, x_0) + stp(self.cum_betas[t] ** 0.5, eps)

    def __repr__(self):
        return f"Schedule({self.betas[:10]}..., {self.T})"


class Interpolate(object):  # discrete time
    """_betas[0...999] = betas[1...1000]
        for t>=1, betas[t] is the variance of q(x_t | x_t - 1)
        for t=0,  betas[0]=0
    Args:
        object (_type_): _description_
    """

    def __init__(
        self,
        t_obs: int,
        linear_start: float = 0.00085,
        linear_end: float = 0.012,
        T: int = 1000,
    ):
        self._betas = stable_diffusion_beta_schedule(linear_start, linear_end, T)
        self.betas = np.append(0.0, self._betas)
        self.alphas = 1.0 - self.betas
        self.T = len(self._betas)

        self.t_obs = t_obs
        self._betas_obs = self._betas[:t_obs]  # + 1]

        assert isinstance(self.betas, np.ndarray) and self.betas[0] == 0
        assert isinstance(self.alphas, np.ndarray) and self.alphas[0] == 1
        assert len(self.betas) == len(self.alphas)

        self.skip_alphas, self.skip_betas = get_skip(self.alphas, self.betas)
        self.cum_alphas = self.skip_alphas[0]  # cum_alphas = alphas.cumprod()
        self.cum_betas = self.skip_betas[0]
        self.snr = self.cum_alphas / self.cum_betas

        assert np.array_equal(self.alphas.cumprod(), self.cum_alphas)
        self.cum_alphas_end = np.flip(np.flip(self.alphas).cumprod())

    def tilde_beta(self, s, t):
        return self.skip_betas[s, t] * self.cum_betas[s] / self.cum_betas[t]

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

            eps_obs = stp(
                1 / self.cum_betas[t_obs] ** 0.5,
                x_obs - stp(self.cum_alphas[t_obs] ** 0.5, x_0[:b_obs]),
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
        x_obs = stp(self.cum_betas[t_obs] ** 0.5, x_obs)
        x_obs += stp(self.cum_alphas[t_obs] ** 0.5, x_0.mean(dim=1, keepdim=True))

        return x_obs.view(b, c, h, w)

    def _sample(self, t: int, x_0: torch.Tensor, eps: torch.Tensor):
        return stp(self.cum_alphas[t] ** 0.5, x_0) + stp(self.cum_betas[t] ** 0.5, eps)

    def __repr__(self):
        return f"Schedule({self.betas[:10]}..., {self.T})"

    # def interpolate(self, x_0: torch.Tensor, x_T: torch.Tensor):
    #     t = np.random.choice(list(range(1, self.T + 1)), (len(x_0),))
    #     eps = torch.randn_like(x_0)
    #     x_t = self._sample(t, x_0, eps)
    #     cum_alphas_end = torch.from_numpy(self.cum_alphas_end[t]).type_as(x_T)
    #     x_t_rev = (x_T - stp((1.0 - cum_alphas_end) ** 0.5, eps)) / (cum_alphas_end ** 0.5)[:, None, None, None]  # fmt: skip
    #     x_t = (x_t + x_t_rev) / 2.0
    #     return torch.tensor(t, device=x_0.device), eps, x_t


class ThingsMEGMomentsDataset(ThingsCLIPDatasetBase):
    def __init__(self, args: ml_collections.FrozenConfigDict) -> None:
        super().__init__()

        self.preproc_dir = args.path
        self.large_test_set = args.large_test_set
        self.num_subjects = 4

        sample_attrs_paths = [
            os.path.join(args.thingsmeg_dir, f"sourcedata/sample_attributes_P{i+1}.csv")
            for i in range(self.num_subjects)
        ]

        subject_idxs_list = []
        categories_list = []
        y_idxs_list = []
        train_idxs_list = []
        test_idxs_list = []
        for subject_id, sample_attrs_path in enumerate(sample_attrs_paths):
            # Indexes
            sample_attrs = np.loadtxt(
                sample_attrs_path, dtype=str, delimiter=",", skiprows=1
            )  # ( 27048, 18 )

            categories_list.append(torch.from_numpy(sample_attrs[:, 2].astype(int)))
            y_idxs_list.append(torch.from_numpy(sample_attrs[:, 1].astype(int)))

            subject_idxs_list.append(torch.ones(len(sample_attrs), dtype=int) * subject_id)

            # Split
            train_idxs, test_idxs = self.make_split(
                sample_attrs, large_test_set=self.large_test_set
            )
            idx_offset = len(sample_attrs) * subject_id
            train_idxs_list.append(train_idxs + idx_offset)
            test_idxs_list.append(test_idxs + idx_offset)

        assert len(set([len(s) for s in subject_idxs_list])) == 1
        self.num_samples = len(subject_idxs_list[0])

        self.subject_idxs = torch.cat(subject_idxs_list, dim=0)

        self.categories = torch.cat(categories_list) - 1
        assert torch.equal(self.categories.unique(), torch.arange(self.categories.max() + 1))  # fmt: skip
        self.num_categories = len(self.categories.unique())

        self.y_idxs = torch.cat(y_idxs_list) - 1
        assert torch.equal(self.y_idxs.unique(), torch.arange(self.y_idxs.max() + 1))

        self.train_idxs = torch.cat(train_idxs_list, dim=0)
        self.test_idxs = torch.cat(test_idxs_list, dim=0)

        self.vis_samples: Dict[str, torch.Tensor] = self._load_vis_samples(args.n_vis_samples)

        cprint(f"X, Y: loaded in __getitem__ | subject_idxs: {self.subject_idxs.shape} | train_idxs: {self.train_idxs.shape} | test_idxs: {self.test_idxs.shape}", "cyan")  # fmt: skip

        del categories_list, y_idxs_list, subject_idxs_list, train_idxs_list, test_idxs_list  # fmt: skip
        gc.collect()

    def _load_vis_samples(self, num_vis_samples: int) -> Dict[str, torch.Tensor]:
        # fmt: off
        return {
            "train_brain": torch.stack([self._load_sample(i, sample_type="MEG") for i in self.train_idxs[:num_vis_samples]]),
            "train_moments": torch.stack([self._load_sample(i, sample_type="Image_moments") for i in self.train_idxs[:num_vis_samples]]),
            "test_brain": torch.stack([self._load_sample(i, sample_type="MEG") for i in self.test_idxs[:num_vis_samples]]),
            "test_moments": torch.stack([self._load_sample(i, sample_type="Image_moments") for i in self.test_idxs[:num_vis_samples]]),
        }
        # fmt: on

    def unpreprocess(self, v: torch.Tensor):
        return (0.5 * (v + 1.0)).clamp(0.0, 1.0)

    @property
    def data_shape(self):
        return 4, 32, 32

    @property
    def fid_stat(self):
        return os.path.join(self.preproc_dir, "fid_stats_thingsmeg_test.npz")

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        X = self._load_sample(i, sample_type="MEG")
        Y = self._load_sample(i, sample_type="Image_moments")

        return X, Y, self.subject_idxs[i], self.y_idxs[i], self.categories[i]


class BrainEncoder(nn.Module):
    def __init__(
        self,
        out_dims: Tuple[int],
        loc: torch.Tensor,
        use_fp16: bool,
        seq_len: int,
        depth: int = 2,
        D1: int = 270,
        D2: int = 64,
        K: int = 32,
        n_heads: int = 4,
        depthwise_ksize: int = 31,
        pos_enc_type: str = "abs",
        d_drop: float = 0.1,
        p_drop: float = 0.1,
    ):
        super().__init__()

        self.out_dims = out_dims  # ( 4, 32, 32 )

        self.spatial_attention = SpatialAttention(loc, D1, K, d_drop)
        self.conv = nn.Conv1d(D1, D1, kernel_size=1, stride=1)

        self.blocks = nn.Sequential()
        for k in range(depth):
            self.blocks.add_module(
                f"block{k}",
                ConformerBlock(
                    k, D1, D2, n_heads, depthwise_ksize, pos_enc_type=pos_enc_type, p_drop=p_drop, use_fp16=use_fp16  # fmt: skip
                ),
                # TransformerBlock(k, D1, D2, n_heads, seq_len, "sine_abs"),
            )

        self.out_proj = nn.Linear(seq_len, np.prod(out_dims) // D2)

    def forward(self, X: torch.Tensor):
        X = self.spatial_attention(X)
        X = self.conv(X)

        X = self.blocks(X)  # ( b, D2, t )

        X = self.out_proj(X)  # ( b, D2, t' )

        return X.reshape(X.shape[0], *self.out_dims)


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


def get_brain_encoder(config):
    loc = min_max_norm(np.load(config.dataset.montage_path))
    loc = torch.from_numpy(loc.astype(np.float32))

    use_fp16 = config.mixed_precision == "fp16"

    return BrainEncoder(config.z_shape, loc, use_fp16, **config.brain_encoder)


def initialize_train_state(config, device):
    params = []

    nnet = utils.get_nnet(**config.nnet)
    params += nnet.parameters()
    nnet_ema = utils.get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f"nnet has {utils.cnt_params(nnet)} parameters")

    # Build brain encoder
    if config.brain_encoder.joint:
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


def sample2dir(accelerator, path, dataloader, config, sample_fn, unpreprocess_fn=None):
    os.makedirs(path, exist_ok=True)
    idx = 0

    if dataloader is None:
        dataloader = utils.amortize(
            config.n_samples, config.mini_batch_size * accelerator.num_processes
        )

    for batch in tqdm(
        dataloader,
        disable=not accelerator.is_main_process,
        desc="sample2dir",
    ):
        samples = unpreprocess_fn(sample_fn(batch))

        _batch_size = len(samples)
        samples = accelerator.gather(samples.contiguous())[:_batch_size]
        if accelerator.is_main_process:
            for sample in samples:
                save_image(sample, os.path.join(path, f"{idx}.png"))
                idx += 1
