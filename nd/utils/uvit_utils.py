import os, sys
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from pathlib import Path
from absl import logging
from typing import Tuple
from termcolor import cprint
from tqdm import tqdm

from uvit import utils

from nd.models.brain_encoder import SpatialAttention, ConformerBlock, TransformerBlock
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
    brain_encoder = get_brain_encoder(config)
    params += brain_encoder.parameters()
    logging.info(f"brain_encoder has {utils.cnt_params(brain_encoder)} parameters")

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


def sample2dir(accelerator, path, dataloader, sample_fn, unpreprocess_fn=None):
    os.makedirs(path, exist_ok=True)
    idx = 0

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
