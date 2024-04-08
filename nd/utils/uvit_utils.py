import os, sys
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from pathlib import Path
from absl import logging
from hydra import initialize, compose
from omegaconf import open_dict
from typing import Tuple

from uvit import utils

from nd.models.brain_encoder import BrainEncoder


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
        if argv[i].startswith("--config.") and not argv[i].startswith(
            "--config.dataset.path"
        ):
            hparam, val = argv[i].split("=")
            hparam = hparam.split(".")[-1]
            if hparam.endswith("path"):
                val = Path(val).stem
            lst.append(f"{hparam}={val}")
    hparams = "-".join(lst)
    if hparams == "":
        hparams = "default"
    return hparams


class BrainEncoderKL(BrainEncoder):
    def __init__(self, args, dims: Tuple[int], mid_dim=128):
        with open_dict(args):
            args.D2 = mid_dim

        super().__init__(args, subjects=1)

        self.dims = dims  # ( 4, 64, 64 )

        self.out_proj = nn.Linear(self.init_temporal_dim, np.prod(dims) // mid_dim)

    def forward(self, X: torch.Tensor):
        X = self.subject_block(X, torch.zeros(X.shape[0]))

        X = self.blocks(X)  # ( b, D2, t )

        X = self.out_proj(X)  # ( b, D2, t' )
        X = rearrange(
            X, "b d2 t -> b c h w", c=self.dims[0], h=self.dims[1], w=self.dims[2]
        )
        return X


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


def initialize_train_state(config, device):
    params = []

    nnet = utils.get_nnet(**config.nnet)
    params += nnet.parameters()
    nnet_ema = utils.get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f"nnet has {utils.cnt_params(nnet)} parameters")

    with initialize(version_base=None, config_path="../../configs/thingsmeg/"):
        args = compose(config_name="clip")
        brain_encoder = BrainEncoderKL(args, config.z_shape)

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
