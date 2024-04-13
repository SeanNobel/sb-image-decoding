import os, sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils._pytree import tree_map
from torchvision.utils import make_grid, save_image
from torch import multiprocessing as mp
import ml_collections
import accelerate
import wandb
from tqdm.auto import tqdm
import tempfile
from absl import logging
import builtins
from termcolor import cprint

from uvit import libs, utils
from uvit.dpm_solver_pp import NoiseScheduleVP, DPM_Solver
from uvit.tools.fid_score import calculate_fid_given_paths

from nd.datasets.things_meg import ThingsMEGMomentsDataset
from nd.utils.uvit_utils import initialize_train_state, get_config_name, get_hparams, sample2dir


class Interpolate(object):  # discrete time
    r"""_betas[0...999] = betas[1...1000]
    for n>=1, betas[n] is the variance of q(xn|xn-1)
    for n=0,  betas[0]=0
    """

    def __init__(self, t_obs: int, linear_start=0.00085, linear_end=0.012, n_timestep=1000):
        self._betas = self.stable_diffusion_beta_schedule(linear_start, linear_end, n_timestep)
        self.betas = np.append(0.0, self._betas)
        self.alphas = 1.0 - self.betas
        self.T = len(self._betas)

        self.t_obs = t_obs
        self._betas_obs = self._betas[:t_obs]  # + 1]

        assert isinstance(self.betas, np.ndarray) and self.betas[0] == 0
        assert isinstance(self.alphas, np.ndarray) and self.alphas[0] == 1
        assert len(self.betas) == len(self.alphas)

        # skip_alphas[s, t] = alphas[s + 1: t + 1].prod()
        self.skip_alphas, self.skip_betas = self._get_skip(self.alphas, self.betas)
        self.cum_alphas = self.skip_alphas[0]  # cum_alphas = alphas.cumprod()
        self.cum_betas = self.skip_betas[0]
        self.snr = self.cum_alphas / self.cum_betas

        assert np.array_equal(self.alphas.cumprod(), self.cum_alphas)
        self.cum_alphas_end = np.flip(np.flip(self.alphas).cumprod())

    @staticmethod
    def stable_diffusion_beta_schedule(linear_start, linear_end, n_timestep) -> np.ndarray:
        _betas = (
            torch.linspace(linear_start**0.5, linear_end**0.5, n_timestep, dtype=torch.float64) ** 2
        )
        return _betas.numpy()

    def tilde_beta(self, s, t):
        return self.skip_betas[s, t] * self.cum_betas[s] / self.cum_betas[t]

    @staticmethod
    def _stp(s, ts: torch.Tensor):  # scalar tensor product
        if isinstance(s, np.ndarray):
            s = torch.from_numpy(s).type_as(ts)
        extra_dims = (1,) * (ts.dim() - 1)
        return s.view(-1, *extra_dims) * ts

    def interpolate(self, x_0: torch.Tensor, x_obs: torch.Tensor):
        """sample from q(x_t|x_0, x_obs), where t is uniform
        Args:
            x_0 (torch.Tensor): _description_
            x_obs (torch.Tensor): _description_
        Returns:
            _type_: _description_
        """
        b = x_0.shape[0]
        t = np.random.choice(list(range(1, self.t_obs + 1)), (b,))

        x_obs = self.rescale(x_obs, x_0)
        t_obs = np.full(b, self.t_obs)
        eps = x_obs - self._stp(self.cum_alphas[t_obs] ** 0.5, x_0)
        eps = self._stp(1 / self.cum_betas[t_obs] ** 0.5, eps)

        x_t = self._sample(t, x_0, eps)

        return torch.tensor(t, device=x_0.device), eps, x_t

    def rescale(self, x_obs, x_0):
        b, c, h, w = x_obs.shape

        x_obs, x_0 = x_obs.view(b, -1), x_0.view(b, -1)

        # normalize sample-wise
        x_obs = (x_obs - x_obs.mean(dim=1, keepdim=True)) / x_obs.std(dim=1, keepdim=True)
        # inverse normalize sample-wise
        t_obs = np.full(b, self.t_obs)
        x_obs = self._stp(self.cum_betas[t_obs] ** 0.5, x_obs) + self._stp(self.cum_alphas[t_obs] ** 0.5, x_0)  # fmt: skip

        return x_obs.view(b, c, h, w)

    def _sample(self, t: int, x_0: torch.Tensor, eps: torch.Tensor):
        return self._stp(self.cum_alphas[t] ** 0.5, x_0) + self._stp(self.cum_betas[t] ** 0.5, eps)

    def __repr__(self):
        return f"Schedule({self.betas[:10]}..., {self.T})"

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

    # def interpolate(self, x_0: torch.Tensor, x_T: torch.Tensor):
    #     t = np.random.choice(list(range(1, self.T + 1)), (len(x_0),))
    #     eps = torch.randn_like(x_0)
    #     x_t = self._sample(t, x_0, eps)
    #     cum_alphas_end = torch.from_numpy(self.cum_alphas_end[t]).type_as(x_T)
    #     x_t_rev = (x_T - self._stp((1.0 - cum_alphas_end) ** 0.5, eps)) / (cum_alphas_end ** 0.5)[:, None, None, None]  # fmt: skip
    #     x_t = (x_t + x_t_rev) / 2.0
    #     return torch.tensor(t, device=x_0.device), eps, x_t


def p_losses(x_0, x_obs, nnet, schedule, **kwargs):
    n, eps, x_n = schedule.interpolate(x_0, x_obs)  # n in {1, ..., T}

    eps_pred = nnet(x_n, n, **kwargs)

    return (eps - eps_pred).pow(2).flatten(start_dim=1).mean(dim=-1)


def train(config):
    if config.get("benchmark", False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    mp.set_start_method("spawn", force=True)
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f"Process {accelerator.process_index} using device: {device}")

    config.mixed_precision = accelerator.mixed_precision
    config = ml_collections.FrozenConfigDict(config)

    assert config.train.batch_size % accelerator.num_processes == 0
    mini_batch_size = config.train.batch_size // accelerator.num_processes

    if accelerator.is_main_process:
        os.makedirs(config.ckpt_root, exist_ok=True)
        os.makedirs(config.sample_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        wandb.init(
            dir=os.path.abspath(config.workdir),
            project="brain_denoiser",
            config=config.to_dict(),
            name=config.hparams,
            job_type="train",
            mode=config.wandb_mode,
        )
        utils.set_logger(log_level="info", fname=os.path.join(config.workdir, "output.log"))
        logging.info(config)
    else:
        utils.set_logger(log_level="error")
        builtins.print = lambda *args: None
    logging.info(f"Run on {accelerator.num_processes} devices")

    dataset = ThingsMEGMomentsDataset(config.dataset)
    assert os.path.exists(dataset.fid_stat)

    train_set = torch.utils.data.Subset(dataset, dataset.train_idxs)
    test_set = torch.utils.data.Subset(dataset, dataset.test_idxs)

    loader_args = {
        "num_workers": 8,
        "pin_memory": True,
        "persistent_workers": True,
    }
    train_loader = DataLoader(
        train_set, batch_size=mini_batch_size, shuffle=True, drop_last=True, **loader_args
    )
    test_loader = DataLoader(
        test_set, batch_size=config.sample.mini_batch_size, shuffle=False, drop_last=False, **loader_args  # fmt: skip
    )

    train_state = initialize_train_state(config, device)

    nnet, nnet_ema, brain_encoder, optimizer, train_loader, test_loader = accelerator.prepare(
        train_state.nnet,
        train_state.nnet_ema,
        train_state.brain_encoder,
        train_state.optimizer,
        train_loader,
        test_loader,
    )
    lr_scheduler = train_state.lr_scheduler
    train_state.resume(config.ckpt_root)

    autoencoder = libs.autoencoder.get_model(config.autoencoder.pretrained_path)
    autoencoder.to(device)

    @torch.cuda.amp.autocast()
    def encode(_batch):
        return autoencoder.encode(_batch)

    @torch.cuda.amp.autocast()
    def decode(_batch):
        return autoencoder.decode(_batch)

    def get_data_generator():
        while True:
            for data in tqdm(train_loader, disable=not accelerator.is_main_process, desc="epoch"):
                yield data

    data_generator = get_data_generator()

    schedule = Interpolate(config.t_obs)
    logging.info(f"use {schedule}")

    def train_step(_batch):
        """_summary_
        Args:
            _batch[0] ( b, c, t ): MEG
            _batch[1] ( b, c, h, w ): Image moments
            _batch[2] ( b, ): Subject idxs
            _batch[3] ( b, ): Image idxs in whole dataset
            _batch[4] ( b, ): Classes of the images
        Returns:
            _type_: _description_
        """
        optimizer.zero_grad()

        x_0 = autoencoder.sample(_batch[1])  # ( b, 4, 32, 32 )
        x_obs = brain_encoder(_batch[0])  # ( b, 4, 32, 32 )

        loss = p_losses(x_0, x_obs, nnet, schedule)

        _metrics = dict()
        _metrics["loss"] = accelerator.gather(loss.detach()).mean()

        accelerator.backward(loss.mean())
        optimizer.step()
        lr_scheduler.step()

        train_state.ema_update(config.get("ema_rate", 0.9999))
        train_state.step += 1

        return dict(lr=train_state.optimizer.param_groups[0]["lr"], **_metrics)

    def dpm_solver_sample(_z_init: torch.Tensor, _sample_steps, **kwargs):
        noise_schedule = NoiseScheduleVP(
            schedule="discrete",
            betas=torch.tensor(schedule._betas_obs, device=device).float(),
        )

        def model_fn(x, t_continuous):
            t = t_continuous * schedule.t_obs
            eps_pre = nnet_ema(x, t, **kwargs)
            return eps_pre

        dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
        _z = dpm_solver.sample(_z_init, steps=_sample_steps, eps=1.0 / schedule.t_obs, T=1.0)
        return decode(_z)

    def eval_step(sample_steps):
        n_samples = len(dataset.test_idxs)

        logging.info(
            f"eval_step: n_samples={n_samples}, sample_steps={sample_steps}"
            f"mini_batch_size={config.sample.mini_batch_size}"
        )

        def sample_fn(batch):
            with torch.no_grad():
                x_obs = brain_encoder(batch[0].to(device))
                x_0 = autoencoder.sample(batch[1].to(device))
                x_obs = schedule.rescale(x_obs, x_0)

            if config.train.mode == "uncond":
                kwargs = dict()
            else:
                raise NotImplementedError("Conditional sampling is not implemented yet.")

            return dpm_solver_sample(x_obs, sample_steps, **kwargs)

        with tempfile.TemporaryDirectory() as temp_path:
            path = config.sample.path or temp_path
            if accelerator.is_main_process:
                os.makedirs(path, exist_ok=True)

            sample2dir(accelerator, path, test_loader, sample_fn, dataset.unpreprocess)

            _fid = 0
            if accelerator.is_main_process:
                _fid = calculate_fid_given_paths((dataset.fid_stat, path))
                logging.info(f"step={train_state.step} fid{n_samples}={_fid}")

                with open(os.path.join(config.workdir, "eval.log"), "a") as f:
                    print(f"step={train_state.step} fid{n_samples}={_fid}", file=f)

                wandb.log({f"fid{n_samples}": _fid}, step=train_state.step)

            _fid = torch.tensor(_fid, device=device)
            _fid = accelerator.reduce(_fid, reduction="sum")

        return _fid.item()

    logging.info(f"Start fitting, step={train_state.step}, mixed_precision={config.mixed_precision}")  # fmt: skip

    step_fid = []
    while train_state.step < config.train.n_steps:
        nnet.train()
        brain_encoder.train()

        batch = tree_map(lambda x: x.to(device), next(data_generator))

        metrics = train_step(batch)

        nnet.eval()
        brain_encoder.eval()

        if accelerator.is_main_process and train_state.step % config.train.log_interval == 0:  # fmt: skip
            logging.info(utils.dct2str(dict(step=train_state.step, **metrics)))
            logging.info(config.workdir)
            wandb.log(metrics, step=train_state.step)

        if train_state.step % config.train.eval_interval == 0:
            # NOTE: training stucks by calling the forward pass only in the main process
            with torch.no_grad():
                x_obs_train = brain_encoder(dataset.vis_samples["train_brain"].to(device))
                x_0_train = autoencoder.sample(dataset.vis_samples["train_moments"].to(device))
                x_obs_train = schedule.rescale(x_obs_train, x_0_train)

                x_obs_test = brain_encoder(dataset.vis_samples["test_brain"].to(device))
                x_0_test = autoencoder.sample(dataset.vis_samples["test_moments"].to(device))
                x_obs_test = schedule.rescale(x_obs_test, x_0_test)

            if accelerator.is_main_process:
                torch.cuda.empty_cache()
                logging.info("Save a grid of images...")

                for split, z_obs in zip(["train", "test"], [x_obs_train, x_obs_test]):
                    if config.train.mode == "uncond":
                        samples = dpm_solver_sample(z_obs, _sample_steps=config.sample.steps)
                    else:
                        raise NotImplementedError("Conditional sampling is not implemented yet.")

                    samples = make_grid(
                        dataset.unpreprocess(samples), config.dataset.n_vis_samples // 2
                    )
                    save_image(
                        samples, os.path.join(config.sample_dir, f"{train_state.step}_{split}.png")
                    )
                    wandb.log({f"{split}_samples": wandb.Image(samples)}, step=train_state.step)

                    torch.cuda.empty_cache()

        accelerator.wait_for_everyone()

        if train_state.step % config.train.save_interval == 0 or train_state.step == config.train.n_steps:  # fmt: skip
            torch.cuda.empty_cache()
            logging.info(f"Save and eval checkpoint {train_state.step}...")

            if accelerator.local_process_index == 0:
                train_state.save(os.path.join(config.ckpt_root, f"{train_state.step}.ckpt"))

            accelerator.wait_for_everyone()

            # calculate fid of the saved checkpoint
            fid = eval_step(sample_steps=config.sample.steps)

            step_fid.append((train_state.step, fid))
            torch.cuda.empty_cache()

        accelerator.wait_for_everyone()

    logging.info(f"Finish fitting, step={train_state.step}")
    logging.info(f"step_fid: {step_fid}")

    step_best = sorted(step_fid, key=lambda x: x[1])[0][0]
    logging.info(f"step_best: {step_best}")

    train_state.load(os.path.join(config.ckpt_root, f"{step_best}.ckpt"))

    del metrics
    accelerator.wait_for_everyone()

    eval_step(sample_steps=config.sample.sample_steps)


from absl import flags
from absl import app
from ml_collections import config_flags


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])
flags.DEFINE_string("workdir", None, "Work unit directory.")


def main(argv):
    config = FLAGS.config
    config.config_name = get_config_name()
    config.hparams = get_hparams()
    config.workdir = FLAGS.workdir or os.path.join("runs", config.config_name, config.hparams)  # fmt: skip
    config.ckpt_root = os.path.join(config.workdir, "ckpts")
    config.sample_dir = os.path.join(config.workdir, "samples")
    train(config)


if __name__ == "__main__":
    app.run(main)
