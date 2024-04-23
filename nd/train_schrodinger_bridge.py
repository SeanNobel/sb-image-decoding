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
from typing import Optional

from uvit import libs, utils
from uvit.dpm_solver_pp import NoiseScheduleVP, DPM_Solver

from uvit.tools.fid_score import calculate_fid_given_paths

from nd.utils.uvit_utils import (
    Bridge,
    ThingsMEGMomentsDataset,
    initialize_train_state,
    get_brain_encoder,
    get_config_name,
    get_hparams,
    sample2dir,
)


def p_losses(x_0, x_T, nnet, schedule: Bridge, **kwargs):
    """_summary_
    Args:
        x_0 ( b, c, h, w ): _description_
        x_T ( b, c, h, w ): _description_
        nnet (_type_): _description_
        schedule (_type_): _description_
    Returns:
        loss ( b, ): _description_
    """
    t, eps, x_t = schedule.sample(x_0, x_T)

    eps_pred = nnet(x_t, t, **kwargs)

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
            project="schrodinger_bridge",
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

    loader_args = {"num_workers": 8, "pin_memory": True, "persistent_workers": True}
    train_loader = DataLoader(
        train_set, batch_size=mini_batch_size, shuffle=True, drop_last=True, **loader_args
    )
    test_loader = DataLoader(
        test_set, batch_size=config.sample.mini_batch_size, shuffle=False, drop_last=False, **loader_args  # fmt: skip
    )

    train_state = initialize_train_state(config, device)
    assert train_state.brain_encoder is None, "Set joint=False for schrodinger bridge."

    nnet, nnet_ema, optimizer, train_loader, test_loader = accelerator.prepare(
        train_state.nnet,
        train_state.nnet_ema,
        train_state.optimizer,
        train_loader,
        test_loader,
    )
    lr_scheduler = train_state.lr_scheduler
    train_state.resume(config.ckpt_root)

    autoencoder = libs.autoencoder.get_model(config.autoencoder.pretrained_path)
    autoencoder.to(device)

    brain_encoder = get_brain_encoder(config)
    brain_encoder.load_state_dict(
        torch.load(config.brain_encoder.pretrained_path, map_location="cpu")
    )
    brain_encoder.eval().to(device)
    brain_encoder.requires_grad_(False)

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

    schedule = Bridge()
    logging.info(f"use {schedule}")

    for split in ["train", "test"]:
        samples_gt = decode(dataset.vis_samples[f"{split}_moments"].to(device))
        samples_gt = make_grid(dataset.unpreprocess(samples_gt), config.dataset.n_vis_samples // 2)
        wandb.log({f"{split}_samples_gt": wandb.Image(samples_gt)})

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

        x_T = brain_encoder(_batch[0])  # ( b, 4, 32, 32 )

        loss = p_losses(x_0, x_T, nnet, schedule)

        _metrics = dict()
        _metrics["p_loss"] = accelerator.gather(loss.detach()).mean()

        accelerator.backward(loss.mean())
        optimizer.step()
        lr_scheduler.step()

        train_state.ema_update(config.get("ema_rate", 0.9999))
        train_state.step += 1

        return dict(lr=train_state.optimizer.param_groups[0]["lr"], **_metrics)

    def dpm_solver_sample(x_init: torch.Tensor, _sample_steps, _betas: np.ndarray, **kwargs):
        noise_schedule = NoiseScheduleVP(
            schedule="discrete", betas=torch.tensor(_betas, device=device).float()
        )

        def model_fn(x, t_continuous):
            t = t_continuous * schedule.T
            eps_pre = nnet_ema(x, t, **kwargs)
            return eps_pre

        dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
        x_0 = dpm_solver.sample(x_init, steps=_sample_steps, eps=1.0 / schedule.T, T=1.0)

        return decode(x_0)

    def eval_step(sample_steps):
        n_samples = len(dataset.test_idxs)

        logging.info(
            f"eval_step: n_samples={n_samples}, sample_steps={sample_steps}"
            f"mini_batch_size={config.sample.mini_batch_size}"
        )

        def sample_fn(batch):
            x_init = brain_encoder(batch[0].to(device))

            if config.train.mode == "uncond":
                kwargs = dict()
            else:
                raise NotImplementedError("Conditional sampling is not implemented yet.")

            return dpm_solver_sample(x_init, sample_steps, schedule._betas, **kwargs)

        with tempfile.TemporaryDirectory() as temp_path:
            path = config.sample.path or temp_path
            if accelerator.is_main_process:
                os.makedirs(path, exist_ok=True)

            sample2dir(
                accelerator, path, test_loader, config.sample, sample_fn, dataset.unpreprocess
            )

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

        batch = tree_map(lambda x: x.to(device), next(data_generator))

        metrics = train_step(batch)

        nnet.eval()

        if accelerator.is_main_process and train_state.step % config.train.log_interval == 0:  # fmt: skip
            logging.info(utils.dct2str(dict(step=train_state.step, **metrics)))
            logging.info(config.workdir)
            wandb.log(metrics, step=train_state.step)

        if train_state.step % config.train.eval_interval == 0:
            # NOTE: training stucks by calling the forward pass only in the main process
            x_eval = {
                split: brain_encoder(dataset.vis_samples[f"{split}_brain"].to(device))
                for split in ["train", "test"]
            }

            if accelerator.is_main_process:
                torch.cuda.empty_cache()
                logging.info("Save a grid of images...")

                for split, x_init in x_eval.items():
                    if config.train.mode == "uncond":
                        samples = dpm_solver_sample(x_init, config.sample.steps, schedule._betas)
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
