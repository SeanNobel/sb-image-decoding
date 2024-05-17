import sys
import ml_collections
import torch
from torch.utils.data import DataLoader
from torch import multiprocessing as mp
import accelerate
import tempfile
from absl import logging
import builtins
from einops import repeat
import numpy as np
from tqdm import tqdm
from termcolor import cprint

from uvit import libs, utils
from uvit.tools.fid_score import calculate_fid_given_paths

from nd.datasets.imagenet_eeg import ImageNetEEGEvalDatasetCond
from nd.utils.uvit_utils import Bridge, get_brain_encoder, sample2dir


def evaluate(config):
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

    if accelerator.is_main_process:
        utils.set_logger(log_level="info", fname=config.output_path)
    else:
        utils.set_logger(log_level="error")
        builtins.print = lambda *args: None

    dataset = ImageNetEEGEvalDatasetCond(config.dataset)
    train_set = torch.utils.data.Subset(dataset, dataset.train_idxs)
    test_set = torch.utils.data.Subset(dataset, dataset.test_idxs)

    loader_args = {
        "batch_size": config.sample.mini_batch_size,
        "shuffle": False,
        "drop_last": False,
        "num_workers": 8,
        "pin_memory": True,
        "persistent_workers": True,
    }
    train_loader = DataLoader(train_set, **loader_args)
    test_loader = DataLoader(test_set, **loader_args)

    autoencoder = libs.autoencoder.get_model(config.autoencoder.pretrained_path)
    autoencoder.to(device)

    brain_encoder = get_brain_encoder(config, dataset)
    brain_encoder.eval().to(device).requires_grad_(False)

    nnet = utils.get_nnet(**config.nnet)

    nnet, brain_encoder, train_loader, test_loader = accelerator.prepare(
        nnet, brain_encoder, train_loader, test_loader
    )

    logging.info(f"load nnet from {config.nnet_path}")
    accelerator.unwrap_model(nnet).load_state_dict(torch.load(config.nnet_path, map_location="cpu"))
    nnet.eval()

    schedule = Bridge()

    @torch.cuda.amp.autocast()
    def encode(_batch):
        return autoencoder.encode(_batch)

    @torch.cuda.amp.autocast()
    def decode(_batch):
        return autoencoder.decode(_batch)

    if config.train.mode == "cond" and config.sample.guidance and config.sample.scale > 0:
        # domain convergent guidance
        cprint(f"Using domain convergent guidance with scale={config.sample.scale}", "cyan")

        def dcg_nnet(x, timesteps, y):
            _cond = nnet(x, timesteps, y=y)
            _uncond = nnet(x, timesteps, y=repeat(dataset.empty_token, "-> b", b=x.shape[0]).to(x.device))  # fmt: skip

            # return -config.sample.scale * _uncond + (1 + config.sample.scale) * _cond
            return config.sample.scale * _uncond + (1 - config.sample.scale) * _cond
            # return (1 + config.sample.scale) * _uncond - config.sample.scale * _cond

    elif config.sample.cond:
        cprint("Conditioning with domain information", "cyan")

        def dcg_nnet(x, timesteps, y):
            return nnet(x, timesteps, y=y)

    else:
        cprint("Not conditioning with domain information", "cyan")

        def dcg_nnet(x, timesteps, y):
            return nnet(
                x, timesteps, y=repeat(dataset.empty_token, "-> b", b=x.shape[0]).to(x.device)
            )

    logging.info(config.sample)
    assert os.path.exists(dataset.fid_stat)
    logging.info(f"sample: mode={config.train.mode}, mixed_precision={config.mixed_precision}")

    @torch.no_grad()
    def ddpm_sample(x_init: torch.Tensor, **kwargs):

        def pred_x0_fn(x_t, t: int):  # t: [1000, 999, ..., 2]
            t = np.full(x_t.shape[0], t, dtype=int)

            pred = dcg_nnet(x_t, torch.from_numpy(t).to(x_t), **kwargs)

            std_fwd = np.sqrt(schedule._var_fwd[t - 1])
            return x_t - schedule._stp(std_fwd, pred)

        x_t = x_init

        steps = np.arange(1, schedule.T + 1)[::-1]  # [1000, 999, ..., 2, 1]
        for prev_step, step in tqdm(
            zip(steps[1:], steps[:-1]), desc="DDPM sampling", total=schedule.T - 1
        ):
            pred_x_0 = pred_x0_fn(x_t, step)
            x_t = schedule.p_posterior(prev_step, step, x_t, pred_x_0)

        return decode(x_t)

    def sample_fn(batch):
        x_init = brain_encoder.encode(batch[0].to(device), batch[2].to(device)).reshape(-1, *dataset.data_shape)  # fmt: skip

        if config.train.mode == "uncond":
            return ddpm_sample(x_init)
        elif config.train.mode == "cond":
            # NOTE: using non-empty token for conditional sampling
            return ddpm_sample(x_init, y=batch[2].to(device))
        else:
            raise ValueError(f"Unknown training mode: {config.train.mode}")

    with tempfile.TemporaryDirectory() as temp_path:
        path = config.sample.path or temp_path
        if accelerator.is_main_process:
            os.makedirs(path, exist_ok=True)
        logging.info(f"Samples are saved in {path}")

        ssim, clip = sample2dir(
            accelerator,
            path,
            test_loader,
            config.sample,
            sample_fn,
            dataset.unpreprocess,
            eval=True,
        )

        if accelerator.is_main_process:
            fid = calculate_fid_given_paths((dataset.fid_stat, path))

            cprint(f"fid={fid:.3f}, ssim={ssim:.3f}, clip={clip:.3f}", "cyan")
            np.savetxt(os.path.join(path, "metrics.txt"), [fid, ssim, clip], fmt="%.5f")


from absl import flags
from absl import app
from ml_collections import config_flags
import os


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])
flags.DEFINE_string("nnet_path", None, "The nnet to evaluate.")
flags.DEFINE_string("output_path", None, "The path to output log.")


def main(argv):
    config = FLAGS.config
    config.nnet_path = FLAGS.nnet_path
    config.output_path = FLAGS.output_path
    evaluate(config)


if __name__ == "__main__":
    app.run(main)
