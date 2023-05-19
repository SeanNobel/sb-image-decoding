import os, sys
import numpy as np
import pandas as pd
import mne
import glob
from natsort import natsorted
import h5py
import re
from termcolor import cprint
from tqdm import tqdm
import multiprocessing

ctx = multiprocessing.get_context("spawn")

from tqdm import tqdm
import cv2
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, open_dict

import torch

torch.multiprocessing.set_start_method("spawn", force=True)

from brain2face.utils.brain_preproc import scale_and_clamp, baseline_correction


def load_ecog_data(args, sync_df: pd.DataFrame) -> np.ndarray:
    brainwave_name = sync_df.brainwave_name.values[0]
    dat = h5py.File(args.ecog_data_root + brainwave_name, "r")["ecog_dat"][()]

    brainwave_frames = sync_df.brainwave_frame.values.astype(int)
    ecog_data = []
    for i, frame in enumerate(tqdm(brainwave_frames)):
        if sync_df.brainwave_name.values[i] != brainwave_name:
            cprint("Loading new ECoG file.", "yellow")
            brainwave_name = sync_df.brainwave_name.values[i]
            dat = h5py.File(args.ecog_data_root + brainwave_name, "r")["ecog_dat"][()]

        ecog_data.append(dat[frame])

    return np.stack(ecog_data)


def face_preproc(face_path: str, sync_df: pd.DataFrame, segment_len: int) -> np.ndarray:
    """Loads interpolated facial features, transforms them for the dataset

    Args:
        args (_type_): _description_
        face_path (str): _description_
        sync_df (pd.DataFrame): _description_
        segment_len (int): _description_

    Returns:
        np.ndarray: ( segments, features, segment_len )
    """
    face_df = pd.read_csv(face_path)

    # face_data = face_df.filter(like="p_", axis=1).values
    face_data = face_df.drop(
        ["frame", " face_id", " timestamp", " confidence", " success"],
        axis=1,
    ).values

    face_data = face_data[sync_df.movie_frame.values.astype(int)]

    face_data = face_data[: -(face_data.shape[0] % segment_len)]
    face_data = face_data.reshape(-1, segment_len, face_data.shape[-1])

    return face_data.transpose(0, 2, 1)


def ecog_preproc(args, ecog: np.ndarray, segment_len: int) -> np.ndarray:
    """
    Args:
        ecog: ( channels, timesteps )
    Returns:
        ecog: ( segments, channels, segment_len )
    """

    """ Filtering """
    # eeg_filtered = mne.filter.filter_data(
    #     ecog_raw,
    #     sfreq=250,
    #     l_freq=args.brain_filter_low,
    #     h_freq=args.brain_filter_high,
    # )

    """ Resampling """
    # eeg_resampled = mne.filter.resample(
    #     eeg_filtered,
    #     down=250 / args.brain_resample_rate,
    # )

    """ Scaling """
    ecog = scale_and_clamp(ecog, clamp_lim=args.clamp_lim)

    """ Segmenting """
    ecog = ecog[:, : -(ecog.shape[1] % segment_len)]
    ecog = ecog.reshape(ecog.shape[0], segment_len, -1)
    ecog = ecog.transpose(2, 0, 1)  # ( segments, channels, segment_len )

    """ Baseline Correction """
    ecog = baseline_correction(
        ecog, int(args.baseline_len * args.fps) # FIXME
    )  # ( segments, channels, segment_len )

    return ecog


@hydra.main(version_base=None, config_path="../../configs", config_name="ylab_ecog")
def main(args: DictConfig) -> None:
    with open_dict(args):
        args.root_dir = get_original_cwd()

    sync_df_all = pd.read_csv(args.sync_data_path)
    movie_names = np.unique(sync_df_all.movie_name.values)

    face_paths = natsorted(glob.glob(args.face_data_root + "E0030_*.csv"))

    session_ids = [int(re.split("[._]", path)[-2]) - 1 for path in face_paths]

    segment_len = args.seq_len * args.fps

    for i, (session_id, face_path) in enumerate(zip(session_ids, face_paths)):
        cprint(f"Processing subject number {i}", color="cyan")

        sync_df = sync_df_all[sync_df_all.movie_name == movie_names[session_id]]

        ecog_raw = load_ecog_data(args, sync_df).T
        X = ecog_preproc(args, ecog_raw, segment_len)
        cprint(f"Subject {i} ECoG: {X.shape}", "cyan")

        Y = face_preproc(face_path, sync_df, segment_len)
        cprint(f"Subject {i} face: {Y.shape}", "cyan")

        assert len(X) == len(Y)

        data_dir = f"{args.root_dir}/data/YLab/{args.preproc_name}/S{i}/"
        os.makedirs(data_dir, exist_ok=True)
        np.save(data_dir + "brain.npy", X)
        np.save(data_dir + "face.npy", Y)


if __name__ == "__main__":
    main()
