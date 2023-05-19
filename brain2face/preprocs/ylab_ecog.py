import os, sys
import numpy as np
import pandas as pd
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

from brain2face.utils.brain_preproc import brain_preproc

# from brain2face.utils.face_preproc import face_preproc
from brain2face.utils.preproc_utils import export_gif


def run_preprocess(tmp) -> None:
    args, i, video_path, video_times_path, eeg_path = tmp

    data_dir = f"data/{args.preproc_name}/S{i}/"

    # FIXME: remove later ?
    if not os.path.exists(data_dir + "example.gif"):
        cprint(f"Processing subject number {i}", color="cyan")
        os.makedirs(data_dir, exist_ok=True)

        Y, Y_times = face_preproc(args, video_path, video_times_path)

        X, y_drops_prev, y_drops_after = eeg_preproc(args, eeg_path, Y_times)
        cprint(f"Subject {i} brain: {X.shape}", color="cyan")

        if y_drops_after == 0:
            cprint("No drops after", color="yellow")
            Y = Y[y_drops_prev:]
        else:
            Y = Y[y_drops_prev:-y_drops_after]

        cprint(f"Subject {i} face: {Y.shape}", color="cyan")

        assert len(X) == len(Y)

        np.save(data_dir + "face.npy", Y)
        np.save(data_dir + "brain.npy", X)
        export_gif(data_dir + "example.gif", Y)
        # cv2.imwrite(data_dir + "first_frame.png", first_frame)


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


@hydra.main(version_base=None, config_path="../../configs", config_name="ylab_ecog")
def main(args: DictConfig) -> None:
    with open_dict(args):
        args.root_dir = get_original_cwd()

    sync_df_all = pd.read_csv(args.sync_data_path)
    movie_names = np.unique(sync_df_all.movie_name.values)

    face_paths = natsorted(glob.glob(args.face_data_root + "E0030_*.csv"))

    session_ids = [int(re.split("[._]", path)[-2]) - 1 for path in face_paths]

    for i, (session_id, face_path) in enumerate(zip(session_ids, face_paths)):
        sync_df = sync_df_all[sync_df_all.movie_name == movie_names[session_id]]

        face_df = pd.read_csv(face_path)
        # face_data = face_df.filter(like="p_", axis=1).values
        face_data = face_df.drop(
            ["frame", " face_id", " timestamp", " confidence", " success"],
            axis=1,
        ).values
        face_data = face_data[sync_df.movie_frame.values.astype(int)]
        cprint(f"Face data: {face_data.shape}", "cyan")
        sys.exit()
        face_data = face_data.reshape(
            -1,
        )

        ecog_data = load_ecog_data(args, sync_df)
        cprint(f"ECoG data: {ecog_data.shape}", "cyan")
        sys.exit()

        ecog_filenames = sync_data.brainwave_name.values
        print(ecog_filenames)
        sys.exit()
        assert np.all(sync_data.brainwave_name.values == ecog_filename)

        ecog_path = os.path.join(args.ecog_data_root, ecog_filename)

        data_root_dir = os.path.split(os.path.split(video_path)[0])[0]

        video_times_path = data_root_dir + "/result/camera5_timestamps.csv"

        if os.path.exists(video_times_path):
            video_times_paths.append(video_times_path)
        else:
            cprint(f"SKIPPING: Timestamps for camera5 not found.", color="yellow")
            continue

        eeg_path = glob.glob(data_root_dir + "/**/*.hdf5")

        if len(eeg_path) == 1:
            video_paths.append(video_path)
            eeg_paths.append(eeg_path[0])
        else:
            cprint(
                f"SKIPPING: {len(eeg_path)} corresponding EEG data found.", color="yellow"
            )
            continue

    cprint(
        f"{len(video_paths)} subjects (counting different sessions as different subjects)",
        color="cyan",
    )
    for path in video_paths:
        cprint(path, color="cyan")
    sys.exit()
    assert len(video_paths) == len(eeg_paths) and len(video_paths) == len(
        video_times_paths
    )

    # -------------------------
    #    Running preprocess
    # -------------------------
    if args.subject_multiprocess:
        subj_list = [
            (args, i, *paths)
            for i, paths in enumerate(zip(video_paths, video_times_paths, eeg_paths))
        ]

        # with ctx.Pool(4) as p:
        with torch.multiprocessing.Pool(4) as p:
            res = p.map(run_preprocess, subj_list)

    else:
        # video_paths = video_paths[args.start_subj : args.end_subj]
        # video_times_paths = video_times_paths[args.start_subj : args.end_subj]
        # eeg_paths = eeg_paths[args.start_subj : args.end_subj]

        for i, paths in enumerate(zip(video_paths, video_times_paths, eeg_paths)):
            if args.start_subj <= i and i < args.end_subj:
                run_preprocess((args, i, *paths))


if __name__ == "__main__":
    main()
