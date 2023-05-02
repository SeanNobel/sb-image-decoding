import os, sys
import numpy as np
import glob
from termcolor import cprint
from multiprocessing import Pool
from tqdm import tqdm
import cv2
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, open_dict

from f2b_contrastive.data.eeg_preproc import eeg_preproc
from f2b_contrastive.data.face_preproc import face_preproc
from f2b_contrastive.utils.preproc_utils import export_gif


def run_preprocess(tmp) -> None:
    args, i, video_path, video_times_path, eeg_path = tmp

    data_dir = f"data/{args.preproc_name}/S{i}/"

    # FIXME: remove later ?
    if not os.path.exists(data_dir + "example.gif"):
        cprint(f"Processing subject number {i}", color="cyan")
        os.makedirs(data_dir, exist_ok=True)

        Y, Y_times, first_frame = face_preproc(args, video_path, video_times_path)

        sys.exit()

        X, y_drops_prev, y_drops_after = eeg_preproc(args, eeg_path, Y_times)
        cprint(f"Subject {i} brain: {X.shape}", color="cyan")

        Y = Y[y_drops_prev:-y_drops_after]
        cprint(f"Subject {i} face: {Y.shape}", color="cyan")

        assert len(X) == len(Y)

        np.save(data_dir + "face.npy", Y)
        np.save(data_dir + "brain.npy", X)
        export_gif(data_dir + "example.gif", Y)
        cv2.imwrite(data_dir + "first_frame.png", first_frame)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(args: DictConfig) -> None:
    with open_dict(args):
        args.root_dir = get_original_cwd()

    video_paths = []
    video_times_paths = []
    eeg_paths = []

    for video_path in glob.glob(args.data_root + "**/camera5*.mp4", recursive=True):
        data_root_dir = os.path.split(os.path.split(video_path)[0])[0]

        video_times_path = data_root_dir + "/result/camera5_timestamps.csv"
        print(video_times_path)

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
    assert len(video_paths) == len(eeg_paths) and len(video_paths) == len(
        video_times_paths
    )

    # -------------------------
    #    Running preprocess
    # -------------------------
    subj_list = [
        (args, i, *paths)
        for i, paths in enumerate(zip(video_paths, video_times_paths, eeg_paths))
    ]

    with Pool(args.multiprocess) as p:
        res = list(
            tqdm(
                p.imap(run_preprocess, subj_list),
                total=len(subj_list),
                bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}",
            )
        )


if __name__ == "__main__":
    main()
