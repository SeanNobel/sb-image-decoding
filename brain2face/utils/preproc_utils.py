import os
import glob
import numpy as np
from termcolor import cprint
from PIL import Image
from typing import Optional


def export_gif(path, example: np.ndarray) -> None:
    example = example[:10].reshape(-1, example.shape[-2], example.shape[-1])
    gif = [Image.fromarray(p) for p in example]
    gif[0].save(
        path,
        save_all=True,
        append_images=gif[1:],
        duration=100,
        loop=0,
    )


def get_arayadriving_dataset_paths(
    data_root: str,
    ica_data_root: Optional[str] = None,
    start_subj: Optional[int] = None,
    end_subj: Optional[int] = None,
):
    video_paths = []
    video_times_paths = []
    eeg_paths = []

    for video_path in glob.glob(data_root + "**/camera5*.mp4", recursive=True):
        data_root_dir = os.path.split(os.path.split(video_path)[0])[0]

        video_times_path = data_root_dir + "/result/camera5_timestamps.csv"

        if os.path.exists(video_times_path):
            video_times_paths.append(video_times_path)
        else:
            cprint(f"SKIPPING: Timestamps for camera5 not found.", color="yellow")
            continue

        if ica_data_root is None:
            eeg_path = glob.glob(data_root_dir + "/**/*.hdf5")
        else:
            eeg_path = glob.glob(
                ica_data_root + data_root_dir.split("/")[-1] + "/eeg.npz"
            )

        if len(eeg_path) == 1:
            video_paths.append(video_path)
            eeg_paths.append(eeg_path[0])
        else:
            cprint(
                f"SKIPPING: {len(eeg_path)} corresponding EEG data found.", color="yellow"
            )
            continue

    assert len(video_paths) == len(eeg_paths) and len(video_paths) == len(
        video_times_paths
    ), "Number of video paths, EEG paths and video times paths must be equal."

    cprint(
        f"{len(video_paths)} subjects (counting different sessions as different subjects)",
        color="cyan",
    )

    if start_subj is not None:
        assert end_subj is not None, "If start_subj is given, end_subj must be given."

        cprint(
            f"Taking subjects from {start_subj} to {end_subj} for preprocessing",
            color="cyan",
        )
        video_paths = video_paths[start_subj:end_subj]
        video_times_paths = video_times_paths[start_subj:end_subj]
        eeg_paths = eeg_paths[start_subj:end_subj]

    for path in eeg_paths:
        cprint(path, color="cyan")

    return video_paths, video_times_paths, eeg_paths
