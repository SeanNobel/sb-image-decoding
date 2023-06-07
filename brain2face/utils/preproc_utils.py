import os
from glob import glob
import numpy as np
from termcolor import cprint
import json
from typing import Optional, List


def crop_and_segment(x: np.ndarray, segment_len: int) -> np.ndarray:
    """Crops the input to multiple of segment_len and segments it for the first dimension.
    Rest of the dimensions are arbitrary. Applies for both EEG and video data. Also works for
    1D array.
    Args:
        x (np.ndarray): ( timesteps, ... )
        segment_len (int): _description_
    Returns:
        x (np.ndarray): ( segments, segment_len, ... )
    """
    # Crop
    x = x[: -(x.shape[0] % segment_len)]  # ( ~100000, * )

    # Segment
    x = x.reshape(-1, segment_len, *x.shape[1:])
    # ( ~100000//segment_len, segment_len, * )

    return x


def get_uhd_data_paths(
    data_root: str, start_subj: Optional[int] = None, end_subj: Optional[int] = None
) -> List[str]:
    """NOTE: Currently only working for sessions in which EEG recording started before
        video recording. TODO: Accept negative shift.
    Args:
        data_root (str): _description_
    Returns:
        List[str]: _description_
    """
    sync_paths = []
    video_paths = []
    eeg_paths = []

    _sync_paths = glob(data_root + "**/estim_delay.json", recursive=True)
    print(f"Found {len(_sync_paths)} sync files.")

    for sync_path in _sync_paths:
        shift = -json.load(open(sync_path))["estim_delay_sec"]
        if shift < 0:
            cprint(f"SKIPPED: {sync_path} (negative shift)", color="yellow")
            continue

        dirname = os.path.dirname(sync_path)

        # NOTE: Loosely ensuring that the video is not a copied one or something
        video_path = glob(dirname + "/*[0-9].mov") + glob(dirname + "/*[0-9].mkv")
        eeg_path = glob(dirname + "/EEG_try*.h5") + glob(dirname + "/EEG_try*.xdf")

        if len(video_path) == 1 and len(eeg_path) == 1:
            cprint(f"USED: {sync_path}", color="cyan")

            sync_paths.append(sync_path)
            video_paths.append(video_path[0])
            eeg_paths.append(eeg_path[0])

        else:
            cprint(
                f"SKIPPED: {sync_path} (found {len(video_path)} video_paths and {len(eeg_path)} eeg_paths)",
                color="yellow",
            )
            continue

    assert len(sync_paths) == len(video_paths) == len(eeg_paths)

    cprint(f"{len(video_paths)} sessions in total.", color="cyan")

    if start_subj is not None:
        assert end_subj is not None, "If start_subj is given, end_subj must be given."

        cprint(
            f"Taking sessions from {start_subj} to {end_subj} for preprocessing",
            color="cyan",
        )
        sync_paths = sync_paths[start_subj:end_subj]
        video_paths = video_paths[start_subj:end_subj]
        eeg_paths = eeg_paths[start_subj:end_subj]

    for path in eeg_paths:
        cprint(path, color="cyan")

    return sync_paths, video_paths, eeg_paths


def get_face2brain_data_paths(
    data_root: str,
    ica_data_root: Optional[str] = None,
    start_subj: Optional[int] = None,
    end_subj: Optional[int] = None,
):
    video_paths = []
    video_times_paths = []
    eeg_paths = []

    for video_path in glob(data_root + "**/camera5*.mp4", recursive=True):
        data_root_dir = os.path.split(os.path.split(video_path)[0])[0]

        video_times_path = data_root_dir + "/result/camera5_timestamps.csv"

        if os.path.exists(video_times_path):
            video_times_paths.append(video_times_path)
        else:
            cprint(f"SKIPPED: Timestamps for camera5 not found.", color="yellow")
            continue

        if ica_data_root is None:
            eeg_path = glob(data_root_dir + "/**/*.hdf5")
        else:
            eeg_path = glob(ica_data_root + data_root_dir.split("/")[-1] + "/eeg.npz")

        if len(eeg_path) == 1:
            video_paths.append(video_path)
            eeg_paths.append(eeg_path[0])
        else:
            cprint(
                f"SKIPPED: {len(eeg_path)} corresponding EEG data found.", color="yellow"
            )
            continue

    assert len(video_paths) == len(eeg_paths) == len(video_times_paths)

    cprint(f"{len(video_paths)} sessions in total.", color="cyan")

    if start_subj is not None:
        assert end_subj is not None, "If start_subj is given, end_subj must be given."

        cprint(
            f"Taking sessions from {start_subj} to {end_subj} for preprocessing",
            color="cyan",
        )
        video_paths = video_paths[start_subj:end_subj]
        video_times_paths = video_times_paths[start_subj:end_subj]
        eeg_paths = eeg_paths[start_subj:end_subj]

    for path in eeg_paths:
        cprint(path, color="cyan")

    return video_paths, video_times_paths, eeg_paths
