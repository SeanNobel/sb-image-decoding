import os, sys
import numpy as np
import glob
from termcolor import cprint
import multiprocessing

ctx = multiprocessing.get_context("spawn")

from tqdm import tqdm
import cv2
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, open_dict

import torch

torch.multiprocessing.set_start_method("spawn", force=True)

from brain2face.utils.eeg_preproc import brain_preproc
from brain2face.utils.face_preproc import face_preproc
from brain2face.utils.preproc_utils import export_gif
from brain2face.utils.gTecUtils.gtec_preproc import eeg_subset_fromTrigger


def brain_preproc(
    args,
    brain_raw: np.ndarray,
    brain_times: Optional[np.ndarray] = None,
    face_times: Optional[np.ndarray] = None,
):
    """
    Args:
        brain_raw: ( channels, timesteps )
        brain_times: ( timesteps, )
    """
    assert np.all(eeg_times[:-1] <= eeg_times[1:])  # ensure it's ascending
    assert np.all(face_times[:-1] <= face_times[1:])

    """ Filtering """
    eeg_filtered = mne.filter.filter_data(
        eeg_raw,
        sfreq=250,
        l_freq=args.brain_filter_low,
        h_freq=args.brain_filter_high,
    )

    """ Resampling """
    eeg_resampled = mne.filter.resample(
        eeg_filtered,
        down=250 / args.brain_resample_rate,
    )

    """ Scaling """
    eeg_scaled = scale_and_clamp(eeg_resampled, clamp_lim=args.clamp_lim)

    """ Segmenting """
    # NOTE: need to ensure that we get same timings by running resample separately for
    #       EEG and timestamps
    eeg_times = mne.filter.resample(eeg_times, down=250 / args.brain_resample_rate)
    segment_len = args.seq_len * args.brain_resample_rate  # 360

    eeg_segmented, face_drops_prev, face_drops_after = segment(
        eeg_scaled, face_times, eeg_times, segment_len
    )  # ( segments=1207, C=32, T=360 )

    """ Baseline Correction """
    eeg_bl_corrected = baseline_correction(
        eeg_segmented, int(args.baseline_len * args.brain_resample_rate)
    )  # ( segments, C, T )

    """ Scaling 2 """
    # NOTE: scale for channels separately
    # eeg_scaled = []
    # for c in range(eeg_bl_corrected.shape[1]):
    #     scaler = RobustScaler().fit(eeg_bl_corrected[:, c])
    #     _eeg_scaled = scaler.transform(eeg_bl_corrected[:, c])
    #     eeg_scaled.append(_eeg_scaled)

    # eeg_scaled = np.stack(eeg_scaled).transpose(1, 0, 2)

    del eeg_filtered, eeg_resampled, eeg_scaled, eeg_segmented

    return eeg_bl_corrected, face_drops_prev, face_drops_after


def segment(eeg, face_times, eeg_times, segment_len):
    face_drops_prev = 0
    face_drops_after = 0
    x_list = []
    for t in tqdm(face_times):
        if t < eeg_times[0]:
            face_drops_prev += 1
            continue

        start_idx = np.searchsorted(eeg_times, t, sorter=None)

        x = eeg[:, start_idx : start_idx + segment_len]

        if x.shape[1] == segment_len:
            x_list.append(x)
        else:
            face_drops_after += 1

    cprint(f"prev: {face_drops_prev} | after: {face_drops_after}", "yellow")

    return np.stack(x_list), face_drops_prev, face_drops_after


def run_preprocess(tmp) -> None:
    args, i, video_path, video_times_path, eeg_path = tmp

    data_dir = f"data/{args.preproc_name}/S{i}/"

    # FIXME: remove later ?
    if not os.path.exists(data_dir + "example.gif"):
        cprint(f"Processing subject number {i}", color="cyan")
        os.makedirs(data_dir, exist_ok=True)

        Y, Y_times = face_preproc(args, video_path, video_times_path)

        eeg_raw, eeg_times, _ = eeg_subset_fromTrigger(args, eeg_path)
        X, y_drops_prev, y_drops_after = brain_preproc(args, eeg_raw, eeg_times, Y_times)
        cprint(f"Subject {i} EEG: {X.shape}", color="cyan")

        if y_drops_after == 0:
            cprint("No drops after", color="yellow")
            Y = Y[y_drops_prev:]
        else:
            Y = Y[y_drops_prev:-y_drops_after]

        cprint(f"Subject {i} face: {Y.shape}", color="cyan")

        assert len(X) == len(Y)

        np.save(data_dir + "brain.npy", X)
        np.save(data_dir + "face.npy", Y)
        export_gif(data_dir + "example.gif", Y)
        # cv2.imwrite(data_dir + "first_frame.png", first_frame)


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
