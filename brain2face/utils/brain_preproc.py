import sys
import numpy as np
import mne
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm
from termcolor import cprint
from typing import Optional, Tuple

from brain2face.utils.preproc_utils import crop_and_segment


def scale_and_clamp(X: np.ndarray, clamp_lim, clamp=True) -> np.ndarray:
    """
    Args:
        X: ( channels, timesteps )
    Returns:
        X: ( channels, timesteps )
    """
    X = RobustScaler().fit_transform(X.T)  # NOTE: must be samples x features

    if clamp:
        X = X.clip(min=-clamp_lim, max=clamp_lim)

    return X.T


def baseline_correction(X: np.ndarray, baseline_len_samp: int) -> np.ndarray:
    """Channel-wise baseline-correction.
    Args:
        X: ( segments, channels, segment_len )
    Returns:
        X ( segments, channels, segment_len )
    """
    X = X.transpose(1, 0, 2)  # ( C, segments, T )

    for chunk_id in range(X.shape[1]):
        baseline = X[:, chunk_id, :baseline_len_samp].mean(axis=1)

        X[:, chunk_id, :] -= baseline.reshape(-1, 1)

    return X.transpose(1, 0, 2)  # Back to ( segments, C, T )


def segment_with_times(eeg, face_times, eeg_times, segment_len):
    """For driving game EEG, but could be modified for other datasets later.
    Args:
        eeg (_type_): _description_
        face_times (_type_): _description_
        eeg_times (_type_): _description_
        segment_len (_type_): _description_
    Returns:
        _type_: _description_
    """
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


def brain_preproc(
    args,
    brain: np.ndarray,
    brain_times: Optional[np.ndarray] = None,
    face_times: Optional[np.ndarray] = None,
    segment_len: Optional[int] = None,
    shift: Optional[float] = None,
) -> Tuple[np.ndarray, Optional[int], Optional[int]]:
    """
    Args:
        brain: EEG or ECoG | ( channels, timesteps )
        brain_times: ( timesteps, )
        face_times: ( timesteps, )
        segment_len: Number of timesteps per segment
        shift: How many seconds to shift brain data to the future (need to be positive)
    Returns:
        brain: ( segments, channels, segment_len )
    """
    if brain_times is not None:
        # NOTE: ensure they're ascending
        assert np.all(brain_times[:-1] <= brain_times[1:])
        assert np.all(face_times[:-1] <= face_times[1:])
    else:
        assert segment_len is not None, "Must provide segment_len if no brain_times"

    """ Filtering """
    brain = mne.filter.filter_data(
        brain,
        sfreq=args.brain_resample_sfreq,
        l_freq=args.brain_filter_low,
        h_freq=args.brain_filter_high,
    )

    """ Resampling """
    brain = mne.filter.resample(
        brain,
        down=args.brain_orig_sfreq / args.brain_resample_sfreq,
    )

    """ Scaling """
    brain = scale_and_clamp(brain, clamp_lim=args.clamp_lim)

    """ Segmenting """
    if brain_times is not None:
        # NOTE: need to ensure that we get same timings by running resample separately for
        #       EEG and timestamps
        brain_times = mne.filter.resample(
            brain_times, down=args.brain_orig_sfreq / args.brain_resample_sfreq
        )

        brain, face_drops_prev, face_drops_after = segment_with_times(
            brain, face_times, brain_times, segment_len
        )  # ( segments, channels, segment_len )

    else:
        brain = brain.T  # ( timesteps, channels )

        if shift is not None:
            brain = brain[int(shift * args.brain_resample_sfreq) :]

        brain = crop_and_segment(
            brain, segment_len
        )  # ( segments, segment_len, channels )
        brain = brain.transpose(0, 2, 1)  # ( segments, channels, segment_len )

    """ Baseline Correction """
    brain = baseline_correction(
        brain, int(args.baseline_len * args.brain_resample_sfreq)  # args.fps
    )  # ( segments, channels, segment_len )

    try:
        return brain, face_drops_prev, face_drops_after
    except NameError:
        return brain


if __name__ == "__main__":
    # from configs.args import args

    # eeg_path = "/home/neurotech_nas01/driving-game/driving-game-data_20211220/2021-12-20_104253_data/result/driving-simulation-RecordSession_2021.12.20_10.42.31.hdf5"

    # x, time = eeg_preproc(args, eeg_path)

    # print(x.shape)
    # print(time.shape)

    _ = baseline_correction(np.random.rand(500, 32, 360), 60)
