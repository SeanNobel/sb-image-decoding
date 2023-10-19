import sys
import numpy as np
import mne
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm
from termcolor import cprint
from typing import Optional, Tuple, Union, List

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


def baseline_correction(
    X: np.ndarray,
    baseline_len_samp: Optional[int] = None,
    baseline: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Channel-wise baseline-correction.
    Args:
        X ( segments, channels, segment_len ): data
        baseline ( segments, channels, baseline_len ): Precomputed baselines for those starting before onsets.
    Returns:
        X ( segments, channels, segment_len )
    """
    assert baseline_len_samp is not None or baseline is not None, "Must provide either baseline_len_samp or baseline."  # fmt: skip
    
    if baseline is None:
        # NOTE: this could be zero with very short seq_len.
        baseline_len_samp = max(baseline_len_samp, 1)
            
        baseline = X[:, :, :baseline_len_samp].mean(axis=-1, keepdims=True)
        # ( segments, channels )
    else:
        baseline = baseline.mean(axis=-1, keepdims=True)
    
    return X - baseline


def segment_with_times(
    eeg: np.ndarray, segment_len: int, face_times: np.ndarray, eeg_times: np.ndarray
) -> Tuple[np.ndarray, int, int]:
    """For driving game EEG, but could be modified for other datasets later.
    Args:
        eeg (np.ndarray): ( channels, timesteps )
        segment_len (int): Number of timesteps per segment
        face_times (np.ndarray): ( timesteps, )
        eeg_times (np.ndarray): ( timestep, )
    Returns:
        _type_: _description_
    """
    cprint(eeg.shape, "yellow")
    cprint(face_times.shape, "yellow")
    cprint(eeg_times.shape, "yellow")
    cprint(segment_len, "yellow")

    # NOTE: ensure they're ascending
    assert np.all(eeg_times[:-1] <= eeg_times[1:])
    assert np.all(face_times[:-1] <= face_times[1:])

    face_drops_prev = 0  # NOTE: Number of segments
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
    segment: bool = True,
    segment_len: Optional[int] = None,
    brain_times: Optional[np.ndarray] = None,
    face_times: Optional[np.ndarray] = None,
    shift: Optional[float] = None,
    resample: bool = True,
    orig_sfreq: Optional[int] = None,
    notch: Optional[List[float]] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, int, int]]:
    """
    Args:
        brain: EEG or ECoG | ( channels, timesteps )
        segment_len: Number of timesteps per segment
        brain_times: ( timesteps, )
        face_times: ( timesteps, )
        shift: How many seconds to shift brain data to the future (need to be positive)
    Returns:
        brain: ( segments, channels, segment_len ) or ( channels, timesteps )
    """
    assert not (segment and segment_len is None), "Must provide segment_len when segmenting."  # fmt: skip
    if orig_sfreq is None:
        orig_sfreq = args.brain_orig_sfreq

    """ Bandpass Filtering """
    brain = mne.filter.filter_data(
        brain,
        sfreq=orig_sfreq,
        l_freq=args.brain_filter_low,
        h_freq=args.brain_filter_high,
    )
    
    """ Notch Filtering """
    if notch is not None:
        brain = mne.filter.notch_filter(
            brain,
            Fs=orig_sfreq,
            freqs=notch,
        )

    """ Resampling """
    if resample:
        brain = mne.filter.resample(
            brain,
            down=orig_sfreq / args.brain_resample_sfreq,
        )

    """ Scaling & clamping """
    brain = scale_and_clamp(brain, clamp_lim=args.clamp_lim)

    if not segment:
        return brain

    """ Segmenting & Baseline Correction """
    brain = segment_then_blcorr(args, brain, segment_len, brain_times, face_times, shift)

    return brain  # NOTE: This could be tuple.


def segment_then_blcorr(
    args,
    brain: np.ndarray,
    segment_len: int,
    brain_times: Optional[np.ndarray] = None,
    face_times: Optional[np.ndarray] = None,
    shift: Optional[float] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, int, int]]:
    """Segmenting"""
    if brain_times is not None:
        assert face_times is not None, "Must provide face_times with brain_times."

        # NOTE: need to ensure that we get same timings by running resample separately for
        #       EEG and timestamps
        brain_times = mne.filter.resample(
            brain_times, down=args.brain_orig_sfreq / args.brain_resample_sfreq
        )

        brain, face_drops_prev, face_drops_after = segment_with_times(
            brain, segment_len, face_times, brain_times
        )  # ( segments, channels, segment_len )

    else:
        brain = brain.T  # ( timesteps, channels )

        if shift is not None:
            brain = brain[int(shift * args.brain_resample_sfreq) :]

        brain = crop_and_segment(brain, segment_len)
        # ( segments, segment_len, channels )
        brain = brain.transpose(0, 2, 1)  # ( segments, channels, segment_len )
        
    """ Baseline Correction """
    brain = baseline_correction(
        brain, int(args.seq_len * args.baseline_ratio * args.brain_resample_sfreq)
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
