import sys
import numpy as np
import mne
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm
from termcolor import cprint
from typing import Optional


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


def scale_and_clamp(X: np.ndarray, clamp_lim, clamp=True) -> np.ndarray:
    """
    X: ( C, T )
    """
    X = RobustScaler().fit_transform(X.T)  # NOTE: must be samples x features

    if clamp:
        X = X.clip(min=-clamp_lim, max=clamp_lim)

    return X.T  # NOTE: make ( ch, time ) again


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


def baseline_correction(X: np.ndarray, baseline_len_samp: int) -> np.ndarray:
    """args:
        X: ( segments, C, T )
    returns:
        X ( segments, C, T ) baseline-corrected channel-wise
    """
    X = X.transpose(1, 0, 2)  # ( C, segments, T )

    for chunk_id in range(X.shape[1]):
        baseline = X[:, chunk_id, :baseline_len_samp].mean(axis=1)

        X[:, chunk_id, :] -= baseline.reshape(-1, 1)

    return X.transpose(1, 0, 2)  # Back to ( segments, C, T )


if __name__ == "__main__":
    # from configs.args import args

    # eeg_path = "/home/neurotech_nas01/driving-game/driving-game-data_20211220/2021-12-20_104253_data/result/driving-simulation-RecordSession_2021.12.20_10.42.31.hdf5"

    # x, time = eeg_preproc(args, eeg_path)

    # print(x.shape)
    # print(time.shape)

    _ = baseline_correction(np.random.rand(500, 32, 360), 60)
