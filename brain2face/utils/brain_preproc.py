import sys
import numpy as np
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm
from termcolor import cprint


def scale_and_clamp(X: np.ndarray, clamp_lim, clamp=True) -> np.ndarray:
    """
    X: ( C, T )
    """
    X = RobustScaler().fit_transform(X.T)  # NOTE: must be samples x features

    if clamp:
        X = X.clip(min=-clamp_lim, max=clamp_lim)

    return X.T  # NOTE: make ( ch, time ) again


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
