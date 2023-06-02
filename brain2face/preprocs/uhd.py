import os, sys
import numpy as np
import mne
import h5py, pyxdf, json
from tqdm import tqdm
from termcolor import cprint
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, open_dict

from brain2face.utils.brain_preproc import brain_preproc
from brain2face.utils.face_preproc import FacePreprocessor
from brain2face.utils.preproc_utils import get_uhd_data_paths


def sequential_load_resample(
    data: h5py._hl.dataset.Dataset, bufsize: int, down: float
) -> np.ndarray:
    """h5 EEG files are sometimes too large (~500GB) to load at once.
    Args:
        data (h5py._hl.dataset.Dataset): ( chunks, orig_sfreq, channels )
        bufsize (int): How many chunks to process at once
        down (float): Downsampling factor for mne.filter.resample()
    Returns:
        np.ndarray: ( channels, timesteps@new_sfreq )
    """
    X = []

    for i in tqdm(range(len(data) // bufsize + 1)):
        chunk = data[i * bufsize : (i + 1) * bufsize]
        chunk = chunk.reshape((-1, data.shape[-1])).astype(np.float64).T
        chunk = mne.filter.resample(chunk, down=down)

        X.append(chunk)

    return np.concatenate(X, axis=1)


@hydra.main(version_base=None, config_path="../../configs", config_name="uhd")
def main(args: DictConfig) -> None:
    with open_dict(args):
        args.root_dir = get_original_cwd()

    sync_paths, video_paths, eeg_paths = get_uhd_data_paths(args.data_root)

    for _i, paths in enumerate(zip(sync_paths, video_paths, eeg_paths)):
        i = args.start_subj + _i
        cprint(f"Processing session {i}", color="cyan")

        sync_path, video_path, eeg_path = paths

        Y, drop_segments = FacePreprocessor(args, video_path)()

        if eeg_path.endswith(".h5"):
            eeg_data = h5py.File(eeg_path, "r")["EEG/EEG"]
            cprint(f"h5 file shape: {eeg_data.shape}", "cyan")

            # NOTE: Resampling will be done twice
            X = sequential_load_resample(
                data=eeg_data,
                bufsize=args.eeg_load_bufsize,
                down=eeg_data.shape[1] / args.brain_orig_sfreq,
            )

        elif eeg_path.endswith(".xdf"):
            X, _ = pyxdf.load_xdf(eeg_path)
            X = X[0]["time_series"].astype(np.float64).T

        else:
            raise NotImplementedError("Only h5 and xdf files are supported.")

        shift = -json.load(open(sync_path))["estim_delay_sec"]
        assert shift >= 0, "Shift must be positive float."

        X = brain_preproc(
            args,
            X,
            segment_len=int(args.brain_resample_sfreq * args.seq_len),
            shift=shift,
        )

        assert len(X) == len(Y), "Brain and face data have different number of segments."

        X = np.delete(X, drop_segments, axis=0)
        Y = np.delete(Y, drop_segments, axis=0)

        cprint(f"Session {i} EEG: {X.shape}", "cyan")
        cprint(f"Session {i} face: {Y.shape}", "cyan")

        data_dir = f"data/uhd/{args.preproc_name}/S{i}/"
        os.makedirs(data_dir, exist_ok=True)
        np.save(data_dir + "brain.npy", X)
        np.save(data_dir + "face.npy", Y)


if __name__ == "__main__":
    main()
