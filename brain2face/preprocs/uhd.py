import os, sys
import numpy as np
import mne
import h5py, pyxdf, json
from functools import partial
from termcolor import cprint
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, open_dict

from brain2face.utils.brain_preproc import brain_preproc
from brain2face.utils.face_preproc import FacePreprocessor
from brain2face.utils.preproc_utils import get_uhd_data_paths, sequential_load, h5_save


# FIXME: might cause error when running
def load_resample(chunk: np.ndarray, down: float):
    chunk = chunk.reshape((-1, chunk.shape[-1])).astype(np.float64).T
    chunk = mne.filter.resample(chunk, down=down)

    return chunk


@hydra.main(version_base=None, config_path="../../configs", config_name="uhd")
def main(args: DictConfig) -> None:
    with open_dict(args):
        args.root_dir = get_original_cwd()

    sync_paths, video_paths, eeg_paths = get_uhd_data_paths(
        args.data_root, args.start_subj, args.end_subj
    )

    for _i, paths in enumerate(zip(sync_paths, video_paths, eeg_paths)):
        i = args.start_subj + _i
        cprint(f"Processing session {i}", color="cyan")

        sync_path, video_path, eeg_path = paths

        Y, drop_segments = FacePreprocessor(args, video_path)()

        if eeg_path.endswith(".h5"):
            eeg_data = h5py.File(eeg_path, "r")["EEG/EEG"]
            cprint(f"h5 file shape: {eeg_data.shape}", "cyan")

            # NOTE: Resampling will be done twice
            X = sequential_load(
                data=eeg_data,
                bufsize=args.eeg_load_bufsize,
                preproc_func=partial(
                    load_resample, down=eeg_data.shape[1] / args.brain_orig_sfreq
                ),
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
            brain=X,
            segment_len=int(args.brain_resample_sfreq * args.seq_len),
            shift=shift,
        )

        cprint(f"Session {i} EEG: {X.shape}", "cyan")
        cprint(f"Session {i} face: {Y.shape}", "cyan")
        if len(X) > len(Y):
            X = X[: len(Y)]
        elif len(X) < len(Y):
            Y = Y[: len(X)]
            drop_segments = np.delete(drop_segments, np.where(drop_segments >= len(X)))

        X = np.delete(X, drop_segments, axis=0)
        Y = np.delete(Y, drop_segments, axis=0)

        cprint(f"Session {i} EEG (after drop): {X.shape}", "cyan")
        cprint(f"Session {i} face (after drop): {Y.shape}", "cyan")

        data_dir = f"data/preprocessed/uhd/{args.preproc_name}/S{i}/"
        os.makedirs(data_dir, exist_ok=True)
        np.save(data_dir + "brain.npy", X)
        h5_save(data_dir + "face.h5", Y)


if __name__ == "__main__":
    main()
