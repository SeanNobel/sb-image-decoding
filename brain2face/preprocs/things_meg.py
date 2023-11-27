"""
Here I use the preprocessed data in Hebart et al., 2023. It looks that scaling and baseline correction
is performed but clamping is not, which is different from the Meta paper. I resample the data from 200Hz
to 120Hz as in the Meta paper.
"""
import os, sys
import numpy as np
import mne
import torch
from termcolor import cprint
from natsort import natsorted
import hydra
from omegaconf import DictConfig


def make_refined_split_from_file(epochs: mne.Epochs, category_idxs: np.ndarray):
    """Old version of make_split."""
    # NOTE: Event ids in the preprocessed data seem to start from 1.
    trials = epochs.events[:, -1] - 1  # ( 27048, )

    # NOTE: There can be some missing events as they were not presented to the subject.
    events, event_counts = np.unique(trials, return_counts=True)  # ( 22248 + 1, )
    # NOTE: 1 is for train images, 12 is for test images, and 2400 is for fixation.
    assert np.array_equal(np.unique(event_counts), [1, 12, 2400])

    test_events = np.take(events, np.where(event_counts == 12)[0])  # ( 200, )
    assert len(test_events) == 200

    test_categories = category_idxs[test_events]  # ( 200, )
    assert len(test_categories) == 200 & len(np.unique(test_categories)) == 200

    train_event_idxs = np.where(
        np.logical_not(np.isin(category_idxs, test_categories))
    )[0]
    test_event_idxs = np.where(np.isin(category_idxs, test_categories))[0]
    assert len(train_event_idxs) + len(test_event_idxs) == len(category_idxs)

    train_trial_idxs = np.where(np.isin(trials, train_event_idxs))[0]
    test_trial_idxs = np.where(np.isin(trials, test_event_idxs))[0]
    assert len(train_trial_idxs) + len(test_trial_idxs) + 2400 == len(epochs)

    return train_trial_idxs, test_trial_idxs


@hydra.main(
    version_base=None, config_path="../../configs/thingsmeg", config_name="clip"
)
def run(args: DictConfig) -> None:
    meg_paths = [
        os.path.join(args.meg_dir, f"preprocessed_P{i+1}-epo.fif") for i in range(4)
    ]

    for subject_id, meg_path in enumerate(meg_paths):
        cprint(f"==== Processing subject {subject_id+1} ====", "cyan")

        cprint("> Loading epochs...", "cyan")
        epochs = mne.read_epochs(meg_path)

        cprint(f"> Resampling epochs to {args.brain_resample_sfreq}Hz...", "cyan")
        epochs = epochs.resample(args.brain_resample_sfreq, n_jobs=8)

        X = torch.from_numpy(epochs.get_data()).to(torch.float32)
        # ( 27048, 271, segment_len )

        cprint(f"MEG P{subject_id+1}: {X.shape}", "cyan")

        torch.save(
            X, os.path.join(args.preprocessed_data_dir, f"MEG_P{subject_id+1}.pt")
        )


if __name__ == "__main__":
    run()
