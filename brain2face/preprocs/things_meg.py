"""
Here I use the preprocessed data in Hebart et al., 2023. It looks that scaling and baseline correction
is performed but clamping is not, which is different from the Meta paper. I resample the data from 200Hz
to 120Hz as in the Meta paper.
"""
import os, sys
import numpy as np
import torch
import mne
from termcolor import cprint
from natsort import natsorted
import hydra
from omegaconf import DictConfig


def modified_split(epochs: mne.Epochs, category_idxs: np.ndarray):
    """
    Splits by Meta, modified from Hebart et al., 2023.
    """
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
    categories = natsorted(os.listdir(args.images_dir))
    categories = [
        c for c in categories if os.path.isdir(os.path.join(args.images_dir, c))
    ]  # ( 1854, )
    cprint(f"Found {len(categories)} categories.", "cyan")

    num_imgs_per_cat = [
        len(os.listdir(os.path.join(args.images_dir, c))) for c in categories
    ]  # ( 1854, )
    category_idxs = np.repeat(
        np.arange(len(categories)), num_imgs_per_cat
    )  # ( 26107, )
    cprint(f"Total number of images: {len(category_idxs)}", "cyan")

    meg_paths = [
        os.path.join(args.meg_dir, f"preprocessed_P{i+1}-epo.fif") for i in range(4)
    ]

    train_X_list = []
    train_y_list = []
    train_subject_idxs_list = []
    test_X_list = []
    test_y_list = []
    test_subject_idxs_list = []
    for subject_id, meg_path in enumerate(meg_paths):
        cprint(f"==== Processing subject {subject_id+1} ====", "cyan")
        cprint("> Loading epochs...", "cyan")
        epochs = mne.read_epochs(meg_path)

        train_trial_idxs, test_trial_idxs = modified_split(epochs, category_idxs)

        cprint(f"> Resampling epochs to {args.brain_resample_sfreq}Hz...", "cyan")
        epochs = epochs.resample(args.brain_resample_sfreq, n_jobs=8)

        X = epochs.get_data()  # ( 27048, 271, segment_len )
        y = epochs.events[:, -1]  # ( 27048, )
        subject_idxs = np.ones_like(y) * subject_id  # ( 27048, )

        train_X_list.append(X[train_trial_idxs])
        train_y_list.append(y[train_trial_idxs])
        train_subject_idxs_list.append(subject_idxs[train_trial_idxs])
        test_X_list.append(X[test_trial_idxs])
        test_y_list.append(y[test_trial_idxs])
        test_subject_idxs_list.append(subject_idxs[test_trial_idxs])

    train_X = np.concatenate(train_X_list, axis=0)
    train_y = np.concatenate(train_y_list, axis=0)
    train_subject_idxs = np.concatenate(train_subject_idxs_list, axis=0)
    test_X = np.concatenate(test_X_list, axis=0)
    test_y = np.concatenate(test_y_list, axis=0)
    test_subject_idxs = np.concatenate(test_subject_idxs_list, axis=0)

    cprint(
        f"[Train] X: {train_X.shape}, y: {train_y.shape}, subject_idxs: {train_subject_idxs.shape}",
        "cyan",
    )
    cprint(
        f"[Test] X: {test_X.shape}, y: {test_y.shape}, subject_idxs: {test_subject_idxs.shape}",
        "cyan",
    )

    np.savez(
        os.path.join(args.preprocessed_data_dir, args.preproc_name),
        train_X=train_X,
        train_y=train_y,
        train_subject_idxs=train_subject_idxs,
        test_X=test_X,
        test_y=test_y,
        test_subject_idxs=test_subject_idxs,
    )


if __name__ == "__main__":
    run()
