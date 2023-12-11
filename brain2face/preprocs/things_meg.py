"""
Here I use the preprocessed data in Hebart et al., 2023. It looks that scaling and baseline correction
is performed but clamping is not, which is different from the Meta paper. I resample the data from 200Hz
to 120Hz as in the Meta paper.
"""
import os, sys
import numpy as np
import mne
import torch
import clip
from PIL import Image
from functools import partial
from termcolor import cprint
from natsort import natsorted
from tqdm import tqdm
from typing import Tuple, List
import hydra
from omegaconf import DictConfig

from brain2face.utils.brain_preproc import scale_clamp
from brain2face.utils.train_utils import sequential_apply


@torch.no_grad()
def encode_images(y_list: List[str], preprocess, clip_model, device) -> torch.Tensor:
    Y = torch.stack(
        [
            preprocess(Image.open(y).convert("RGB"))
            for y in tqdm(y_list, desc="Preprocessing images")
        ]
    )

    Y = sequential_apply(
        Y,
        clip_model.encode_image,
        batch_size=32,
        device=device,
        desc="Encoding images",
    ).float()

    return Y


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
    sample_attrs_paths = [
        os.path.join(args.thingsmeg_dir, f"sourcedata/sample_attributes_P{i+1}.csv")
        for i in range(4)
    ]

    save_dir = os.path.join(args.preprocessed_data_dir, args.preproc_name)
    os.makedirs(save_dir, exist_ok=True)

    for subject_id, (meg_path, sample_attrs_path) in enumerate(zip(meg_paths, sample_attrs_paths)):  # fmt: skip
        cprint(f"==== Processing subject {subject_id+1} ====", "cyan")

        # -----------------
        #        MEG
        # -----------------
        if not args.skip_meg:
            cprint("> Loading epochs...", "cyan")
            epochs = mne.read_epochs(meg_path)

            cprint(f"> Resampling epochs to {args.brain_resample_sfreq}Hz...", "cyan")
            epochs.resample(args.brain_resample_sfreq, n_jobs=8)

            cprint(f"> Scale and clamping epochs to ±{args.clamp_lim}...", "cyan")
            epochs.apply_function(
                partial(scale_clamp, scale_transposed=False, clamp_lim=args.clamp_lim),
                n_jobs=8,
            )

            cprint("> Baseline correction...", "cyan")
            epochs.apply_baseline((None, 0))

            X = torch.from_numpy(epochs.get_data()).to(torch.float32)
            # ( 27048, 271, segment_len )

            cprint(f"MEG P{subject_id+1}: {X.shape}", "cyan")

            torch.save(X, os.path.join(save_dir, f"MEG_P{subject_id+1}.pt"))

        # -----------------
        #      Images
        # -----------------
        if args.vision.pretrained and not args.skip_images:
            device = f"cuda:{args.cuda_id}"

            clip_model, preprocess = clip.load(args.vision.pretrained_model)
            clip_model = clip_model.eval().to(device)

            sample_attrs = np.loadtxt(
                sample_attrs_path, dtype=str, delimiter=",", skiprows=1
            )

            y_list = []
            for path in sample_attrs[:, 8]:
                if "images_meg" in path:
                    y_list.append(
                        os.path.join(args.images_dir, "/".join(path.split("/")[1:]))
                    )
                elif "images_test_meg" in path:
                    y_list.append(
                        os.path.join(
                            args.images_dir,
                            "_".join(os.path.basename(path).split("_")[:-1]),
                            os.path.basename(path),
                        )
                    )
                elif "images_catch_meg" in path:
                    y_list.append(os.path.join(args.images_dir, "black.jpg"))
                else:
                    raise ValueError(f"Unknown image path type: {path}")

            Y = encode_images(y_list, preprocess, clip_model, device)

            cprint(f"Images P{subject_id+1}: {Y.shape}", "cyan")

            torch.save(Y, os.path.join(save_dir, f"Images_P{subject_id+1}.pt"))


if __name__ == "__main__":
    run()
