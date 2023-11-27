import os, sys
import torch
import numpy as np
import mne
from PIL import Image
from termcolor import cprint
from glob import glob
from natsort import natsorted
from typing import Tuple

import clip


class ThingsMEGCLIPDataset(torch.utils.data.Dataset):
    def __init__(self, args) -> None:
        super().__init__()

        categories = np.loadtxt(
            os.path.join(args.metadata_dir, "Concept-specific/unique_id.csv"),
            dtype=str,
        )
        category_idxs = np.loadtxt(
            os.path.join(args.metadata_dir, "Concept-specific/image_concept_index.csv"),
            dtype=int,
        )
        cprint(f"Categories: {categories.shape} | category indices: {category_idxs.shape}", "cyan")  # fmt: skip

        meg_paths = [
            os.path.join(args.preprocessed_data_dir, f"MEG_P{i+1}.pt") for i in range(4)
        ]
        sample_attrs_paths = [
            os.path.join(args.thingsmeg_dir, f"sourcedata/sample_attributes_P{i+1}.csv")
            for i in range(4)
        ]

        X_list = []
        self.y_list = []
        subject_idxs_list = []
        train_idxs_list = []
        test_idxs_list = []
        for subject_id, (meg_path, sample_attrs_path) in enumerate(
            zip(meg_paths, sample_attrs_paths)
        ):
            # MEG
            X_list.append(torch.load(meg_path))  # ( 27048, 271, segment_len )

            # Image (path) and subject index
            sample_attrs = np.loadtxt(
                sample_attrs_path, dtype=str, delimiter=",", skiprows=1
            )  # ( 27048, 18 )

            self.y_list += [
                os.path.join(args.images_dir, "/".join(path.split("/")[1:]))
                if not "images_test" in path
                else os.path.join(
                    args.images_dir,
                    "_".join(os.path.basename(path).split("_")[:-1]),
                    os.path.basename(path),
                )
                for path in sample_attrs[:, 8]
            ]

            subject_idxs_list.append(
                torch.ones(len(sample_attrs), dtype=int) * subject_id
            )

            # Split
            train_idxs, test_idxs = self.make_split(
                sample_attrs, refined=args.refined_split
            )
            idx_offset = len(sample_attrs) * subject_id
            train_idxs_list.append(train_idxs + idx_offset)
            test_idxs_list.append(test_idxs + idx_offset)

        self.X = torch.cat(X_list, dim=0)
        self.subject_idxs = torch.cat(subject_idxs_list, dim=0)

        self.train_idxs = torch.cat(train_idxs_list, dim=0)
        self.test_idxs = torch.cat(test_idxs_list, dim=0)

        cprint(f"X: {self.X.shape} | y (paths): {len(self.y_list)} | subject_idxs: {self.subject_idxs.shape} | train_idxs: {self.train_idxs.shape} | test_idxs: {self.test_idxs.shape}", "cyan")  # fmt: skip

        if args.vision.pretrained:
            _, self.preprocess = clip.load(args.vision.pretrained_model)
        else:
            self.preprocess = None

        self.subject_names = [f"s0{i+1}" for i in range(4)]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i):
        Y = Image.open(self.y_list[i]).convert("RGB")

        if self.preprocess is not None:
            Y = self.preprocess(Y)
        else:
            raise NotImplementedError

        return self.X[i], Y, self.subject_idxs[i]

    @staticmethod
    def make_split(
        sample_attrs: np.ndarray,
        refined: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            sample_attrs ( 27048, 18 ): Elements are strs.
            refined (bool): If True, use splits by Meta, modified from Hebart et al., 2023.
        Returns:
            train_trial_idxs ( 22248, ): _description_
            test_trial_idxs ( 2400, ): _description_
        """
        trial_types = sample_attrs[:, 0]  # ( 27048, )

        if not refined:
            train_trial_idxs = np.where(trial_types == "exp")[0]  # ( 22248, )
            test_trial_idxs = np.where(trial_types == "test")[0]  # ( 2400, )

        else:
            category_idxs = sample_attrs[:, 2].astype(int)  # ( 27048, )

            test_trial_idxs = np.where(trial_types == "test")[0]  # ( 2400, )
            test_category_idxs = np.unique(np.take(category_idxs, test_trial_idxs))
            # ( 200, )

            test_trial_idxs = np.where(np.isin(category_idxs, test_category_idxs))[0]

            train_trial_idxs = np.where(
                np.logical_and(
                    trial_types == "exp",
                    np.logical_not(np.isin(category_idxs, test_category_idxs)),
                )
            )[0]

        return torch.from_numpy(train_trial_idxs), torch.from_numpy(test_trial_idxs)


if __name__ == "__main__":
    from hydra import initialize, compose

    with initialize(version_base=None, config_path="../../configs/thingsmeg"):
        args = compose(config_name="clip")

    dataset = ThingsMEGCLIPDataset(args)
