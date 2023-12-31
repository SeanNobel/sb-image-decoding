import os, sys
import torch
import numpy as np
import clip
import mne
from PIL import Image
from termcolor import cprint
from glob import glob
from natsort import natsorted
from tqdm import tqdm
from typing import Tuple, List
import gc


class ThingsMEGCLIPDataset(torch.utils.data.Dataset):
    def __init__(self, args) -> None:
        super().__init__()

        self.large_test_set = args.large_test_set

        # categories = np.loadtxt(
        #     os.path.join(args.metadata_dir, "Concept-specific/unique_id.csv"),
        #     dtype=str,
        # )
        # category_idxs = np.loadtxt(
        #     os.path.join(args.metadata_dir, "Concept-specific/image_concept_index.csv"),
        #     dtype=int,
        # )
        # cprint(f"Categories: {categories.shape} | category indices: {category_idxs.shape}", "cyan")  # fmt: skip

        # NOTE: Some categories
        high_categories = np.loadtxt(
            os.path.join(
                args.things_dir, "27 higher-level categories/category_mat_manual.tsv"
            ),
            dtype=int,
            delimiter="\t",
            skiprows=1,
        )  # ( 1854, 27 )

        preproc_dir = os.path.join(args.preprocessed_data_dir, args.preproc_name)

        sample_attrs_paths = [
            os.path.join(args.thingsmeg_dir, f"sourcedata/sample_attributes_P{i+1}.csv")
            for i in range(4)
        ]

        X_list = []
        Y_list = []
        subject_idxs_list = []
        categories_list = []
        y_idxs_list = []
        train_idxs_list = []
        test_idxs_list = []
        for subject_id, sample_attrs_path in enumerate(sample_attrs_paths):
            # MEG
            X_list.append(
                torch.load(os.path.join(preproc_dir, f"MEG_P{subject_id+1}.pt"))
            )
            # ( 27048, 271, segment_len )

            # Images
            Y_list.append(
                torch.load(os.path.join(preproc_dir, f"Images_P{subject_id+1}.pt"))
            )

            # Indexes
            sample_attrs = np.loadtxt(
                sample_attrs_path, dtype=str, delimiter=",", skiprows=1
            )  # ( 27048, 18 )

            categories_list.append(torch.from_numpy(sample_attrs[:, 2].astype(int)))
            y_idxs_list.append(torch.from_numpy(sample_attrs[:, 1].astype(int)))

            subject_idxs_list.append(
                torch.ones(len(sample_attrs), dtype=int) * subject_id
            )

            # Split
            train_idxs, test_idxs = self.make_split(
                sample_attrs, large_test_set=self.large_test_set
            )
            idx_offset = len(sample_attrs) * subject_id
            train_idxs_list.append(train_idxs + idx_offset)
            test_idxs_list.append(test_idxs + idx_offset)

        self.X = torch.cat(X_list, dim=0)
        self.Y = torch.cat(Y_list, dim=0)
        self.subject_idxs = torch.cat(subject_idxs_list, dim=0)

        self.categories = torch.cat(categories_list) - 1
        assert torch.equal(self.categories.unique(), torch.arange(self.categories.max() + 1))  # fmt: skip
        self.num_categories = len(self.categories.unique())

        self.high_categories = self.to_high_categories(self.categories, high_categories)
        self.num_high_categories = self.high_categories.max() + 1

        self.y_idxs = torch.cat(y_idxs_list) - 1
        assert torch.equal(self.y_idxs.unique(), torch.arange(self.y_idxs.max() + 1))

        self.train_idxs = torch.cat(train_idxs_list, dim=0)
        self.test_idxs = torch.cat(test_idxs_list, dim=0)

        cprint(f"X: {self.X.shape} | Y: {self.Y.shape} | subject_idxs: {self.subject_idxs.shape} | train_idxs: {self.train_idxs.shape} | test_idxs: {self.test_idxs.shape}", "cyan")  # fmt: skip

        self.subject_names = [f"s0{i+1}" for i in range(4)]

        if args.chance:
            self.X = self.X[torch.randperm(len(self.X))]

        del X_list, Y_list, categories_list, y_idxs_list, subject_idxs_list, train_idxs_list, test_idxs_list  # fmt: skip
        gc.collect()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.Y[i], self.subject_idxs[i], self.y_idxs[i], self.categories[i], self.high_categories[i]  # fmt: skip

    @staticmethod
    def make_split(
        sample_attrs: np.ndarray,
        large_test_set: bool = True,
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

        if not large_test_set:
            # Small test set
            train_trial_idxs = np.where(trial_types == "exp")[0]  # ( 22248, )
            test_trial_idxs = np.where(trial_types == "test")[0]  # ( 2400, )

            assert len(train_trial_idxs) == 22248 and len(test_trial_idxs) == 2400
        else:
            category_idxs = sample_attrs[:, 2].astype(int)  # ( 27048, )

            test_trial_idxs = np.where(trial_types == "test")[0]  # ( 2400, )
            test_category_idxs = np.unique(np.take(category_idxs, test_trial_idxs))
            # ( 200, )

            test_trial_idxs = np.where(
                np.logical_and(
                    np.isin(category_idxs, test_category_idxs),
                    np.logical_not(trial_types == "test"),
                )
            )[0]
            # ( 2400, )

            train_trial_idxs = np.where(
                np.logical_and(
                    trial_types == "exp",
                    np.logical_not(np.isin(category_idxs, test_category_idxs)),
                )
            )[0]
            # ( 19848, )

            assert len(train_trial_idxs) == 19848 and len(test_trial_idxs) == 2400

        return torch.from_numpy(train_trial_idxs), torch.from_numpy(test_trial_idxs)

    def to_high_categories(
        self, categories: torch.Tensor, high_categories: np.ndarray
    ) -> torch.Tensor:
        """_summary_
        Args:
            categories ( 27048 * 4, ): Elements are integers of [0, 1854].
                End value 1854 is for catch trials.
            high_categories ( 1854, 27 ): Each row is a zero to three -hot vector.
        Returns:
            high_categories ( 27048 * 4, ): Elements are integers of [0, 27].
        """
        # Set categories that are not in any of higher categories as "uncategorized".
        unc = np.where(high_categories.sum(axis=1) == 0)[0]
        # This takes the first higher-category for categories that are in multiple higher-categories.
        high_categories = np.argmax(high_categories, axis=1)  # ( 1854, )

        # Higher-category 27 for "uncategorized" and catch trials.
        high_categories[unc] = high_categories.max() + 1  # ( 1854, )
        high_categories = np.append(high_categories, high_categories.max())  # ( 1855, )

        return torch.from_numpy(high_categories)[categories]


if __name__ == "__main__":
    from hydra import initialize, compose

    with initialize(version_base=None, config_path="../../configs/thingsmeg"):
        args = compose(config_name="clip")

    dataset = ThingsMEGCLIPDataset(args)
