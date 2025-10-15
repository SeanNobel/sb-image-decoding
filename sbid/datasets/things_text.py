import os, sys
import numpy as np
import torch
import csv
from termcolor import cprint

import clip

from sbid.datasets.things_meg import ThingsCLIPDatasetBase


class ThingsTextCLIPDataset(ThingsCLIPDatasetBase):
    def __init__(self, args) -> None:
        super().__init__()

        self.num_clip_tokens = args.num_clip_tokens
        num_noises = args.num_noises

        self.preproc_dir = os.path.join(args.preprocessed_data_dir, args.preproc_name)

        # NOTE: Using only subject 1 as image-text pairs are the same across subjects.
        sample_attrs_path = os.path.join(
            args.thingsmeg_dir, "sourcedata/sample_attributes_P1.csv"
        )
        sample_attrs = np.loadtxt(
            sample_attrs_path, dtype=str, delimiter=",", skiprows=1
        )  # ( 27048, 18 )

        self.num_samples = len(sample_attrs)

        # -------------
        #    Images
        # -------------
        vision_path = os.path.join(self.preproc_dir, "Images_P1.pt")
        self.Y = self._extract_tokens(
            torch.load(vision_path, map_location="cpu"), tokens=args.align_tokens
        )  # ( 27048, 1, 768 )

        # -------------
        #     Text
        # -------------
        keys = [os.path.splitext(os.path.basename(key))[0] for key in sample_attrs[:, 8]]  # fmt: skip

        texts = []
        for n in range(num_noises):
            texts_path = os.path.join(
                self.preproc_dir, f"noise_level_{args.noise_level}", f"Texts_N{n}.csv"
            )
            texts_dict = {key: text for key, text in csv.reader(open(texts_path, "r"))}
            texts += [texts_dict[key] if key in texts_dict else "" for key in keys]

        self.X = clip.tokenize(texts)  # ( 27048 * num_noises, 77 )

        # -------------
        #    Others
        # -------------
        self.categories = torch.from_numpy(sample_attrs[:, 2].astype(int))
        self.num_categories = len(self.categories.unique())

        self.y_idxs = torch.from_numpy(sample_attrs[:, 1].astype(int)) - 1

        self.train_idxs, self.test_idxs = self.make_split(
            sample_attrs, large_test_set=args.large_test_set
        )

        # ----------------------------------
        # Repeat for number of noisy samples
        # ----------------------------------
        self.Y = self.Y.repeat(num_noises, 1, 1)  # ( 27048 * num_noises, 1, 768 )
        self.categories = self.categories.repeat(num_noises)
        self.y_idxs = self.y_idxs.repeat(num_noises)
        self.train_idxs = torch.cat(
            [self.train_idxs + i * self.num_samples for i in range(num_noises)]
        )
        self.test_idxs = torch.cat(
            [self.test_idxs + i * self.num_samples for i in range(num_noises)]
        )

        cprint(f"X: {self.X.shape} | Y: {self.Y.shape} | train_idxs: {self.train_idxs.shape} | test_idxs: {self.test_idxs.shape}", "cyan")  # fmt: skip

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int):
        return self.X[i], self.Y[i], self.y_idxs[i], self.categories[i]
