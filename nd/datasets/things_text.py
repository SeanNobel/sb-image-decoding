import os, sys
import numpy as np
import torch
import csv
from termcolor import cprint
from typing import List

import clip

from nd.datasets.things_meg import ThingsCLIPDatasetBase


class ThingsTextCLIPDataset(ThingsCLIPDatasetBase):
    def __init__(self, args) -> None:
        super().__init__()

        self.num_clip_tokens = args.num_clip_tokens

        self.preproc_dir = os.path.join(args.preprocessed_data_dir, args.preproc_name)

        # NOTE: Using only subject 1 as image-text pairs are the same across subjects.
        sample_attrs_path = os.path.join(
            args.thingsmeg_dir, "sourcedata/sample_attributes_P1.csv"
        )
        sample_attrs = np.loadtxt(
            sample_attrs_path, dtype=str, delimiter=",", skiprows=1
        )  # ( 27048, 18 )

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
        text_path = os.path.join(self.preproc_dir, "Captions.csv")
        text_dict = {key: text for key, text in csv.reader(open(text_path, "r"))}
        keys = [os.path.splitext(os.path.basename(key))[0] for key in sample_attrs[:, 8]]  # fmt: skip
        texts = [text_dict[key] if key in text_dict else "" for key in keys]

        if args.noise_level > 0:
            texts = self._add_noise(texts, args.noise_level)

        self.X = clip.tokenize(texts)  # ( 27048, 77 )
        self.num_samples = len(self.X)

        self.categories = torch.from_numpy(sample_attrs[:, 2].astype(int))
        self.num_categories = len(self.categories.unique())

        self.y_idxs = torch.from_numpy(sample_attrs[:, 1].astype(int)) - 1

        self.train_idxs, self.test_idxs = self.make_split(
            sample_attrs, large_test_set=args.large_test_set
        )

        Y_shape = self.Y.shape if hasattr(self, "Y") else "to be loaded in __getitem__"
        cprint(f"X: {self.X.shape} | Y: {Y_shape} | train_idxs: {self.train_idxs.shape} | test_idxs: {self.test_idxs.shape}", "cyan")  # fmt: skip

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, i: int):
        return self.X[i], self.Y[i], self.y_idxs[i], self.categories[i]

    @staticmethod
    def _add_noise(texts: List[str], noise_level: float) -> List[str]:
        """Randomly replace characters in text with noise_level probability."""
        noisy_texts = []
        for i, text in enumerate(texts):
            noisy_text = ""

            for char in text:
                if np.random.rand() < noise_level:
                    noisy_text += chr(np.random.randint(32, 127))
                else:
                    noisy_text += char

            noisy_texts.append(noisy_text)

            # cprint(text, "cyan")
            # cprint(noisy_text, "yellow")
            # if i > 10:
            #     sys.exit()

        return noisy_texts
