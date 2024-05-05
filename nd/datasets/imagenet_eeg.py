import os
import torch
import torch.nn as nn
import ml_collections
import omegaconf
from typing import Dict, Union, Optional
import itertools


class ImageNetEEGBrainDataset(torch.utils.data.Dataset):
    def __init__(self, args: Union[omegaconf.DictConfig, ml_collections.FrozenConfigDict], cv=0):
        super().__init__()

        self.num_subjects = 6
        self.preproc_dir = os.path.join(args.preproc_dir, args.preproc_name)

        self.X = torch.load(os.path.join(self.preproc_dir, "eeg.pt"))
        self.subject_idxs = torch.load(os.path.join(self.preproc_dir, "subject_idxs.pt"))

        self.train_idxs = torch.load(os.path.join(self.preproc_dir, "train_idxs", f"cv{cv+1}.pt"))
        self.test_idxs = torch.load(os.path.join(self.preproc_dir, "test_idxs", f"cv{cv+1}.pt"))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.subject_idxs[i]


class ImageNetEEGCLIPDataset(ImageNetEEGBrainDataset):
    def __init__(self, args):
        super().__init__(args)

        self.Y = torch.load(os.path.join(self.preproc_dir, "images_clip.pt"))

    def __getitem__(self, i):
        return self.X[i], self.subject_idxs[i], self.Y[i]


class ImageNetEEGMomentsDataset(ImageNetEEGBrainDataset):
    def __init__(self, args: ml_collections.FrozenConfigDict):
        super().__init__(args)

        self.Y = torch.load(os.path.join(self.preproc_dir, "image_moments.pt"))

        self.vis_samples: Dict[str, torch.Tensor] = {
            "train_brain": self.X[self.train_idxs[: args.n_vis_samples]],
            "train_moments": self.Y[self.train_idxs[: args.n_vis_samples]],
            "train_subject_idxs": self.subject_idxs[self.train_idxs[: args.n_vis_samples]],
            "test_brain": self.X[self.test_idxs[: args.n_vis_samples]],
            "test_moments": self.Y[self.test_idxs[: args.n_vis_samples]],
            "test_subject_idxs": self.subject_idxs[self.test_idxs[: args.n_vis_samples]],
        }

    def __getitem__(self, i):
        return self.X[i], self.Y[i], self.subject_idxs[i]

    def unpreprocess(self, v: torch.Tensor):
        return (0.5 * (v + 1.0)).clamp(0.0, 1.0)

    @property
    def data_shape(self):
        return 4, 32, 32

    @property
    def fid_stat(self):
        return os.path.join(self.preproc_dir, "fid_stats_imneteeg.npz")
