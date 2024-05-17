import os, sys
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms.functional import to_tensor
import ml_collections
from PIL import Image
import omegaconf
from typing import Dict, Union, Optional
from termcolor import cprint


class ImageNetEEGBrainDataset(torch.utils.data.Dataset):
    def __init__(self, args: Union[omegaconf.DictConfig, ml_collections.FrozenConfigDict], cv=0):
        super().__init__()

        self.num_subjects = 6
        self.preproc_dir = os.path.join(args.preproc_dir, args.preproc_name)

        self.X = torch.load(os.path.join(self.preproc_dir, "eeg.pt"))
        self.subject_idxs = torch.load(os.path.join(self.preproc_dir, "subject_idxs.pt"))
        assert self.subject_idxs.max() == self.num_subjects - 1

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

        self.p_uncond = args.p_uncond
        self.empty_token = torch.tensor(self.num_subjects)

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
        cond = self.empty_token if random.random() < self.p_uncond else self.subject_idxs[i]

        return self.X[i], self.Y[i], self.subject_idxs[i], cond

    def unpreprocess(self, v: torch.Tensor):
        return (0.5 * (v + 1.0)).clamp(0.0, 1.0)

    @property
    def data_shape(self):
        return 4, 32, 32

    @property
    def fid_stat(self):
        return os.path.join(self.preproc_dir, "fid_stats_imneteeg.npz")


class ImageNetEEGMomentsDatasetCond(ImageNetEEGMomentsDataset):
    def __init__(self, args: ml_collections.FrozenConfigDict):
        super().__init__(args)

        del self.empty_token

    def __getitem__(self, i):
        if random.random() < self.p_uncond:
            cond, cond_subject_idx = torch.zeros_like(self.X[i]), torch.tensor(0)
        else:
            cond, cond_subject_idx = self.X[i], self.subject_idxs[i]

        return self.X[i], self.Y[i], self.subject_idxs[i], cond, cond_subject_idx


class ImageNetEEGEvalDataset(ImageNetEEGMomentsDataset):
    def __init__(self, args: ml_collections.FrozenConfigDict):
        super().__init__(args)

        del self.Y, self.vis_samples

        self.image_paths = np.loadtxt(os.path.join(self.preproc_dir, "image_paths.txt"), dtype=str)
        assert len(self.image_paths) == len(self.X)

    def __getitem__(self, i):
        Y = Image.open(self.image_paths[i]).convert("RGB")

        return self.X[i], to_tensor(Y), self.subject_idxs[i]
