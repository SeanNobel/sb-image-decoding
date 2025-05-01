import sys
from os.path import join
import numpy as np
import torch
import torch.utils
from einops import repeat
from PIL import Image
from glob import glob
from natsort import natsorted
from termcolor import cprint
from tqdm import tqdm


class ThingsEEG2CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, args, train=True):
        split = "training" if train else "test"
        self.num_subjects = 10
        self.num_repetitions = 4 if train else 80

        subject_dirs = natsorted(glob(join(args.data_dir, "sub-*/")))
        assert len(subject_dirs) == self.num_subjects, f"Expected {self.num_subjects} subjects, but got {len(subject_dirs)}."

        self.X, self.subject_idxs = [], []
        for i, subject_dir in enumerate(tqdm(subject_dirs, desc="Loading EEG data")):
            path = join(subject_dir, f"preprocessed_eeg_{split}.npy")
            eeg = np.load(path, allow_pickle=True)[()]["preprocessed_eeg_data"]
            # ( images, repetitions, channels, time )
            assert eeg.shape[1] == self.num_repetitions, f"Expected {self.num_repetitions} repetitions, but got {eeg.shape[1]}."

            eeg = eeg.reshape(-1, eeg.shape[-2], eeg.shape[-1])
            # ( image * repetitions, channels, time )

            self.X.append(torch.from_numpy(eeg).to(torch.float32))
            self.subject_idxs.extend([i] * eeg.shape[0])

        self.X = torch.cat(self.X, dim=0)

        self.Y = torch.load(join(args.data_dir, f"{split}_image_embs.pt")) # ( images, d )
        self.Y = repeat(self.Y, "n d -> (s n r) d", s=self.num_subjects, r=self.num_repetitions)
        # ( subjects, images * repetitions, d )
        
        assert len(self.X) == len(self.Y), f"EEG and image embeddings have different lengths: {len(self.X)} vs {len(self.Y)}."

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.subject_idxs[i], self.Y[i]


if __name__ == "__main__":
    from hydra import initialize, compose

    with initialize(config_path="../../configs/thingseeg2", version_base=None):
        args = compose(config_name="clip")

    dataset = ThingsEEG2CLIPDataset(args, train=True)
