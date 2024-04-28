import os
import torch


class ImageNetEEGBrainDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        data = torch.load(os.path.join(args.eeg_dir, "eeg_5_95_std.pth"))

        self.num_subjects = 6

        # make it start from 0
        self.subject_idxs = torch.tensor([d["subject"] for d in data["dataset"]]) - 1

        # The dataset authors say 'we discarded the first 20 samples (20 ms) to reduce interference from the previous image and then cut the signal to a common length of 440 samples'. https://github.com/perceivelab/eeg_visual_classification
        self.eeg = torch.stack([d["eeg"][:, 20:][:, :440] for d in data["dataset"]])

    def __len__(self):
        return len(self.eeg)

    def __getitem__(self, idx):
        return self.eeg[idx], self.subject_idxs[idx]
