import os
import torch


class ImageNetEEGBrainDataset(torch.utils.data.Dataset):
    def __init__(self, args, cv=0):
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
