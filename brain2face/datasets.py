import os, sys
import torch
import numpy as np
import glob
import mne
import mediapipe as mp
from typing import Union, List, Optional, Callable
from termcolor import cprint
from omegaconf import DictConfig

mne.set_log_level(verbose="WARNING")
mp_face_mesh = mp.solutions.face_mesh


class Brain2FaceCLIPDatasetBase(torch.utils.data.Dataset):
    def __init__(
        self,
        args,
        session_paths: List[str],
        train: bool = True,
        y_reformer: Optional[Callable] = None,
    ) -> List[torch.Tensor]:
        super().__init__()

        # NOTE: No need to be natsorted.
        # NOTE: Selecting directories with preprocessed data.
        session_paths = self.drop_bads(session_paths)

        # NOTE: picks 20% sessions for test, without considering subjects' identity
        if args.split == "subject_random":
            split_idx = int(len(session_paths) * args.train_ratio)
            if train:
                session_paths = session_paths[:split_idx]
            else:
                session_paths = session_paths[split_idx:]

        # NOTE: each subject has one or two test sessions
        # FIXME: Need to add subject names with underscore to S0, S1, ... folders for this to work
        elif args.split == "subject_each":
            subject_names = list(set([path.split("_")[-1] for path in session_paths]))
            cprint(f"Subject names: {subject_names}", color="cyan")

            _session_paths = []
            for name in subject_names:
                subsession_paths = [path for path in session_paths if name in path]
                # print(subsession_paths)

                split_idx = int(len(subsession_paths) * args.train_ratio)
                if train:
                    subsession_paths = subsession_paths[:split_idx]
                else:
                    subsession_paths = subsession_paths[split_idx:]

                _session_paths += subsession_paths

            session_paths = _session_paths.copy()

        if args.split == "subject_each":
            self.num_subjects = len(subject_names)
        else:
            self.num_subjects = len(session_paths)
        cprint(f"Num subjects: {self.num_subjects}", color="cyan")

        X_list = []
        Y_list = []
        subject_idx_list = []
        for subject_idx, subject_path in enumerate(session_paths):
            X = torch.from_numpy(np.load(subject_path + "/brain.npy").astype(np.float32))
            Y = torch.from_numpy(np.load(subject_path + "/face.npy").astype(np.float32))

            if y_reformer is not None:
                Y = y_reformer(Y)

            if args.split == "deep":
                assert X.shape[0] == Y.shape[0]
                split_idx = int(X.shape[0] * args.train_ratio)
                if train:
                    X = X[:split_idx]
                    Y = Y[:split_idx]
                else:
                    X = X[split_idx:]
                    Y = Y[split_idx:]

            # NOTE: identify subject for subject_each split
            if args.split == "subject_each":
                name = subject_path.split("_")[-1]
                subject_idx = np.where(np.array(subject_names) == name)[0][0]
                # print(subject_idx)

            subject_idx *= torch.ones(X.shape[0], dtype=torch.uint8)
            print(f"X: {X.shape} | Y: {Y.shape} | subject_idx: {subject_idx.shape}")

            X_list.append(X)
            Y_list.append(Y)
            subject_idx_list.append(subject_idx)

            del X, Y, subject_idx

        self.session_lengths = [len(X) for X in X_list]

        self.subject_idx = torch.cat(subject_idx_list)
        del subject_idx_list
        cprint(f"self.subject_idx: {self.subject_idx.shape}", color="cyan")

        self.X = torch.cat(X_list)
        del X_list
        cprint(f"self.X: {self.X.shape}", color="cyan")

        return Y_list

        if args.chance:
            self.Y = self.Y[torch.randperm(self.Y.shape[0])]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i], self.subject_idx[i]

    @staticmethod
    def drop_bads(_subject_paths):
        subject_paths = []
        for path in _subject_paths:
            if os.path.exists(path + "brain.npy") and os.path.exists(path + "face.npy"):
                subject_paths.append(path)
        return subject_paths


class Brain2FaceYLabECoGDataset(Brain2FaceCLIPDatasetBase):
    def __init__(self, args, train: bool = True) -> None:
        session_paths = glob.glob("data/YLab/" + args.preproc_name + "/*/")
        Y_list = super().__init__(args, session_paths, train)

        self.Y = torch.cat(Y_list)
        cprint(f"self.Y: {self.Y.shape}", color="cyan")
        del Y_list


class Brain2FaceStyleGANDataset(Brain2FaceCLIPDatasetBase):
    def __init__(self, args, train: bool = True) -> None:
        session_paths = glob.glob(
            "data/preprocessed/stylegan/" + args.preproc_name + "/*/"
        )
        Y_list = super().__init__(
            args, session_paths, train, y_reformer=self.reshape_stylegan_latent
        )

        self.Y = torch.cat(Y_list)
        cprint(f"self.Y: {self.Y.shape}", color="cyan")
        del Y_list

    @staticmethod
    def reshape_stylegan_latent(Y: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            Y (torch.Tensor): ( samples, segment_len=90, styles=18, features=512 )
        Returns:
            Y (torch.Tensor): ( samples, features=512, segment_len*sub_styles=360 )
        """
        # Take four styles (TODO: take all styles. But will run out of memory)
        Y = Y[:, :, 4:8]

        # NOTE: squash latent layers to time dimension ( samples, segment_len*styles=360, features )
        Y = Y.contiguous().view(Y.shape[0], -1, Y.shape[-1])

        Y = Y.permute(0, 2, 1)  # ( samples, features=512, segment_len*styles=360 )

        return Y


if __name__ == "__main__":
    from hydra import initialize, compose

    with initialize(version_base=None, config_path="../configs/"):
        args = compose(config_name="ylab_ecog.yaml")

    dataset = Brain2FaceYLabECoGDataset(args)
