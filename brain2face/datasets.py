import os, sys
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
import numpy as np
import cv2
import glob
from natsort import natsorted
import mne
import mediapipe as mp
import h5py
from functools import partial
from typing import Union, List, Optional, Callable
from termcolor import cprint
from omegaconf import DictConfig

import clip
from clip.model import CLIP

from brain2face.utils.train_utils import sequential_apply
from brain2face.utils.preproc_utils import sequential_load

mne.set_log_level(verbose="WARNING")
mp_face_mesh = mp.solutions.face_mesh


class Brain2FaceCLIPDatasetBase(torch.utils.data.Dataset):
    def __init__(
        self,
        args,
        session_paths: List[str],
        train: bool,
        y_reformer: Callable,
    ) -> List[torch.Tensor]:
        super().__init__()

        # NOTE: No need to be natsorted.
        # NOTE: Selecting directories with preprocessed data.
        session_paths = self._drop_bads(session_paths)

        # DEBUG
        # session_paths = session_paths[:1]

        if args.split in ["subject_random", "subject_each"]:
            session_paths, self.num_subjects, subject_names = self._split_sessions(train)
        else:
            self.num_subjects = len(session_paths)

        cprint(f"Num subjects: {self.num_subjects}", color="cyan")

        X_list = []
        Y_list = []
        subject_idx_list = []
        for subject_idx, subject_path in enumerate(session_paths):
            X = torch.from_numpy(
                np.load(os.path.join(subject_path, "brain.npy")).astype(np.float32)
            )

            try:
                Y = h5py.File(os.path.join(subject_path, "face.h5"), "r")["data"]
                Y = sequential_load(data=Y, bufsize=256, preproc_func=y_reformer)
            except FileNotFoundError:
                Y = np.load(os.path.join(subject_path, "face.npy"))
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

        self.Y = torch.cat(Y_list)
        del Y_list
        cprint(f"self.Y: {self.Y.shape}", color="cyan")

        if args.chance:
            self.Y = self.Y[torch.randperm(self.Y.shape[0])]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i], self.subject_idx[i]

    @staticmethod
    def _drop_bads(_subject_paths):
        subject_paths = []
        for path in _subject_paths:
            if os.path.exists(path + "brain.npy") and os.path.exists(path + "face.npy"):
                subject_paths.append(path)
        return subject_paths

    @staticmethod
    def _split_sessions(train: bool):
        # NOTE: picks 20% sessions for test, without considering subjects' identity
        if args.split == "subject_random":
            split_idx = int(len(session_paths) * args.train_ratio)
            if train:
                session_paths = session_paths[:split_idx]
            else:
                session_paths = session_paths[split_idx:]

            num_subjects = len(session_paths)

            subject_names = None

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

            num_subjects = len(subject_names)

        else:
            raise ValueError

        return session_paths, num_subjects, subject_names


class Brain2FaceUHDDataset(Brain2FaceCLIPDatasetBase):
    def __init__(self, args, train: bool = True) -> None:
        session_paths = glob.glob("data/preprocessed/uhd/" + args.preproc_name + "/*/")

        if args.face.type == "dynamic":
            y_reformer = partial(self.transform_video, image_size=args.vivit.image_size)

        elif args.face.type == "static":
            if args.face.encoded:
                device = f"cuda:{args.cuda_id}"
                clip_model, preprocess = clip.load(args.face.clip_model, device=device)

                y_reformer = partial(
                    self.to_single_frame,
                    reduction=args.face.reduction,
                    clip_model=clip_model,
                    preprocess=preprocess,
                    device=device,
                )

            else:
                y_reformer = partial(self.to_single_frame, reduction=args.face.reduction)
        else:
            raise ValueError("Face type is only static or dynamic.")

        super().__init__(args, session_paths, train, y_reformer)

        # FIXME: I've forgot to take first 128 channels from 139 while preprocessing
        self.X = self.X[:, : args.num_channels]

    @staticmethod
    def to_single_frame(
        Y: np.ndarray,
        reduction: str = "extract",
        clip_model: Optional[CLIP] = None,
        preprocess: Optional[transforms.Compose] = None,
        device: str = "cuda",
    ) -> torch.Tensor:
        """Extracts single frame from video sequence, then encodes to pre-trained CLIP space.
        NOTE: This function is called from sequential_load(), so there's no need to split Y into batch.
        Args:
            Y: ( samples, segment_len=90, face_extractor.output_size=256, face_extractor.output_size=256, 3 )
            clip_model: Pretrained image encoder of Radford 2021.
            preprocess: Transforms for the pretrained image encoder.
        Returns:
            Y: ( samples, face_extractor.output_size=256, face_extractor.output_size=256, 3 )
                or ( samples, F )
        """
        if reduction == "extract":
            # NOTE: Take the frame in the middle
            Y = Y[:, Y.shape[1] // 2]
        elif reduction == "mean":
            Y = Y.mean(dim=1)
        else:
            raise ValueError("Reduction is either extract or mean.")
        # ( samples, face_extractor.output_size=256, face_extractor.output_size=256, 3)

        if clip_model is not None:
            assert preprocess is not None, "clip_model needs preprocess."

            Y = sequential_apply(Y, preprocess, batch_size=1).to(device)

            with torch.no_grad():
                # NOTE: CLIP model somehow returns fp16 (https://colab.research.google.com/github/openai/clip/blob/master/notebooks/Interacting_with_CLIP.ipynb#scrollTo=cuxm2Gt4Wvzt)
                Y = clip_model.encode_image(Y).cpu().float()

        else:
            Y = torch.from_numpy(Y).permute(0, 3, 1, 2)
            Y = Y.to(torch.float32) / 255.0

        return Y

    @staticmethod
    def transform_video(
        Y: np.ndarray, image_size: int, to_grayscale: bool = True
    ) -> torch.Tensor:
        """
        - Resizes the video frames if args.face_extractor.output_size != args.vivit.image_size
        - Reduces the video frames to single channel it specified
        - Scale [0 - 255] -> [0. - 1.]
        Args:
            Y: ( samples, segment_len=90, face_extractor.output_size=256, face_extractor.output_size=256, 3 )
            image_size: args.vivit.image_size
        Returns:
            Y: ( samples, segment_len=90, 1, vivit.image_size, vivit.image_size )
        """
        segment_len = Y.shape[1]

        Y = torch.from_numpy(Y).view(-1, *Y.shape[-3:]).permute(0, 3, 1, 2)
        # ( samples*segment_len, 3, size, size )

        video_transforms = [
            transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
        ]
        if to_grayscale:
            video_transforms += [transforms.Grayscale()]

        video_transforms = transforms.Compose(video_transforms)

        # NOTE: Avoid CPU out of memory
        Y = sequential_apply(Y, video_transforms, batch_size=256)

        Y = Y.contiguous().view(-1, segment_len, *Y.shape[-3:])

        Y = Y.to(torch.float32) / 255.0

        return Y


class Brain2FaceYLabECoGDataset(Brain2FaceCLIPDatasetBase):
    def __init__(self, args, train: bool = True) -> None:
        session_paths = glob.glob("data/preprocessed/ylab/" + args.preproc_name + "/*/")

        if args.face.type == "dynamic":
            y_reformer = self.ylab_reformer
        elif args.face.type == "static":
            y_reformer = partial(self.ylab_reformer, reduction=args.face.reduction)
        else:
            raise ValueError("Face type is only static or dynamic.")

        super().__init__(args, session_paths, train, y_reformer)

        assert not self.Y.requires_grad

    @staticmethod
    def ylab_reformer(Y: np.ndarray, reduction: Optional[bool] = None) -> torch.Tensor:
        """
        Args:
            Y: ( samples, features=709, segment_len=90 )
        """
        Y = torch.from_numpy(Y).to(torch.float32)

        cprint(Y.shape, "yellow")
        if reduction == "extract":
            Y = Y[:, :, Y.shape[-1] // 2]
        elif reduction == "mean":
            Y = Y.mean(dim=-1)
        else:
            assert reduction is None, "Reduction is either extract or mean."

        return Y


class Brain2FaceStyleGANDataset(Brain2FaceCLIPDatasetBase):
    def __init__(self, args, train: bool = True) -> None:
        session_paths = glob.glob(
            "data/preprocessed/stylegan/" + args.preproc_name + "/*/"
        )

        super().__init__(
            args, session_paths, train, y_reformer=self.reshape_stylegan_latent
        )

        assert not self.Y.requires_grad

    @staticmethod
    def reshape_stylegan_latent(Y: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            Y (torch.Tensor): ( samples, segment_len=90, styles=18, features=512 )
        Returns:
            Y (torch.Tensor): ( samples, features=512, segment_len*sub_styles=360 )
        """
        Y = Y.to(torch.float32)

        # Take four styles (TODO: take all styles. But will run out of memory)
        Y = Y[:, :, 4:8]

        # NOTE: squash latent layers to time dimension ( samples, segment_len*styles=360, features )
        Y = Y.contiguous().view(Y.shape[0], -1, Y.shape[-1])

        Y = Y.permute(0, 2, 1)  # ( samples, features=512, segment_len*styles=360 )

        return Y


class Brain2FaceCLIPEmbDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: str) -> None:
        super().__init__()

        self.Z = torch.load(f"data/clip_embds/{dataset.lower()}/brain_embds.pt")
        self.Y = torch.load(f"data/clip_embds/{dataset.lower()}/face_embds.pt")

        assert self.Z.shape == self.Y.shape

    def __len__(self):
        return len(self.Z)

    def __getitem__(self, i):
        return self.Z[i], self.Y[i]


class Brain2FaceCLIPEmbImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: str) -> None:
        super().__init__()

        self.Y = torch.load(f"data/clip_embds/{dataset.lower()}/face_embds.pt")
        self.Y_img = self._load_images(f"data/clip_embds/{dataset.lower()}/face_images")

        assert len(self.Y) == len(self.Y_img)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, i):
        return self.Y[i], self.Y_img[i]

    def _load_images(dir: str) -> torch.Tensor:
        images = []
        for path in natsorted(glob.glob(dir + "/*.jpg")):
            image = cv2.imread(path).astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)
            images.append(image)

        return torch.stack(images)


if __name__ == "__main__":
    from hydra import initialize, compose

    with initialize(version_base=None, config_path="../configs/"):
        args = compose(config_name="ylab_ecog.yaml")

    dataset = Brain2FaceYLabECoGDataset(args)
