import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import numpy as np
import cv2
from PIL import Image
import glob
from tqdm import tqdm
from natsort import natsorted
import mne
import mediapipe as mp
import h5py
from functools import partial
from typing import Optional, Union, List, Tuple, Callable
from termcolor import cprint
from omegaconf import DictConfig

import clip
from clip.model import CLIP

from brain2face.utils.brain_preproc import segment_then_blcorr
from brain2face.utils.train_utils import sequential_apply
from brain2face.utils.preproc_utils import crop_and_segment, crop_longer, sequential_load

mne.set_log_level(verbose="WARNING")
mp_face_mesh = mp.solutions.face_mesh


class NeuroDiffusionCLIPDatasetBase(torch.utils.data.Dataset):
    def __init__(
        self,
        args,
        session_paths: List[str],
        train: bool,
        loader: Callable,
    ) -> List[torch.Tensor]:
        super().__init__()

        # NOTE: No need to be natsorted.
        # NOTE: Selecting directories with preprocessed data.
        # session_paths = self._drop_bads(session_paths)

        if args.debug:
            session_paths = session_paths[:1]

        if args.split in ["subject_random", "subject_each"]:
            # NOTE: subject_names must be 'S*_{name}' format in this case.
            session_paths, self.num_subjects, self.subject_names = self._split_sessions(
                train
            )
        else:
            self.num_subjects = len(session_paths)
            self.subject_names = [path.split("/")[-2] for path in session_paths]

        cprint(f"Num subjects: {self.num_subjects}", color="cyan")

        X_list = []
        Y_list = []
        subject_idx_list = []
        self.Y_ref = []  # NOTE: for video data that occupies a large amount of memory
        for subject_idx, subject_path in enumerate(session_paths):
            X, Y = loader(subject_path=subject_path)

            if isinstance(Y, h5py._hl.dataset.Dataset):
                self.Y_ref.append(Y)
                Y = torch.arange(len(Y), dtype=torch.int64)

            if not args.segment_in_preproc:
                assert not isinstance(Y, h5py._hl.dataset.Dataset), "Y must be segmented in preproc when it's video."  # fmt: skip
                X, Y = crop_longer(X, Y)

            assert X.shape[0] == Y.shape[0]

            if args.split == "deep":
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
                subject_idx = np.where(np.array(self.subject_names) == name)[0][0]

            subject_idx *= torch.ones(X.shape[0], dtype=torch.uint8)
            cprint(
                f"X: {X.shape} | Y: {Y.shape} | subject_idx: {subject_idx.shape}", "cyan"
            )

            X_list.append(X)
            Y_list.append(Y)
            subject_idx_list.append(subject_idx)

            del X, Y, subject_idx

        self.subject_idx = torch.cat(subject_idx_list)
        del subject_idx_list
        cprint(f"self.subject_idx: {self.subject_idx.shape}", color="cyan")

        # FIXME: _trim_channels() is only for debug
        if args.dataset == "YLabGOD" and args.layout == "ch_locations_2d":
            X_list = self._trim_channels(X_list)
        else:
            # NOTE: for dataset where subjects have different channel numbers
            X_list = self._pad_channels(X_list)

        self.X = torch.cat(X_list)
        del X_list
        cprint(
            f"self.X: {self.X.shape} (channels are zero-padded when channel numbers depend on subjects)",
            color="cyan",
        )

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
    def _pad_channels(X_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """Pads channels to the same length with zeros, if they are different"""
        num_channels = np.array([X.shape[1] for X in X_list])

        if not np.all(num_channels == num_channels[0]):
            X_list = [
                F.pad(X, (0, 0, 0, np.max(num_channels) - X.shape[1]), "constant", 0)
                for X in X_list
            ]

        return X_list

    @staticmethod
    def _trim_channels(X_list: List[torch.Tensor]) -> List[torch.Tensor]:
        num_channels = np.array([X.shape[1] for X in X_list])

        if not np.all(num_channels == num_channels[0]):
            X_list = [X[:, : np.min(num_channels)] for X in X_list]

        return X_list

    @staticmethod
    def _trim_channels(X_list: List[torch.Tensor]) -> List[torch.Tensor]:
        num_channels = np.array([X.shape[1] for X in X_list])

        if not np.all(num_channels == num_channels[0]):
            X_list = [X[:, : np.min(num_channels)] for X in X_list]

        return X_list

    @staticmethod
    def _drop_bads(_subject_paths):
        subject_paths = []
        for path in _subject_paths:
            if os.path.exists(path + "brain.npy") and os.path.exists(path + "vision.npy"):
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


class YLabGODCLIPDataset(NeuroDiffusionCLIPDatasetBase):
    def __init__(self, args, train: bool = True) -> None:
        # NOTE: it is important to sort here to match montage paths in layout.py
        subject_paths = natsorted(
            glob.glob(f"data/preprocessed/ylab/god/{args.preproc_name}/*/")
        )

        loader = partial(self._loader, args=args, train=train)

        super().__init__(args, subject_paths, train, loader)

    @staticmethod
    def _loader(
        args,
        subject_path: str,
        train: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            X: ( samples, channels, timesteps )
            Y: ( samples, 3, image_size, image_size )
        """
        cprint(f"Preprocessing subject {subject_path.split('/')[-2]}", "cyan")

        image_fnames = np.loadtxt(
            os.path.join(subject_path, f"image_{'train' if train else 'test'}.txt"),
            delimiter=",",
            dtype=str,
        )

        images_dir = os.path.join(args.data_root, f"images_{'trn' if train else 'val'}")
        Y = []
        dropped_idxs = []
        for i, fname in tqdm(enumerate(image_fnames), desc="Loading images"):
            try:
                image = Image.open(os.path.join(images_dir, fname)).resize(
                    (args.image_size, args.image_size)
                )
                # NOTE: Some images seem to be grayscale. Convert them to RGB.
                if image.getbands()[0] == "L":
                    image = image.convert("RGB")

                Y.append(image)
            except:
                dropped_idxs.append(i)

        Y = torch.from_numpy(np.stack(Y))

        if not args.vision.pretrained:
            Y = Y.permute(0, 3, 1, 2).to(torch.float32) / 255.0

        X = np.load(
            os.path.join(subject_path, f"brain_{'train' if train else 'test'}.npy")
        )
        X = torch.from_numpy(np.delete(X, dropped_idxs, axis=0).astype(np.float32))
        X = X[:, :, int(args.seq_onset * args.brain_resample_sfreq) : int((args.seq_onset + args.seq_len) * args.brain_resample_sfreq)]  # fmt: skip

        return X, Y


class YLabE0030CLIPDataset(NeuroDiffusionCLIPDatasetBase):
    def __init__(self, args, train: bool = True) -> None:
        is_segmented = "segmented" if args.segment_in_preproc else "unsegmented"
        session_paths = glob.glob(f"data/preprocessed/ylab/{is_segmented}/{args.preproc_name}/*/")  # fmt: skip

        loader = partial(
            self._loader,
            args=args,
            y_reformer=partial(
                self._y_reformer,
                segmented=args.segment_in_preproc,
                segment_len=int(args.fps * args.seq_len),
                reduce_time=args.reduce_time,
                reduction=args.vision.reduction,
            ),
        )

        super().__init__(args, session_paths, train, loader)

        assert not self.Y.requires_grad

    @staticmethod
    def _loader(
        args,
        subject_path: str,
        y_reformer: Callable,
    ):
        X = np.load(os.path.join(subject_path, "brain.npy"))
        if not args.segment_in_preproc:
            X = segment_then_blcorr(
                args, X, segment_len=int(args.brain_resample_sfreq * args.seq_len)
            )
        X = torch.from_numpy(X).to(torch.float32)

        Y = h5py.File(os.path.join(subject_path, "face.h5"), "r")["data"]
        Y = sequential_load(data=Y, bufsize=256, preproc_func=y_reformer)

        return X, Y

    @staticmethod
    def _y_reformer(
        Y: np.ndarray,
        reduce_time: bool,
        segmented: bool,
        segment_len: Optional[int] = None,
        reduction: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Args:
            Y: ( samples, features=17, segment_len=90 ) or ( features=17, timesteps )
        """
        if not segmented:
            Y = crop_and_segment(Y.T, segment_len).transpose(0, 2, 1)

        Y = torch.from_numpy(Y).to(torch.float32)

        # if reduce_time:
        #     if reduction == "extract":
        #         Y = Y[:, :, Y.shape[-1] // 2]
        #     elif reduction == "mean":
        #         Y = Y.mean(dim=-1)
        #     else:
        #         assert reduction is None, "Reduction is either extract or mean."

        return Y


class UHDCLIPDataset(NeuroDiffusionCLIPDatasetBase):
    def __init__(self, args, train: bool = True) -> None:
        session_paths = glob.glob(f"data/preprocessed/uhd/{args.preproc_name}/*/")

        if not args.reduce_time:
            # y_reformer = partial(
            #     self.transform_video, image_size=args.vision_encoder.image_size
            # )
            loader = partial(self._loader, y_reformer=None)

        else:
            loader = partial(
                self._loader,
                y_reformer=partial(
                    self.to_single_frame,
                    reduction=args.vision.reduction,
                    vision_pretrained=args.vision.pretrained,
                ),
            )

        super().__init__(args, session_paths, train, loader)

        # FIXME: I've forgot to take first 128 channels from 139 while preprocessing
        self.X = self.X[:, : args.num_channels]

    @staticmethod
    def _loader(
        subject_path: str, y_reformer: Optional[Callable]
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, h5py._hl.dataset.Dataset]]:
        X = np.load(os.path.join(subject_path, "brain.npy"))
        X = torch.from_numpy(X.astype(np.float32))

        Y = h5py.File(os.path.join(subject_path, "face.h5"), "r")["data"]

        if y_reformer is not None:
            Y = sequential_load(data=Y, bufsize=256, preproc_func=y_reformer)

        return X, Y

    @staticmethod
    def to_single_frame(
        Y: np.ndarray,
        reduction: str = "extract",
        vision_pretrained: bool = False,
    ) -> torch.Tensor:
        """Extracts single frame from video sequence, then encodes to pre-trained CLIP space.
        NOTE: This function is called from sequential_load(), so there's no need to split Y into batch.
        Args:
            Y: ( samples, segment_len=90, face_extractor.output_size=256, face_extractor.output_size=256, 3 )
            clip_model: Pretrained image encoder of Radford 2021.
            preprocess: Transforms for the pretrained image encoder.
        Returns:
            Y: ( samples, 3, face_extractor.output_size=256, face_extractor.output_size=256 )
                or ( samples, F )
        """
        if reduction == "extract":
            # NOTE: Take the frame in the middle
            Y = Y[:, Y.shape[1] // 2]
        elif reduction == "mean":
            Y = Y.mean(axis=1)
        else:
            raise ValueError("Reduction is either extract or mean.")
        # ( samples, face_extractor.output_size=256, face_extractor.output_size=256, 3)

        Y = torch.from_numpy(Y)

        # if clip_model is not None:
        #     assert preprocess is not None, "clip_model needs preprocess."

        #     Y = sequential_apply(Y, preprocess, batch_size=1).to(device)

        #     with torch.no_grad():
        #         # NOTE: CLIP model somehow returns fp16 (https://colab.research.google.com/github/openai/clip/blob/master/notebooks/Interacting_with_CLIP.ipynb#scrollTo=cuxm2Gt4Wvzt)
        #         Y = clip_model.encode_image(Y).cpu().float()

        # else:
        if not vision_pretrained:
            Y = Y.permute(0, 3, 1, 2).to(torch.float32) / 255.0

        return Y


class CollateFunctionForVideoHDF5(nn.Module):
    def __init__(self, Y_ref: List[h5py._hl.dataset.Dataset], frame_size: int) -> None:
        super().__init__()
        self.Y_ref = Y_ref
        self.frame_size = frame_size

    def forward(self, batch: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            batch: ( samples, segment_len=90, features=17 ) or ( samples, features=17, segment_len=90 )
        Returns:
            batch: ( samples, features=17, segment_len=90 )
        """
        X = torch.stack([item[0] for item in batch])
        # NOTE: item[2] is subject_idx and item[1] is sample_idx
        Y = np.stack([self.Y_ref[item[2]][item[1]] for item in batch])
        subject_idxs = torch.stack([item[2] for item in batch])

        Y = self._video_transforms(Y, self.frame_size)

        return X, Y, subject_idxs

    @staticmethod
    def _video_transforms(
        Y: np.ndarray, frame_size: int, to_grayscale: bool = False
    ) -> torch.Tensor:
        """
        - Resizes the video frames if args.face_extractor.output_size != args.vivit.image_size
        - Reduces the video to grayscale if specified
        - Scale [0 - 255] -> [0. - 1.]
        Args:
            Y: ( batch_size, segment_len=90, face_extractor.output_size=256, face_extractor.output_size=256, 3 )
            image_size: args.vivit.image_size
        Returns:
            Y: ( batch_size, segment_len=90, 3, vision_encoder.image_size, vision_encoder.image_size )
        """
        segment_len = Y.shape[1]

        Y = torch.from_numpy(Y).view(-1, *Y.shape[-3:]).permute(0, 3, 1, 2)
        # ( samples*segment_len, 3, size, size )

        video_transforms = [
            transforms.Resize(
                frame_size, interpolation=InterpolationMode.BICUBIC, antialias=True
            ),
        ]
        if to_grayscale:
            video_transforms += [transforms.Grayscale()]

        video_transforms = transforms.Compose(video_transforms)

        # NOTE: Avoid CPU out of memory
        # Y = sequential_apply(Y, video_transforms, batch_size=256)
        Y = video_transforms(Y)

        Y = Y.contiguous().view(-1, segment_len, *Y.shape[-3:])

        Y = Y.to(torch.float32) / 255.0

        return Y


class StyleGANCLIPDataset(NeuroDiffusionCLIPDatasetBase):
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


class NeuroDiffusionCLIPEmbDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: str, train: bool = True) -> None:
        super().__init__()

        self.Z = torch.load(
            f"data/clip_embds/{dataset.lower()}/{'train' if train else 'test'}/brain_embds.pt"
        )
        self.Y = torch.load(
            f"data/clip_embds/{dataset.lower()}/{'train' if train else 'test'}/image_embds.pt"
        )

        assert self.Z.shape == self.Y.shape

    def __len__(self):
        return len(self.Z)

    def __getitem__(self, i):
        return self.Z[i], self.Y[i]


class NeuroDiffusionCLIPEmbImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: str, train: bool = True) -> None:
        super().__init__()

        self.Y = torch.load(
            f"data/clip_embds/{dataset.lower()}/{'train' if train else 'test'}/image_embds.pt"
        )
        self.Y_img = self._load_images(
            f"data/clip_embds/{dataset.lower()}/{'train' if train else 'test'}/images"
        )

        assert len(self.Y) == len(self.Y_img)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, i):
        return self.Y[i], self.Y_img[i]

    @staticmethod
    def _load_images(dir: str) -> torch.Tensor:
        images = []
        for path in tqdm(natsorted(glob.glob(dir + "/*.jpg")), desc="Loading images"):
            image = cv2.imread(path).astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)
            images.append(image)

        return torch.stack(images)


class NeuroDiffusionCLIPEmbVideoDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: str, train: bool = True) -> None:
        super().__init__()

        self.Y = torch.load(
            f"data/clip_embds/{dataset.lower()}/{'train' if train else 'test'}/video_embds.pt"
        )
        self.Y_video = self._load_images(
            f"data/clip_embds/{dataset.lower()}/{'train' if train else 'test'}/video"
        )

        assert len(self.Y) == len(self.Y_video)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, i):
        return self.Y[i], self.Y_img[i]

    @staticmethod
    def _load_videos(dir: str) -> torch.Tensor:
        # TODO: implement
        images = []
        for path in tqdm(natsorted(glob.glob(dir + "/*.jpg")), desc="Loading images"):
            image = cv2.imread(path).astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)
            images.append(image)

        return torch.stack(images)


class UHDPipelineDataset(UHDCLIPDataset):
    def __init__(self, args, session_id: int, train: bool = True) -> None:
        super().__init__(args, train)

        sample_idxs = torch.where(self.subject_idx == session_id)[0]
        self.X = self.X[sample_idxs]
        self.Y = self.Y[sample_idxs]
        self.subject_idx = self.subject_idx[sample_idxs]

        self.X = self.to_sequence(args, self.X)
        self.subject_idx = torch.full((len(self.X),), session_id, dtype=torch.uint8)

    def __getitem__(self, i):
        return self.X[i], self.subject_idx[i]

    @staticmethod
    def to_sequence(args, X: torch.Tensor) -> torch.Tensor:
        X = X.permute(0, 2, 1)
        X = X.contiguous().view(-1, X.shape[-1])

        X = X[:1200]  # 10 seconds

        segment_len = args.brain_resample_sfreq * args.seq_len
        X = torch.stack(
            [
                X[i : i + segment_len]
                for i in range(
                    0, X.shape[0] - segment_len, args.brain_resample_sfreq // args.fps
                )
            ]
        ).permute(0, 2, 1)

        cprint(X.shape, "cyan")

        return X


if __name__ == "__main__":
    from hydra import initialize, compose

    with initialize(version_base=None, config_path="../configs/ylab"):
        args = compose(config_name="god.yaml")

    dataset = YLabGODCLIPDataset(args)
