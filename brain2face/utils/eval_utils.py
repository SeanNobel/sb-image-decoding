import os, sys
import numpy as np
import torch
import cv2
from PIL import Image
import h5py
from omegaconf import DictConfig, OmegaConf
from termcolor import cprint
from typing import Optional

from brain2face.utils.constants import EMB_CHUNK_SIZE


class VisionSaver:
    def __init__(
        self,
        args,
        save_dir: str,
        channels: int = 3,  # FIXME: hard-coded
        for_webdataset: bool = False,
    ) -> None:
        self.to_tensored = not args.vision.pretrained
        self.is_video = not args.reduce_time
        self.as_h5 = args.as_h5
        self.fps = args.fps
        frames = args.fps * args.seq_len
        size = args.vision_encoder.image_size

        self.sample_idx = 0
        self.chunk_idx = 0

        if for_webdataset:
            self.save_dir_prefix = os.path.join(
                save_dir, "for_webdataset", "images" if not self.is_video else "videos"
            )
            self.save_dir = self._update_save_dir(self.chunk_idx)

            self.save = self._save_for_webdataset

        else:
            if self.as_h5:
                self.save_dir = save_dir
            else:
                self.save_dir = os.path.join(
                    save_dir, "images" if not self.is_video else "videos"
                )

            os.makedirs(self.save_dir, exist_ok=True)

            if self.as_h5:
                # NOTE: h5 file contains many videos as a single file.
                # with h5py.File(os.path.join(save_dir, "videos.h5"), "a") as hdf:
                self.hdf = h5py.File(os.path.join(save_dir, "videos.h5"), "w")
                self.dataset = self.hdf.require_dataset(
                    name="videos",
                    shape=(0, frames, size, size, channels),
                    maxshape=(None, frames, size, size, channels),
                    dtype=np.float32,
                )

            self.save = self._save

    def _save(self, Y: torch.Tensor) -> None:
        if self.as_h5:
            self._save_video_as_h5(Y)

        else:
            for y in Y:
                if not self.is_video:
                    self._save_image(y)
                else:
                    self._save_video(y)

                self.sample_idx += 1

    def _save_image(self, y: torch.Tensor) -> None:
        save_path = os.path.join(self.save_dir, str(self.sample_idx).zfill(5) + ".jpg")

        image = y.numpy()

        if self.to_tensored:
            image = (image * 255).astype(np.uint8).transpose(1, 2, 0)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        cv2.imwrite(save_path, image)

    def _save_video_as_h5(self, y: torch.Tensor) -> None:
        """_summary_
        Args:
            y ( b, frames, channels, size, size ): _description_
        """
        video = y.numpy().transpose(0, 1, 3, 4, 2)
        # ( b, frames, size, size, channels )

        self.dataset.resize(self.dataset.shape[0] + video.shape[0], axis=0)
        self.dataset[-video.shape[0] :] = video

    def _save_video(self, y: torch.Tensor) -> None:
        """_summary_
        Args:
            y ( frames, channels, size, size ): _description_
            fps (float, optional): _description_. Defaults to 30.0.
        """
        video = y.numpy()  # ( frames, channels, size, size )
        video = (video * 255).astype(np.uint8).transpose(0, 2, 3, 1)
        # ( frames, size, size, channels )

        save_path = os.path.join(self.save_dir, str(self.sample_idx).zfill(5) + ".mp4")

        fmt = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fmt, self.fps, tuple(video.shape[1:3]))

        for frame in video:
            writer.write(frame)

        writer.release()

    def _save_for_webdataset(self, Y: torch.Tensor) -> None:
        """Saves batch of images to save_dir. (00000.jpg, 00001.jpg, ...)
        Continues from the last index in the last batch.
        Args:
            Y: ( batch_size, channels=3, size=256, size=256 )
        """
        for y in Y:
            save_path = os.path.join(
                self.save_dir,
                str(self.chunk_idx).zfill(4)
                + str(self.sample_idx).zfill(len(str(EMB_CHUNK_SIZE)) - 1)
                + ".mp4"
                if self.is_video
                else ".jpg",
            )

            if not self.is_video:
                self._save_image(y, save_path)
            else:
                self._save_video(y, save_path)

            self.sample_idx += 1

            if self.sample_idx == EMB_CHUNK_SIZE:
                self.sample_idx = 0
                self.chunk_idx += 1

                self.save_dir = self._update_save_dir(self.chunk_idx)

    def _update_save_dir(self, chunk_idx: int) -> str:
        """Updates self.save_dir, creates it, and returns it."""
        save_dir = os.path.join(self.save_dir_prefix, str(chunk_idx).zfill(4))
        os.makedirs(save_dir, exist_ok=True)

        return save_dir

    def close(self) -> None:
        if self.as_h5:
            self.hdf.close()


class EmbeddingSaver:
    def __init__(self, save_dir: str, for_webdataset: bool = False) -> None:
        if for_webdataset:
            self.save_dir = os.path.join(save_dir, "for_webdataset")
            self.save = self._save_for_webdataset
        else:
            self.save_dir = save_dir
            os.makedirs(self.save_dir, exist_ok=True)
            self.save = self._save

    def _save(self, brain: torch.Tensor, vision: torch.Tensor) -> None:
        """
        Args:
            brain: ( samples, emb_dim=512 )
            image: ( samples, emb_dim=512 )
        """
        assert brain.shape == vision.shape

        torch.save(brain, os.path.join(self.save_dir, "brain_embds.pt"))
        torch.save(vision, os.path.join(self.save_dir, "vision_embds.pt"))

    def _save_for_webdataset(self, brain: torch.Tensor, vision: torch.Tensor) -> None:
        """
        Args:
            brain: ( samples~=13000, emb_dim=512 )
            image: ( samples~=13000, emb_dim=512 )
        """
        assert brain.shape == vision.shape

        brain_save_dir = os.path.join(self.save_dir, "brain")
        os.makedirs(brain_save_dir, exist_ok=True)

        vision_save_dir = os.path.join(self.save_dir, "vision")
        os.makedirs(vision_save_dir, exist_ok=True)

        brain = torch.split(brain, EMB_CHUNK_SIZE)
        vision = torch.split(vision, EMB_CHUNK_SIZE)

        for i, (b, f) in enumerate(zip(brain, vision)):
            np.save(
                os.path.join(brain_save_dir, f"brain_embds_{str(i).zfill(4)}.npy"),
                b.numpy(),
            )
            np.save(
                os.path.join(vision_save_dir, f"vision_embds_{str(i).zfill(4)}.npy"),
                f.numpy(),
            )


def update_with_eval(args: DictConfig) -> DictConfig:
    """Keeps args.eval intact and updates args with args.eval.
    Args:
        args: Requires 'eval' key.
    Returns:
        args: _description_
    """
    args = OmegaConf.to_container(args, resolve=True)

    args_eval = args.pop("eval")

    args = recursive_update(args, args_eval)

    args.update({"eval": args_eval})

    args = OmegaConf.create(args)

    return args


def get_run_dir(args: DictConfig) -> str:
    """_summary_
    Args:
        args: Requires 'eval' key.
    Returns:
        run_dir: _description_
    """
    run_name = "".join(
        [k + "-" + str(v) + "_" for k, v in sorted(collapse_nest(args.eval).items())]
    )
    run_dir = os.path.join("runs", args.dataset.lower(), args.train_name, run_name)
    assert os.path.exists(run_dir), "run_dir doesn't exist."

    return run_dir


def recursive_update(dict_base: dict, other: dict) -> dict:
    """Updates a dict with other dict recursively.
    Args:
        dict_base: _description_
        other: _description_
    Returns:
        dict_base: Updated new dict.
    """
    for k, v in other.items():
        if isinstance(v, dict) and k in dict_base:
            recursive_update(dict_base[k], v)
        else:
            dict_base[k] = v

    return dict_base


def collapse_nest(args: DictConfig) -> dict:
    """e.g.) {"a": {"b": 1}} -> {"a.b": 1}
    NOTE: This function only works for 2-level nested dict.
    Args:
        args: _description_
    Returns:
        args: _description_
    """
    args = OmegaConf.to_container(args)

    for k, v in args.copy().items():
        if isinstance(v, dict):
            for k_, v_ in v.items():
                assert not isinstance(
                    v_, dict
                ), "collapse_nest() only works for 2-level nested dict."

                args.update({f"{k}.{k_}": v_})

            del args[k]

    return args
