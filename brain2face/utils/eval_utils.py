import os
import numpy as np
import torch
import cv2
from PIL import Image
from omegaconf import DictConfig, OmegaConf
from termcolor import cprint

from brain2face.utils.constants import EMB_CHUNK_SIZE


class ImageSaver:
    def __init__(self, save_dir: str, for_webdataset: bool, to_tensored: bool) -> None:
        self.sample_idx = 0
        self.chunk_idx = 0

        if for_webdataset:
            self.save_dir_prefix = os.path.join(save_dir, "for_webdataset", "images")
            self.save_dir = self._update_save_dir(self.chunk_idx)

            self.save = self._save_for_webdataset

        else:
            self.save_dir = os.path.join(save_dir, "images")
            os.makedirs(self.save_dir, exist_ok=True)

            self.save = self._save
            
        self.to_tensored = to_tensored

    def _save(self, Y: torch.Tensor) -> None:
        for y in Y:
            save_path = os.path.join(
                self.save_dir, str(self.sample_idx).zfill(5) + ".jpg"
            )

            self._save_image(y, save_path)

            self.sample_idx += 1

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
                + ".jpg",
            )

            self._save_image(y, save_path)

            self.sample_idx += 1

            if self.sample_idx == EMB_CHUNK_SIZE:
                self.sample_idx = 0
                self.chunk_idx += 1

                self.save_dir = self._update_save_dir(self.chunk_idx)

    def _save_image(self, y: torch.Tensor, save_path: str) -> None:
        image = y.numpy()
        
        if self.to_tensored:
            image = (image * 255).astype(np.uint8).transpose(1, 2, 0)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        cv2.imwrite(save_path, image)

    def _update_save_dir(self, chunk_idx: int) -> str:
        """Updates self.save_dir, creates it, and returns it."""
        save_dir = os.path.join(self.save_dir_prefix, str(chunk_idx).zfill(4))
        os.makedirs(save_dir, exist_ok=True)

        return save_dir


class EmbeddingSaver:
    def __init__(self, save_dir: str, for_webdataset: bool) -> None:
        if for_webdataset:
            self.save_dir = os.path.join(save_dir, "for_webdataset")
            self.save = self._save_for_webdataset
        else:
            self.save_dir = save_dir
            self.save = self._save

    def _save(self, brain: torch.Tensor, image: torch.Tensor) -> None:
        """
        Args:
            brain: ( samples, emb_dim=512 )
            image: ( samples, emb_dim=512 )
        """
        assert brain.shape == image.shape

        torch.save(brain, os.path.join(self.save_dir, "brain_embds.pt"))
        torch.save(image, os.path.join(self.save_dir, "image_embds.pt"))

    def _save_for_webdataset(self, brain: torch.Tensor, face: torch.Tensor) -> None:
        """
        Args:
            brain: ( samples~=13000, emb_dim=512 )
            face: ( samples~=13000, emb_dim=512 )
        """
        assert brain.shape == face.shape
        brain_save_dir = os.path.join(self.save_dir, "brain")
        os.makedirs(brain_save_dir, exist_ok=True)

        face_save_dir = os.path.join(self.save_dir, "image")
        os.makedirs(face_save_dir, exist_ok=True)

        brain = torch.split(brain, EMB_CHUNK_SIZE)
        face = torch.split(face, EMB_CHUNK_SIZE)

        for i, (b, f) in enumerate(zip(brain, face)):
            np.save(
                os.path.join(brain_save_dir, f"brain_embds_{str(i).zfill(4)}.npy"),
                b.numpy(),
            )
            np.save(
                os.path.join(face_save_dir, f"image_embds_{str(i).zfill(4)}.npy"),
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
    run_dir = os.path.join("runs", args.dataset.lower(), run_name)
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
