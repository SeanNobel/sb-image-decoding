import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from typing import Union, Optional, Callable, Dict
from termcolor import cprint


class Models:
    def __init__(
        self,
        brain_encoder: nn.Module,
        face_encoder: Optional[nn.Module],
        loss_func: nn.Module,
    ):
        self.brain_encoder = brain_encoder
        self.face_encoder = face_encoder
        self.loss_func = loss_func

        self.brain_encoder_params = self._clone_params_list(self.brain_encoder)
        if self.face_encoder is not None:
            self.face_encoder_params = self._clone_params_list(self.face_encoder)

    def get_params(self):
        params = list(self.brain_encoder.parameters()) + list(self.loss_func.parameters())

        if self.face_encoder is not None:
            params += list(self.face_encoder.parameters())

        return params

    @staticmethod
    def _clone_params_list(model: nn.Module) -> Dict[str, torch.Tensor]:
        return {name: params.clone().cpu() for name, params in model.named_parameters()}

    @staticmethod
    def _get_non_updated_layers(
        new_params: Dict[str, torch.Tensor],
        prev_params: Dict[str, torch.Tensor],
    ) -> list:
        return [
            key for key in new_params.keys() if torch.equal(
                prev_params[key],
                new_params[key]
            )
        ]
        
    def params_updated(self) -> bool:
        updated = True

        new_params = self._clone_params_list(self.brain_encoder)
        non_updated_layers = self._get_non_updated_layers(
            new_params, self.brain_encoder_params
        )
        if len(non_updated_layers) > 0:
            cprint(
                f"Following layers in brain encoder are not updated: {non_updated_layers}", "red"
            )
            updated = False
        self.brain_encoder_params = new_params

        if self.face_encoder is not None:
            new_params = self._clone_params_list(self.face_encoder)
            non_updated_layers = self._get_non_updated_layers(
                new_params, self.face_encoder_params
            )
            if len(non_updated_layers) > 0:
                cprint(
                    f"Following layers in face encoder are not updated: {non_updated_layers}", "red"
                )
                updated = False
            self.face_encoder_params = new_params

        return updated

    def train(self) -> None:
        self.brain_encoder.train()
        if self.face_encoder is not None:
            self.face_encoder.train()
        self.loss_func.train()

    def eval(self) -> None:
        self.brain_encoder.eval()
        if self.face_encoder is not None:
            self.face_encoder.eval()
        self.loss_func.eval()

    def save(self, run_dir: str, best: bool = False) -> None:
        torch.save(
            self.brain_encoder.state_dict(),
            os.path.join(run_dir, f"brain_encoder_{'best' if best else 'last'}.pt"),
        )

        if self.face_encoder is not None:
            torch.save(
                self.face_encoder.state_dict(),
                os.path.join(run_dir, f"face_encoder_{'best' if best else 'last'}.pt"),
            )


def sequential_apply(
    X: Union[torch.Tensor, np.ndarray],
    # nn.Module is a hint for general DNNs. Callable is a hint for CLIP encoder
    model: Union[transforms.Compose, nn.Module, Callable],
    batch_size: int,
    device: Optional[str] = None,
    subject_idxs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Avoid CPU / CUDA out of memory.
    Args:
        X (torch.Tensor): _description_
        model (Union[transforms.Compose, FaceEncoder]): _description_
        batch_size (int): _description_
        subject_idxs (Optional[torch.Tensor], optional): _description_. Defaults to None.
    Returns:
        torch.Tensor: _description_
    """
    # NOTE: This is for torchvision transforms, which doesn't accept a batch of samples.
    # A bit of messy implementation.
    if batch_size == 1:
        assert isinstance(model, transforms.Compose) and isinstance(X, np.ndarray)

        # NOTE: np.split needs number of subarrays, while torch.split needs the size of chunks.
        return torch.cat(
            [
                model(Image.fromarray(_X.squeeze())).unsqueeze(0)
                for _X in np.split(X, X.shape[0])
            ]
        )

    orig_device = X.device

    if device is None:
        device = orig_device

    if subject_idxs is None:
        return torch.cat(
            [model(_X.to(device)).to(orig_device) for _X in torch.split(X, batch_size)]
        )
    else:
        return torch.cat(
            [
                model(_X.to(device), _subject_idxs.to(device)).to(orig_device)
                for _X, _subject_idxs in zip(
                    torch.split(X, batch_size),
                    torch.split(subject_idxs, batch_size),
                )
            ]
        )
