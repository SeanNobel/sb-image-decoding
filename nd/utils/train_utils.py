import os, sys
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from typing import Union, Optional, Callable, Dict
from termcolor import cprint


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Models:
    def __init__(
        self,
        brain_encoder: nn.Module,
        brain_decoder: Optional[nn.Module] = None,
        vision_encoder: Optional[nn.Module] = None,
        loss_func: Optional[nn.Module] = None,
    ):
        self.brain_encoder = brain_encoder
        self.brain_decoder = brain_decoder
        self.vision_encoder = vision_encoder
        self.loss_func = loss_func

        self.brain_encoder_params = self._clone_params_list(self.brain_encoder)
        self.brain_decoder_params, self.vision_encoder_params = None, None
        if self.brain_decoder is not None:
            self.brain_decoder_params = self._clone_params_list(self.brain_decoder)
        if self.vision_encoder is not None:
            self.vision_encoder_params = self._clone_params_list(self.vision_encoder)

    def get_params(self):
        params = list(self.brain_encoder.parameters()) + list(self.loss_func.parameters())  # fmt: skip

        if self.brain_decoder is not None:
            params += list(self.brain_decoder.parameters())
        if self.vision_encoder is not None:
            params += list(self.vision_encoder.parameters())

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
            key
            for key in new_params.keys()
            if torch.equal(prev_params[key], new_params[key]) and new_params[key].requires_grad  # fmt: skip
        ]

    def params_updated(self, show_non_updated: bool = True) -> bool:
        updated = True

        models_list = [self.brain_encoder, self.brain_decoder, self.vision_encoder]
        params_list = [self.brain_encoder_params, self.brain_decoder_params, self.vision_encoder_params]  # fmt: skip
        names_list = ["brain encoder", "brain decoder", "vision encoder"]

        for model, params, name in zip(models_list, params_list, names_list):
            if model is None:
                continue

            new_params = self._clone_params_list(model)
            non_updated_layers = self._get_non_updated_layers(new_params, params)
            if len(non_updated_layers) > 0:
                if show_non_updated:
                    cprint(f"Following layers in {name} are not updated: {non_updated_layers}","red")  # fmt: skip

                updated = False

            if model is self.brain_encoder:
                self.brain_encoder_params = new_params
            elif model is self.brain_decoder:
                self.brain_decoder_params = new_params
            elif model is self.vision_encoder:
                self.vision_encoder_params = new_params

        return updated

    def train(self) -> None:
        self.brain_encoder.train()
        if self.brain_decoder is not None:
            self.brain_decoder.train()
        if self.vision_encoder is not None:
            self.vision_encoder.train()
        self.loss_func.train()

    def eval(self) -> None:
        self.brain_encoder.eval()
        if self.brain_decoder is not None:
            self.brain_decoder.eval()
        if self.vision_encoder is not None:
            self.vision_encoder.eval()
        self.loss_func.eval()

    def save(self, run_dir: str, best: bool = False) -> None:
        torch.save(self.brain_encoder.state_dict(), os.path.join(run_dir, f"brain_encoder_{'best' if best else 'last'}.pt"))  # fmt: skip
        if self.brain_decoder is not None:
            torch.save(self.brain_decoder.state_dict(), os.path.join(run_dir, f"brain_decoder_{'best' if best else 'last'}.pt"))  # fmt: skip
        if self.vision_encoder is not None:
            torch.save(self.vision_encoder.state_dict(), os.path.join(run_dir, f"vision_encoder_{'best' if best else 'last'}.pt"))  # fmt: skip


def sequential_apply(
    X: Union[torch.Tensor, np.ndarray],
    # NOTE: nn.Module is a hint for general DNNs. Callable is a hint for CLIP encoder
    fn: Union[transforms.Compose, nn.Module, Callable],
    batch_size: int,
    device: Optional[str] = None,
    subject_idxs: Optional[torch.Tensor] = None,
    desc: str = "",
    reduction: str = "mean",
) -> torch.Tensor:
    """Avoid CPU / CUDA out of memory.
    Args:
        X (torch.Tensor): _description_
        model (Union[transforms.Compose, VisionEncoder]): _description_
        batch_size (int): _description_
        subject_idxs (Optional[torch.Tensor], optional): _description_. Defaults to None.
    Returns:
        torch.Tensor: _description_
    """
    # NOTE: This is for torchvision transforms, which doesn't accept a batch of samples.
    # A bit of messy implementation.
    if isinstance(fn, transforms.Compose) and isinstance(X, np.ndarray):
        # NOTE: np.split needs number of subarrays, while torch.split needs the size of chunks.
        return torch.cat(
            [
                fn(Image.fromarray(_X.squeeze())).unsqueeze(0)
                for _X in np.split(X, X.shape[0])
            ]
        )

    orig_device = X.device

    if device is None:
        device = orig_device

    # NOTE: sequential_apply doesn't do sequential application if batch_size == X.shape[0].
    if batch_size == X.shape[0]:
        # assert isinstance(X, torch.Tensor) and isinstance(model, nn.Module)
        if subject_idxs is None:
            return fn(X.to(device)).to(orig_device)
        else:
            return fn(X.to(device), subject_idxs.to(device)).to(orig_device)

    if subject_idxs is None:
        output = [
            fn(_X.to(device)) for _X in tqdm(torch.split(X, batch_size), desc=desc)
        ]
    else:
        output = [
            fn(_X.to(device), _subject_idxs.to(device))
            for _X, _subject_idxs in tqdm(
                zip(
                    torch.split(X, batch_size),
                    torch.split(subject_idxs, batch_size),
                ),
                desc=desc,
            )
        ]

    if isinstance(output[0], torch.Tensor):
        return torch.cat(output).to(orig_device)

    elif isinstance(output[0], dict):
        stacked_dict = {}

        for key in output[0].keys():
            _output = [_dict[key] for _dict in output]

            if isinstance(_output[0], torch.Tensor):
                if _output[0].ndim == 0:
                    _output = torch.stack(_output)

                    if reduction == "mean":
                        _output = _output.mean()
                    elif reduction == "sum":
                        _output = _output.sum()
                else:
                    _output = torch.cat(_output)

            stacked_dict.update({key: _output})

        return stacked_dict
    else:
        raise ValueError(f"Unknown output type: {type(output[0])}")


def conv_output_size(
    input_size: int,
    ksize: int,
    stride: int,
    repetition: int,
    downsample: int,
) -> int:
    for _ in range(downsample):
        input_size = (input_size - 1) // 2 + 1

    for _ in range(repetition):
        input_size = (input_size - ksize) // stride + 1

    return input_size
