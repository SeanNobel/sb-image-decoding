import torch
import torch.nn as nn
from torchvision import transforms
from typing import Union, Optional, Callable


def sequential_apply(
    X: torch.Tensor,
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
