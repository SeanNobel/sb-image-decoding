import torch
import torch.nn as nn
from typing import Tuple


class GradientReversalFunction(torch.autograd.Function):
    """https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    Args:
        torch (_type_): _description_
    """

    @staticmethod
    def forward(ctx, X: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(scale)

        return X

    @staticmethod
    def backward(ctx, grad_backward: torch.Tensor) -> Tuple[torch.Tensor]:
        (scale,) = ctx.saved_tensors

        return scale * -grad_backward, None


class GradientReversalLayer(nn.Module):
    def __init__(self, scale: float):
        super().__init__()

        self.scale = torch.tensor(scale)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(X, self.scale)


class DANN(nn.Module):
    """DANN model
    Args:
        nn (_type_): _description_
    """

    def __init__(self, in_dim: int, num_domains: int, scale: float):
        super().__init__()

        self.grad_rev = GradientReversalLayer(scale)

        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, in_dim // 4),
            nn.LayerNorm(in_dim // 4),
            nn.GELU(),
            nn.Linear(in_dim // 4, num_domains),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.grad_rev(X)

        return self.head(X)
