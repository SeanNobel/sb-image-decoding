import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def torch_exp(x: torch.Tensor):  # x: ( N, )
    return torch.exp(x.clamp(max=10))


def torch_log(x: torch.Tensor):
    return torch.log(x.clamp(min=1e-10))


class MSELoss(nn.Module):
    """Takes reduction mean only for batch direction"""

    def __init__(self):
        super().__init__()

        self.mse = nn.MSELoss(reduction="none")

    def forward(self, Y, Z):  # Y, Z: both ( B, 512, 256 )
        return self.mse(Y, Z).sum(dim=(-1, -2)).mean()


class CLIPLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.compute_similarity = nn.CosineSimilarity(dim=-1)
        self._criterion = nn.CrossEntropyLoss(reduction=args.reduction)

        self.temp = nn.Parameter(torch.tensor([float(args.clip_temp_init)]))
        if not args.clip_temp_learn:
            self.temp.requires_grad = False

    def forward(self, x, y, fast=True, return_logits=False):
        batch_size = x.size(0)
        assert batch_size > 1, "Batch size must be greater than 1."
        targets = torch.arange(batch_size, requires_grad=False).long().to(device=x.device)  # fmt: skip

        if not fast:
            # less efficient way
            x_ = rearrange(x, "b f t -> 1 b (f t)")
            y_ = rearrange(y, "b f t -> b 1 (f t)")
            logits = self.compute_similarity(x_, y_)  # s

        else:
            # fast way
            x = x.reshape(batch_size, -1)
            y = y.reshape(batch_size, -1)

            # NOTE: scale the embeddings to unit norm
            x = x / x.norm(dim=-1, keepdim=True)
            y = y / y.norm(dim=-1, keepdim=True)

            # get dot products
            logits = torch.matmul(x, y.T)

        # scale by temperature
        logits *= torch.exp(self.temp)

        # NOTE: as in https://arxiv.org/abs/2103.00020
        loss = self._criterion(logits, targets) + self._criterion(logits.t(), targets) / 2  # fmt: skip

        if return_logits:
            return logits, loss
        else:
            return loss
