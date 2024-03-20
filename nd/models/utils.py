import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class DropBlock1D(nn.Module):
    """
    Dropout on conv is weird: https://towardsdatascience.com/dropout-on-convolutional-layers-is-weird-5c6ab14f19b2
    DropBlock paper: https://proceedings.neurips.cc/paper_files/paper/2018/file/7edcfb2d8f6a659ef4cd1e6c9b6d7079-Paper.pdf
    Implementation modified from: https://github.com/miguelvr/dropblock/issues/30
    """

    def __init__(self, p: float = 0.2, block_size: int = 3):
        super().__init__()
        self.block_size = block_size
        self.gamma = p / (block_size**2)

    def forward(self, X):
        if not self.training:
            return X

        mask = torch.rand(X.shape[0], *X.shape[2:]) < self.gamma

        bm = self._compute_block_mask(mask.float().to(X.device))

        return X * bm[:, None, :] * bm.numel() / bm.sum()

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool1d(
            input=mask[:, None, :],
            kernel_size=self.block_size,
            stride=1,
            padding=self.block_size // 2,
        )

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1]

        return 1 - block_mask.squeeze(1)


class MLPTemporalReducer(nn.Module):
    def __init__(self, in_tokens: int, out_tokens: int):
        super().__init__()

        # self.net = nn.Sequential(
        #     Rearrange("b t d -> b d t"),
        #     nn.Linear(in_tokens, 512),
        #     nn.LayerNorm(512),
        #     nn.GELU(),
        #     nn.Linear(512, 128),
        #     nn.LayerNorm(128),
        #     nn.GELU(),
        #     nn.Linear(128, out_tokens),
        #     Rearrange("b d t -> b t d"),
        # )
        self.net = nn.Sequential(
            Rearrange("b t d -> b d t"),
            nn.Linear(in_tokens, out_tokens),
            Rearrange("b d t -> b t d"),
        )

    def forward(self, X):
        return self.net(X)

    def encode(self, X):
        return self(X)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            Rearrange("b t d -> b (t d)"),
            nn.Linear(in_dim, in_dim // 4),
            nn.LayerNorm(in_dim // 4),
            nn.GELU(),
            nn.Linear(in_dim // 4, out_dim),
        )

    def forward(self, X):
        return self.net(X)

    def encode(self, X):
        return self(X)
