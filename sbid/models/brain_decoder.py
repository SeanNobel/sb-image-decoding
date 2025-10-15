import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from termcolor import cprint


class BrainDecoderBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, temporal_dim: int, ksize: int = 3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="nearest"),
            nn.Conv1d(in_dim, out_dim, ksize, padding="same"),
            nn.LayerNorm([out_dim, temporal_dim]),
            nn.GELU(),
        )

    def forward(self, X: torch.Tensor):
        return self.net(X)


class BrainDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temporal_dim: int,
        mid_channels: int = 256,
        ksize: int = 3,
    ):
        super().__init__()

        # Gradually upsample from ( b, 768, 1 ) to ( b, 271, 169 )
        self.net = nn.Sequential(
            Rearrange("b f -> b f 1"),
            BrainDecoderBlock(in_channels, mid_channels, 4, ksize),  # ( b, 128, 4 )
            BrainDecoderBlock(mid_channels, mid_channels, 16, ksize),  # ( b, 256, 16 )
            BrainDecoderBlock(mid_channels, out_channels, 64, ksize),  # ( b, 271, 64 )
            nn.Linear(64, temporal_dim),  # ( b, 271, 169 )
        )

    def forward(self, X: torch.Tensor):
        """_summary_
        Args:
            X ( b, d ): _description_
        Returns:
            X ( b, c, t ): _description_
        """
        return self.net(X)
