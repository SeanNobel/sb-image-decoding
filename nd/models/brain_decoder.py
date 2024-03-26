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
    def __init__(self, args):
        super().__init__()

        in_dim: int = args.vae_zdim
        mid_dim: int = 256
        num_channels: int = args.num_channels
        final_temporal_dim: int = int(args.seq_len * args.brain_resample_sfreq)
        ksize: int = args.conv_block_ksize

        # Gradually upsample from ( b, 768, 1 ) to ( b, 271, 169 )
        self.net = nn.Sequential(
            Rearrange("b f -> b f 1"),
            BrainDecoderBlock(in_dim, mid_dim // 2, 4, ksize),  # ( b, 128, 4 )
            BrainDecoderBlock(mid_dim // 2, mid_dim, 16, ksize),  # ( b, 256, 16 )
            BrainDecoderBlock(mid_dim, num_channels, 64, ksize),  # ( b, 271, 64 )
            nn.Linear(64, final_temporal_dim),  # ( b, 271, 169 )
        )

    def forward(self, X: torch.Tensor):
        """_summary_
        Args:
            X ( l * b, vae_zdim ): _description_
        Returns:
            X ( l * b, c, t ): _description_
        """
        return self.net(X)
