import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops import rearrange
from termcolor import cprint

from nd.models import BrainEncoder, BrainDecoder
from nd.models.transformer import PositionalEncoding


class BrainAutoencoder(nn.Module):
    def __init__(self, args, subjects) -> None:
        super().__init__()

        self.encoder = BrainEncoder(args, subjects)
        self.decoder = BrainDecoder(
            args.F_mse,
            args.num_channels,
            int(args.seq_len * args.resample_freq),
            mid_channels=args.decoder_dim,
        )

    def forward(self, X: torch.Tensor, subject_idxs: torch.Tensor):
        Z = self.encode(X, subject_idxs)

        if Z.ndim == 3:
            Z = rearrange(Z, "b () f -> b f")

        return self.decoder(Z)

    def encode(self, X: torch.Tensor, subject_idxs: torch.Tensor):
        return self.encoder(X, subject_idxs)["Z_mse"]


class BrainMAE(BrainAutoencoder):
    def __init__(self, args, subjects, mask_ratio: float = 0.75) -> None:
        super().__init__(args, subjects)

        self.mask_ratio = mask_ratio

        D1, D2 = args.D1, args.D2
        patch_size: int = args.patch_size
        pos_enc: str = args.pos_enc

        init_temporal_dim = int(args.seq_len * args.resample_freq)
        temporal_dim = int(init_temporal_dim // patch_size * (1 - self.mask_ratio))

        self.encoder = BrainEncoder(args, subjects, temporal_dim=temporal_dim)

        self.patch_embed = nn.Sequential(
            nn.Conv1d(D1, D2, kernel_size=patch_size, stride=patch_size),
            Rearrange("b d t -> b t d"),
        )

        # Override
        self.pos_enc = PositionalEncoding(
            init_temporal_dim // patch_size, D2, pos_enc.split("_")[0]
        )

    def encode(self, X: torch.Tensor, subject_idxs: torch.Tensor):
        if hasattr(self.encoder, "subject_block"):
            X = self.encoder.subject_block(X, subject_idxs)
        else:
            X = self.encoder.spatial_attention(X)

        X = self.patch_embed(X)  # ( b, t // patch_size, D2 )

        X = self.pos_enc(X)

        X, _, _ = self._random_mask(X)
        X = rearrange(X, "b t d -> b d t")

        X = self.encoder.blocks(X)

        X = F.gelu(self.encoder.conv_final(X))

        X = self.encoder.temporal_aggregation(X)

        return self.encoder.mse_head(X)

    def _random_mask(self, x):
        """https://github.com/bbaaii/DreamDiffusion/blob/main/code/sc_mbm/mae_for_eeg.py
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - self.mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
