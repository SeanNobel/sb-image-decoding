import sys
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from termcolor import cprint

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from typing import Optional, Union

from sbid.utils.train_utils import conv_output_size


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim, heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        dim,
        depth,
        heads,
        mlp_dim,
        num_classes: Optional[int] = None,
        pool="cls",
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        if num_classes is not None:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(dim), nn.Linear(dim, num_classes)
            )
        else:
            self.mlp_head = None

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.to_latent(x)

        if self.mlp_head is not None:
            x = self.mlp_head(x)

        return x


class ViViT(nn.Module):
    def __init__(
        self,
        image_size,
        patch_size,
        num_frames,
        dim=192,
        depth=4,
        heads=3,
        pool="cls",
        in_channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        scale_dim=4,
    ):
        super().__init__()

        # assert pool in {'cls',
        #                 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        assert (
            image_size % patch_size == 0
        ), "Image dimensions must be divisible by the patch size."
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size**2
        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c t (h p1) (w p2) -> b t (h w) (p1 p2 c)",
                p1=patch_size,
                p2=patch_size,
            ),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_frames, num_patches + 1, dim)
        )
        # self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = Transformer(
            dim, depth, heads, dim_head, dim * scale_dim, dropout
        )

        # self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(
            dim, depth, heads, dim_head, dim * scale_dim, dropout
        )

        self.dropout = nn.Dropout(emb_dropout)
        # self.pool = pool

        # self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, t, n, _ = x.shape

        # cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', b=b, t=t)
        # x = torch.cat((cls_space_tokens, x), dim=2)
        x += self.pos_embedding[:, :, :n]  # self.pos_embedding[:, :, :(n + 1)]
        x = self.dropout(x)

        x = rearrange(x, "b t n d -> (b t) n d")
        x = self.space_transformer(x)
        x = rearrange(x[:, 0], "(b t) ... -> b t ...", b=b)

        # cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        # x = torch.cat((cls_temporal_tokens, x), dim=1)

        h = self.temporal_transformer(x)
        # print(h.shape)

        # x = h.mean(dim=1) if self.pool == 'mean' else h[:, 0]

        return h.permute(0, 2, 1)


class ViViTReduceTime(nn.Module):
    def __init__(self, dim: int, num_frames: int, *args, **kwargs):
        super().__init__()

        self.vivit = ViViT(*args, dim=dim, num_frames=num_frames, **kwargs)

        self.reduce_time = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, stride=2),
            nn.SiLU(),
            nn.Conv1d(dim, dim, kernel_size=3, stride=2),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(
                in_features=dim
                * conv_output_size(num_frames, ksize=3, stride=2, repetition=2),
                out_features=dim,
            ),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.vivit(x)

        return self.reduce_time(x)


def Downsample3D(dim: int, dim_out: int):
    # https://arxiv.org/abs/2208.03641 shows this is the most optimal way to downsample
    # named SP-conv in the paper, but basically a pixel unshuffle
    return nn.Sequential(
        Rearrange("b c t (h s1) (w s2) -> b (c s1 s2) t h w", s1=2, s2=2),
        nn.Conv3d(dim * 4, dim_out, 1),
    )


def Block3D(dim: int, dim_out: int, groups: int = 8):
    return nn.Sequential(
        nn.Conv3d(
            dim, dim_out, kernel_size=(5, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1)
        ),
        nn.GroupNorm(groups, dim_out),
        nn.SiLU(),
    )


class Unet3DEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        image_size: int,
        num_frames: int,
        num_layers: int = 3,
        in_channels: int = 3,
        init_conv_dim: int = 16,
    ):
        super().__init__()

        mid_dims = [in_channels, init_conv_dim]
        for _ in range(num_layers - 1):
            mid_dims.append(mid_dims[-1] * 2)

        in_out = list(zip(mid_dims[:-1], mid_dims[1:]))
        # e.g. [[(3, 16), (16, 32), (32, 64)]

        self.downs = nn.ModuleList([])

        for dim_in, dim_out in in_out:
            self.downs.append(
                nn.Sequential(
                    Downsample3D(dim_in, dim_out),
                    Block3D(dim_out, dim_out),
                    # Block3D(dim_out, dim_out),
                    Downsample3D(dim_out, dim_out),
                )
            )

        flat_dim = (
            mid_dims[-1]
            * conv_output_size(num_frames, 5, 2, num_layers)
            * (image_size // ((2 * 2) ** num_layers)) ** 2
        )

        self.to_out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, dim),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x ( b, c, t, h, w ): Videos.
        Returns:
            x: _description_
        """
        for down in self.downs:
            x = down(x)

        return self.to_out(x)


class OpenFaceMapper(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hid_dim: int,
        out_channels: int,
        ksize_stride: int,
        reduce_time: bool,
        seq_len: Union[float, int],
        fps: int,
        time_multiplier: int = 1,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels, hid_dim, kernel_size=3, stride=1, padding="same"
        )
        self.batchnorm = nn.BatchNorm1d(num_features=hid_dim)

        self.conv2 = nn.Conv1d(
            hid_dim, out_channels, kernel_size=3, stride=1, padding="same"
        )

        if reduce_time:
            self.flatten = nn.Flatten()
            self.linear = nn.Linear(
                in_features=out_channels * int(seq_len * fps),
                out_features=out_channels * time_multiplier,
            )

        else:
            self.linear = None

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.conv1(X)
        X = F.gelu(self.batchnorm(X))

        X = F.gelu(self.conv2(X))

        if self.linear is not None:
            X = self.linear(self.flatten(X))

        return X


if __name__ == "__main__":
    video = torch.ones([64, 90, 1, 64, 64]).cuda()

    model = ViViT(
        image_size=64, patch_size=16, num_frames=90, dim=512, depth=2, in_channels=1
    ).cuda()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print("Trainable Parameters: %.3fM" % parameters)

    out = model(video)

    print("Shape of out :", out.shape)  # [B, num_classes]
