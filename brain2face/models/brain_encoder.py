import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio as ta
from einops.layers.torch import Rearrange
from functools import partial
from typing import Optional, Union, Callable, List, Tuple
from termcolor import cprint

from brain2face.models.utils import DropBlock1D
from brain2face.utils.layout import ch_locations_2d, DynamicChanLoc2d
from brain2face.utils.train_utils import conv_output_size


class SpatialAttention(nn.Module):
    def __init__(self, args, loc: torch.Tensor, flat: bool = True):
        super().__init__()

        self.D1 = args.D1
        self.K = args.K
        self.flat = flat
        x, y = loc.T

        # TODO: Check if those two are identical.

        if flat:  # Implementation version 1
            self.z_re = nn.Parameter(torch.Tensor(self.D1, self.K, self.K))
            self.z_im = nn.Parameter(torch.Tensor(self.D1, self.K, self.K))
            nn.init.kaiming_uniform_(self.z_re, a=np.sqrt(5))
            nn.init.kaiming_uniform_(self.z_im, a=np.sqrt(5))

            k_arange = torch.arange(self.K)
            rad1 = torch.einsum("k,c->kc", k_arange, x)
            rad2 = torch.einsum("l,c->lc", k_arange, y)
            rad = rad1.unsqueeze(1) + rad2.unsqueeze(0)
            self.register_buffer("cos", torch.cos(2 * torch.pi * rad))
            self.register_buffer("sin", torch.sin(2 * torch.pi * rad))

        else:  # Implementation version 2
            # make a complex-valued parameter, reshape k,l into one dimension
            self.z = nn.Parameter(
                torch.rand(size=(self.D1, self.K**2), dtype=torch.cfloat)
            )

            # vectorize of k's and l's
            a = []
            for k in range(self.K):
                for l in range(self.K):
                    a.append((k, l))
            a = torch.tensor(a)
            k, l = a[:, 0], a[:, 1]
            # NOTE: pre-compute the values of cos and sin (they depend on k, l, x and y which repeat)
            phi = 2 * torch.pi * (torch.einsum("k,x->kx", k, x) + torch.einsum("l,y->ly", l, y))  # fmt: skip
            self.register_buffer("cos", torch.cos(phi))
            self.register_buffer("sin", torch.sin(phi))

        self.spatial_dropout = SpatialDropout(loc, args.d_drop)

    def forward(self, X):
        """_summary_

        Args:
            X ( b, c, t ): _description_

        Returns:
            _type_: _description_
        """
        # NOTE: drop some channels within a d_drop of the sampled channel
        X = self.spatial_dropout(X)  # ( b, c, t )

        if self.flat:
            real = torch.einsum("dkl,klc->dc", self.z_re, self.cos)
            imag = torch.einsum("dkl,klc->dc", self.z_im, self.sin)
            # ( D1, c )
        else:
            real = torch.einsum("jm, me -> je", self.z.real, self.cos)
            imag = torch.einsum("jm, me -> je", self.z.imag, self.sin)

        # NOTE: to get the softmax spatial attention weights over input electrodes,
        # we don't compute exp, etc (as in the eq. 5), we take softmax instead:
        a = F.softmax(real + imag, dim=-1)  # ( D1, c )

        # NOTE: each output is a diff weighted sum over each input channel
        return torch.einsum("oi,bit->bot", a, X)


class SpatialDropout(nn.Module):
    """Using same drop center for all samples in batch"""

    def __init__(self, loc, d_drop):
        super().__init__()
        self.loc = loc  # ( num_channels, 2 )
        self.d_drop = d_drop
        self.num_channels = loc.shape[0]

    def forward(self, X):  # ( B, num_channels, seq_len )
        assert X.shape[1] == self.num_channels

        if self.training:
            drop_center = self.loc[np.random.randint(self.num_channels)]  # ( 2, )
            distances = (self.loc - drop_center).norm(dim=-1)  # ( num_channels, )
            mask = torch.where(distances < self.d_drop, 0.0, 1.0).to(device=X.device)
            # ( num_channels, )
            X = torch.einsum("c,bct->bct", mask, X)
            # cprint(1 - torch.count_nonzero(X) / torch.numel(X), "yellow")

        return X


class SubjectSpatialAttention(nn.Module):
    def __init__(self, args, loc: np.ndarray):
        super().__init__()

        self.num_channels = loc.shape[0]

        self.spatial_attention = SpatialAttention(args, loc)

        self.conv = nn.Conv1d(
            in_channels=args.D1,
            out_channels=args.D1,
            kernel_size=1,
            stride=1,
            bias=args.biases.conv_subj_sa,
        )

    def forward(self, X):
        """
        Args:
            X: ( 1, channels (+pad), timesteps )
        """
        X, pad = torch.split(
            X, [self.num_channels, X.shape[1] - self.num_channels], dim=1
        )
        assert pad.sum() == 0

        X = self.spatial_attention(X)

        X = self.conv(X)
        # X = self.conv2(X)

        return X


class SubjectBlock(nn.Module):
    def __init__(self, args, num_subjects: int, loc: np.ndarray):
        super().__init__()

        self.num_subjects = num_subjects
        self.D1 = args.D1
        self.K = args.K

        if args.spatial_attention:
            self.spatial_attention = SpatialAttention(args, loc)
        else:
            cprint("Not using spatial attention.", "yellow")
            self.spatial_attention = None

        self.conv = nn.Conv1d(
            in_channels=self.D1 if args.spatial_attention else args.num_channels,
            out_channels=self.D1,
            kernel_size=1,
            stride=1,
        )
        self.subject_layer = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=self.D1,
                    out_channels=self.D1,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                )
                for _ in range(self.num_subjects)
            ]
        )

    def forward(
        self, X: torch.Tensor, subject_idxs: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if self.spatial_attention is not None:
            X = self.spatial_attention(X)  # ( B, 270, 256 )

        X = self.conv(X)  # ( B, 270, 256 )

        if subject_idxs is not None:
            X = torch.cat(
                [
                    self.subject_layer[i](x.unsqueeze(dim=0))
                    for i, x in zip(subject_idxs, X)
                ]
            )  # ( B, 270, 256 )

        else:
            cprint("Unknown subject.", "yellow")

            X = torch.stack(
                [self.subject_layer[i](X) for i in range(self.num_subjects)]
            ).mean(dim=0)

        return X


class SubjectBlockConvDynamic(nn.Module):
    def __init__(self, args, num_subjects: int, layouts: DynamicChanLoc2d) -> None:
        super().__init__()

        self.num_subjects = num_subjects
        self.num_channels = [
            layouts.get_loc(i).shape[0] for i in range(self.num_subjects)
        ]

        self.subject_layer = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=self.num_channels[i],
                    out_channels=args.D1,
                    kernel_size=1,
                    bias=False,
                    stride=1,
                )
                for i in range(self.num_subjects)
            ]
        )

    def forward(self, X: torch.Tensor, subject_idxs: torch.Tensor) -> torch.Tensor:
        X = torch.cat(
            [
                self.subject_layer[i](
                    torch.split(
                        x,
                        [self.num_channels[i], x.shape[0] - self.num_channels[i]],
                        dim=0,
                    )[0].unsqueeze(dim=0)
                )
                for i, x in zip(subject_idxs, X)
            ]
        )

        return X


class SubjectBlockSA(nn.Module):
    """Applies Spatial Attention to each subject separately"""

    def __init__(self, args, num_subjects: int, layouts: DynamicChanLoc2d) -> None:
        super().__init__()

        self.layouts = layouts
        self.num_subjects = num_subjects

        self.subject_layer = nn.ModuleList(
            [
                SubjectSpatialAttention(args, self.layouts.get_loc(i))
                for i in range(self.num_subjects)
            ]
        )

        self.conv = nn.Conv1d(
            in_channels=args.D1,
            out_channels=args.D1,
            kernel_size=1,
            stride=1,
            bias=args.biases.conv_block,
        )

    def forward(
        self,
        X: torch.Tensor,
        subject_idxs: torch.Tensor,
        subbatch: bool = False,
    ) -> torch.Tensor:
        """Currently SubjectBlockSA doesn't allow unknown subject.
        TODO: think of how to incorporate unknown layout.
        NOTE: X dim=1 is zero-padded depending on its original channel numbers.
        FIXME: inputting samples with batch_size=1 might make learning unstable.
        """
        if subbatch:
            X = torch.cat(
                [
                    self.subject_layer[i](X[subject_idxs == i])
                    for i in range(self.num_subjects)
                ]
            )

            regather_idxs = []
            for i in range(len(subject_idxs)):
                prev, after = torch.tensor_split(subject_idxs, [i])
                after = after[1:]
                regather_idxs.append(
                    (prev <= subject_idxs[i]).sum() + (after < subject_idxs[i]).sum()
                )
            regather_idxs = torch.tensor(regather_idxs).to(X.device)

            X = torch.index_select(X, dim=0, index=regather_idxs)

        else:
            # Sequential batch size = 1 input to each subject layer
            X = torch.cat(
                [
                    self.subject_layer[i](x.unsqueeze(dim=0))
                    for i, x in zip(subject_idxs, X)
                ]
            )

        X = self.conv(X)

        return X


def get_dropout(mode: str, p_drop: float) -> nn.Module:
    if mode == "dropout":
        return nn.Dropout(p=p_drop)
    elif mode == "dropout1d":
        return nn.Dropout1d(p=p_drop)
    elif mode == "dropblock":
        return DropBlock1D(p=p_drop)
    else:
        raise ValueError(f"Unknown dropout mode: {mode}")


class Inception1DBlock(nn.Module):
    def __init__(
        self, k: int, D1: int, D2: int, drop_mode: str = "dropout", p_drop: float = 0.1
    ) -> None:
        super().__init__()

        self.k = k
        self.D2 = D2
        self.in_channels = D1 if k == 0 else D2

        self.conv1 = nn.Conv1d(self.in_channels, D2 // 4, kernel_size=1, padding="same")
        self.conv2 = nn.Conv1d(self.in_channels, D2 // 4, kernel_size=3, padding="same")
        self.conv3 = nn.Conv1d(self.in_channels, D2 // 4, kernel_size=5, padding="same")
        self.conv4 = nn.Conv1d(self.in_channels, D2 // 4, kernel_size=7, padding="same")

        self.norm = nn.BatchNorm1d(num_features=D2)
        self.dropout = get_dropout(drop_mode, p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        _X = torch.cat(
            [self.conv1(X), self.conv2(X), self.conv3(X), self.conv4(X)], dim=1
        )

        if self.k == 0:
            X = F.gelu(self.norm(_X))
        else:
            X = F.gelu(self.norm(_X + X))  # skip connection

        return self.dropout(X)


class ConvBlock(nn.Module):
    def __init__(
        self,
        k: int,
        D1: int,
        D2: int,
        ksize: int = 3,
        drop_mode: str = "dropout",
        p_drop: float = 0.1,
    ) -> None:
        super().__init__()

        self.k = k
        self.D2 = D2
        self.in_channels = D1 if k == 0 else D2

        self.conv0 = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.D2,
            kernel_size=ksize,
            padding="same",
            dilation=2 ** ((2 * k) % 5),
        )
        self.batchnorm0 = nn.BatchNorm1d(num_features=self.D2)
        self.conv1 = nn.Conv1d(
            in_channels=self.D2,
            out_channels=self.D2,
            kernel_size=ksize,
            padding="same",
            dilation=2 ** ((2 * k + 1) % 5),
        )
        self.batchnorm1 = nn.BatchNorm1d(num_features=self.D2)
        self.conv2 = nn.Conv1d(
            in_channels=self.D2,
            out_channels=2 * self.D2,
            kernel_size=ksize,
            padding="same",
            dilation=2,  # NOTE: The text doesn't say this, but the picture shows dilation=2
        )
        self.dropout = get_dropout(drop_mode, p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.k == 0:
            X = self.conv0(X)
        else:
            X = self.conv0(X) + X  # skip connection

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        X = self.conv2(X)
        X = F.glu(X, dim=-2)

        return self.dropout(X)


class Downsample1D(nn.Module):
    def __init__(self, D2: int) -> None:
        super().__init__()

        self.rearrange = Rearrange("b c (t s) -> b (c s) t", s=2)
        self.conv = nn.Conv1d(D2 * 2, D2, 1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.shape[-1] % 2 != 0:
            X = F.pad(X, (0, 1), "constant", 0)

        return self.conv(self.rearrange(X))


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        num_embeds: int,
        embed_dim: int,
        use_ema: bool = True,
        alpha: float = 1.0,
        beta: float = 0.25,
        gamma: float = 0.99,
        epsilon: float = 1e-5,
        emb_init: str = "normal",
    ) -> None:
        """
        Parameters
        ----------
        num_embeds : int
            Codebook vectors (embedding space)の数．図中のK.
        embed_dim : int
            Codebook vectorsの長さ（embedding spaceの次元数）．
        beta : float
            エンコーダの正則化項の係数．目的関数のbeta.
        """
        super().__init__()

        self.num_embeds = num_embeds
        self.embed_dim = embed_dim
        self.use_ema = use_ema
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon

        self.embedding = nn.Embedding(num_embeds, embed_dim)
        if emb_init == "normal":
            self.embedding.weight.data.normal_()
        elif emb_init == "uniform":
            self.embedding.weight.data.uniform_(-1 / num_embeds, 1 / num_embeds)

        if use_ema:
            self.register_buffer("ema_cluster_size", torch.zeros(num_embeds))

            self.ema_w = nn.Parameter(torch.Tensor(num_embeds, embed_dim))
            if emb_init == "normal":
                self.ema_w.data.normal_()
            elif emb_init == "uniform":
                self.ema_w.data.uniform_(-1 / num_embeds, 1 / num_embeds)

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Args:
            z_e ( b, embed_dim, t' ):
        Returns:
            loss : torch.Tensor (, )
                目的関数の再構成誤差以外の部分．
            z_q : torch.Tensor ( b, embed_dim, t' )
                離散化された潜在変数．
        """
        z_e = z_e.permute(0, 2, 1).contiguous()  # ( b, t', embed_dim )
        z_e_shape = z_e.shape

        z_e_flat = z_e.view(-1, self.embed_dim)  # ( b * t', embed_dim )

        # L2 distances
        distances = (
            torch.sum(z_e_flat**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_e_flat, self.embedding.weight.T)
        )
        # ( b * t', num_embeds )

        # 最も距離の近いembedding vectorのインデックス
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        # ( b * t', )
        encodings = torch.zeros(
            encoding_indices.shape[0], self.num_embeds, device=z_e.device
        )
        # ( b * t', num_embeds )
        encodings.scatter_(1, encoding_indices, 1)
        # ( b * t', num_embeds )

        z_q = torch.matmul(encodings, self.embedding.weight).view(z_e_shape)
        # ( b, t', embed_dim )

        if self.use_ema and self.training:
            self.ema_cluster_size = self.ema_cluster_size * self.gamma \
                + (1 - self.gamma) * torch.sum(encodings, 0)  # fmt: skip

            # Laplace smoothing of the cluster size
            n = torch.sum(self.ema_cluster_size.data)
            self.ema_cluster_size = (
                (self.ema_cluster_size + self.epsilon)
                / (n + self.num_embeds * self.epsilon)
                * n
            )

            dw = torch.matmul(encodings.T, z_e_flat)
            self.ema_w = nn.Parameter(self.ema_w * self.gamma + (1 - self.gamma) * dw)

            self.embedding.weight = nn.Parameter(
                self.ema_w / self.ema_cluster_size.unsqueeze(1)
            )

        # Regularization loss
        e_latent_loss = F.mse_loss(z_q.detach(), z_e)

        if self.use_ema:
            vq_loss = self.alpha * self.beta * e_latent_loss
        else:
            q_latent_loss = F.mse_loss(z_q, z_e.detach())
            vq_loss = self.alpha * (
                (1 - self.beta) * q_latent_loss + self.beta * e_latent_loss
            )

        # Straight-through estimator
        z_q = z_e + (z_q - z_e).detach()

        z_q = z_q.permute(0, 2, 1).contiguous()  # ( b, embed_dim, t' )

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return z_q, vq_loss, perplexity


class OriginalAggregator(nn.Module):
    """Original temporal aggregation module"""

    def __init__(self, args, temporal_dim: int, temporal_multiplier: int = 1) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels=args.F,
                out_channels=args.F,
                kernel_size=args.final_ksize,
                stride=args.final_stride,
            ),
            nn.Conv1d(
                in_channels=args.F,
                out_channels=args.F,
                kernel_size=args.final_ksize,
                stride=args.final_stride,
            ),
            nn.Flatten(),
            nn.Linear(
                args.F * temporal_dim,
                args.F * temporal_multiplier,
                bias=args.biases.linear_reduc_time,
            ),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.layers(X)  # ( b, F * temporal_multiplier )


class TemporalAggregation(nn.Module):
    def __init__(self, args, temporal_dim: int, multiplier: int = 1) -> None:
        super().__init__()

        if args.temporal_aggregation == "original":
            self.layers = OriginalAggregator(args, temporal_dim, multiplier)
        else:
            """Modified from: https://ai.meta.com/static-resource/image-decoding"""
            self.layers = nn.Sequential()

            self.layers.add_module(
                "linear_projection",
                nn.Conv1d(
                    in_channels=args.F,
                    out_channels=args.F * 4 * multiplier,
                    kernel_size=1,
                ),
            )

            if args.temporal_aggregation == "affine":
                self.layers.add_module(
                    "temporal_aggregation", nn.Linear(temporal_dim, 1)
                )
            elif args.temporal_aggregation == "pool":
                self.layers.add_module("temporal_aggregation", nn.AdaptiveAvgPool1d(1))
            else:
                raise NotImplementedError()

            self.layers.add_module(
                "mlp_projector",
                nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(args.F * 4 * multiplier, args.F * 2 * multiplier),
                    nn.GELU(),
                    nn.Linear(args.F * 2 * multiplier, args.F * multiplier),
                    nn.GELU(),
                ),
            )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.layers(X)  # ( b, F * multiplier )


class AggregatedVectorQuantizer(nn.Module):
    def __init__(
        self,
        args,
        num_concepts: int,
        num_embeds: int,
        embed_dim: int,
        temporal_dim: int,
        use_ema: bool = True,
        alpha: float = 1.0,
        beta: float = 0.25,
        gamma: float = 0.99,
        epsilon: float = 1e-5,
    ) -> None:
        super().__init__()

        self.aggregator = nn.Sequential(
            TemporalAggregation(args, temporal_dim, multiplier=num_concepts),
            nn.Unflatten(dim=1, unflattened_size=(embed_dim, num_concepts)),
        )

        self.vector_quantizer = VectorQuantizer(
            num_embeds, embed_dim, use_ema, alpha, beta, gamma, epsilon
        )

        self.mlp_projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(embed_dim * num_concepts, embed_dim),
            nn.GELU(),
        )

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor]:
        X = self.aggregator(X)

        X, reg_loss, perplexity = self.vector_quantizer(X)

        X = self.mlp_projector(X)

        return X, reg_loss, perplexity


class BrainEncoder(nn.Module):
    def __init__(
        self,
        args,
        subject_names: List[str],
        layout: Union[Callable, DynamicChanLoc2d] = ch_locations_2d,
        vq: bool = False,
        num_conv_blocks: int = 5,
        downsample: Union[bool, List[bool]] = False,
        temporal_aggregation: Optional[str] = None,
        unknown_subject: bool = False,
    ) -> None:
        super().__init__()

        self.D1 = args.D1
        self.D2 = args.D2
        self.F = args.F

        if isinstance(downsample, bool):
            downsample = [downsample] * num_conv_blocks
        else:
            assert len(downsample) == num_conv_blocks, "downsample and num_conv_blocks should have the same length."  # fmt: skip

        self.vq = vq
        self.unknown_subject = unknown_subject

        if layout == ch_locations_2d:
            self.subject_block = SubjectBlock(args, len(subject_names), layout(args))

        elif layout == DynamicChanLoc2d:
            if args.spatial_attention:
                self.subject_block = SubjectBlockSA(
                    args, len(subject_names), layout(args, subject_names)
                )
            else:
                self.subject_block = SubjectBlockConvDynamic(
                    args, len(subject_names), layout(args, subject_names)
                )
        else:
            raise TypeError

        block_args = {
            "D1": self.D1,
            "D2": self.D2,
            "drop_mode": args.drop_mode,
            "p_drop": args.p_drop,
        }

        self.conv_blocks = nn.Sequential()
        for k in range(num_conv_blocks):
            if args.conv_block == "dilated_conv":
                self.conv_blocks.add_module(
                    f"conv{k}", ConvBlock(k, ksize=args.ksizes.conv_block, **block_args)
                )
            elif args.conv_block == "inception":
                self.conv_blocks.add_module(
                    f"conv{k}", Inception1DBlock(k, **block_args)
                )
            else:
                raise NotImplementedError()

            if downsample[k]:
                self.conv_blocks.add_module(f"downsample{k}", Downsample1D(self.D2))

        self.conv_final1 = nn.Conv1d(
            in_channels=self.D2,
            out_channels=2 * self.D2,
            kernel_size=args.final_ksize,
            stride=args.final_stride,
        )
        self.conv_final2 = nn.Conv1d(
            in_channels=2 * self.D2,
            out_channels=self.F,
            kernel_size=args.final_ksize,
            stride=args.final_stride,
        )

        temporal_dim = conv_output_size(
            int(args.seq_len * args.brain_resample_sfreq),
            ksize=args.final_ksize,
            stride=args.final_stride,
            repetition=4 if args.temporal_aggregation == "original" else 2,
            downsample=sum(downsample),
        )

        vq_args = {
            "num_embeds": args.vq_num_embeds,
            "embed_dim": args.F,
            "use_ema": args.vq_use_ema,
            "alpha": args.vq_alpha,
            "beta": args.vq_beta,
            "gamma": args.vq_gamma,
            "epsilon": args.vq_epsilon,
        }

        if vq and args.vq_type == "aggregated":
            self.vector_quantizer = AggregatedVectorQuantizer(
                args,
                num_concepts=args.vq_num_concepts,
                temporal_dim=temporal_dim,
                **vq_args,
            )

            self.temporal_aggregation = None
        else:
            if vq:
                self.vector_quantizer = VectorQuantizer(**vq_args)

            if temporal_aggregation is not None:
                self.temporal_aggregation = TemporalAggregation(args, temporal_dim)
            else:
                self.temporal_aggregation = None

    def forward(
        self, X: torch.Tensor, subject_idxs: Optional[torch.Tensor]
    ) -> torch.Tensor:
        assert self.unknown_subject or subject_idxs is not None, "You need to provide subject_idxs when it's not unknown subject."  # fmt: skip

        X = self.subject_block(X, subject_idxs)

        X = self.conv_blocks(X)

        X = F.gelu(self.conv_final1(X))
        X = F.gelu(self.conv_final2(X))

        if self.vq:
            X, reg_loss, perplexity = self.vector_quantizer(X)

        if self.temporal_aggregation is not None:
            X = self.temporal_aggregation(X)

        if self.vq:
            return X, reg_loss, perplexity
        else:
            return X
