import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio as ta
from einops import rearrange
from einops.layers.torch import Rearrange
from functools import partial
from typing import Optional, Union, Callable, List, Tuple
from termcolor import cprint

from fairseq.modules.conformer_layer import ConformerEncoderLayer
from fairseq.modules import RelPositionalEncoding

# from conformer.encoder import ConformerBlock as ConformerBlock2
from conformer.conformer.encoder import ConformerBlock as ConformerBlock2

from nd.models.vector_quantizer import get_vector_quantizer, AggregatedVectorQuantizer
from nd.models.transformer import (
    SelfAttention,
    FeedForward,
    PreNorm,
    Residual,
    PositionalEncoding,  # positional_encoding,
    relative_positional_encoding,
)
from nd.models.crate import MSSA, ISTA
from nd.models.dann import DANN
from nd.models.utils import DropBlock1D
from nd.models.subject_sa import SubjectBlockSA, SubjectBlockConvDynamic
from nd.utils.layout import ch_locations_2d, DynamicChanLoc2d
from nd.utils.train_utils import conv_output_size
from nd.utils.power_spherical import PowerSpherical
from nd.utils.von_mises_fisher import VonMisesFisher


def is_in(s: Optional[str], _s: str) -> bool:
    if s is None:
        return False
    else:
        return _s in s


class SpatialAttention(nn.Module):
    def __init__(self, loc: torch.Tensor, D1: int, K: int, d_drop: float = 0.1, flat: bool = True):
        super().__init__()

        self.flat = flat
        x, y = loc.T

        # TODO: Check if those two are identical.

        if flat:  # Implementation version 1
            self.z_re = nn.Parameter(torch.Tensor(D1, K, K))
            self.z_im = nn.Parameter(torch.Tensor(D1, K, K))
            nn.init.kaiming_uniform_(self.z_re, a=np.sqrt(5))
            nn.init.kaiming_uniform_(self.z_im, a=np.sqrt(5))

            k_arange = torch.arange(K)
            rad1 = torch.einsum("k,c->kc", k_arange, x)
            rad2 = torch.einsum("l,c->lc", k_arange, y)
            rad = rad1.unsqueeze(1) + rad2.unsqueeze(0)
            self.register_buffer("cos", torch.cos(2 * torch.pi * rad))
            self.register_buffer("sin", torch.sin(2 * torch.pi * rad))

        else:  # Implementation version 2
            # make a complex-valued parameter, reshape k,l into one dimension
            self.z = nn.Parameter(torch.rand(size=(D1, K**2), dtype=torch.cfloat))

            # vectorize of k's and l's
            a = []
            for k in range(K):
                for l in range(K):
                    a.append((k, l))
            a = torch.tensor(a)
            k, l = a[:, 0], a[:, 1]
            # NOTE: pre-compute the values of cos and sin (they depend on k, l, x and y which repeat)
            phi = 2 * torch.pi * (torch.einsum("k,x->kx", k, x) + torch.einsum("l,y->ly", l, y))  # fmt: skip
            self.register_buffer("cos", torch.cos(phi))
            self.register_buffer("sin", torch.sin(phi))

        self.spatial_dropout = SpatialDropout(loc, d_drop)

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


class SubjectBlock(nn.Module):
    def __init__(self, args, num_subjects: int, loc: np.ndarray):
        super().__init__()

        self.num_subjects = num_subjects
        self.D1 = args.D1
        self.K = args.K

        if args.spatial_attention:
            self.spatial_attention = SpatialAttention(loc, self.D1, self.K, args.d_drop)
        else:
            cprint("Not using spatial attention.", "yellow")

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

    def forward(self, X: torch.Tensor, subject_idxs: Optional[torch.Tensor]) -> torch.Tensor:
        if hasattr(self, "spatial_attention"):
            X = self.spatial_attention(X)

        X = self.conv(X)

        if subject_idxs is not None:
            X = torch.cat(
                [self.subject_layer[i](x.unsqueeze(dim=0)) for i, x in zip(subject_idxs, X)]
            )

        else:
            cprint("Unknown subject.", "yellow")

            X = torch.stack(
                [self.subject_layer[i](X) for i in range(self.num_subjects)]
            ).mean(dim=0)  # fmt: skip

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
        _X = torch.cat([self.conv1(X), self.conv2(X), self.conv3(X), self.conv4(X)], dim=1)

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


class TransformerBlock(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        n_heads: int,
        block_size: int,
        rel_pos: bool = False,
        p_drop: float = 0.1,
    ):
        super().__init__()

        self.rel_pos = rel_pos

        if rel_pos:
            self.register_buffer("pos_enc_k", relative_positional_encoding(block_size, self.d_qk))
            # ( t, t, d_qk )
            self.register_buffer("pos_enc_v", relative_positional_encoding(block_size, self.d_v))
            # ( t, t, d_v )

        self.attn = Residual(PreNorm(SelfAttention(emb_dim, n_heads, block_size), emb_dim))

        self.ff = FeedForward(emb_dim, ff_pdrop=p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X ( b, d, t ): _description_
        Returns:
            X ( b, d, t ): _description_
        """
        X = X.permute(0, 2, 1)

        if self.rel_pos:
            X = self.attn(X, self.pos_enc_k, self.pos_enc_v)
        else:
            X = self.attn(X)

        X = self.ff(X)

        return X.permute(0, 2, 1)


class ConformerBlock(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        n_heads: int,
        block_size: int,
        depthwise_ksize: int,
        activation_fn: str = "swish",
        attn_type: Optional[str] = None,
        pos_enc: str = "sine_abs",
        p_drop: float = 0.1,
        use_fp16: bool = False,
    ):
        super().__init__()

        if attn_type is not None:
            pe_dict = {"sine_abs": "abs", "sine_rel": "rel_pos", "rotary": "rope"}
            pos_enc = pe_dict[pos_enc]

        if pos_enc == "rel_pos":
            self.rel_pos = RelPositionalEncoding(block_size, emb_dim)

        self.attn = ConformerEncoderLayer(
            embed_dim=emb_dim,
            ffn_embed_dim=emb_dim * 4,
            attention_heads=n_heads,
            dropout=p_drop,
            use_fp16=use_fp16,
            depthwise_conv_kernel_size=depthwise_ksize,
            activation_fn=activation_fn,
            attn_type=attn_type,
            pos_enc_type=pos_enc,
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X ( b, d, t ): _description_
        Returns:
            X ( b, d, t ): _description_
        """
        X = rearrange(X, "b d t -> t b d")

        if hasattr(self, "rel_pos"):
            positions = self.rel_pos(X)
        else:
            positions = None

        X, _ = self.attn(X, encoder_padding_mask=None, position_emb=positions)

        return rearrange(X, "t b d -> b d t")


class CRATEBlock(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        n_heads: int,
        dim_head: int = 64,
        p_drop: float = 0.1,
        step_size: float = 0.1,
    ):
        super().__init__()

        self.attn = Residual(
            PreNorm(MSSA(emb_dim, heads=n_heads, dim_head=dim_head, dropout=p_drop), emb_dim)
        )
        self.ff = PreNorm(ISTA(emb_dim, step_size=step_size), emb_dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X ( b, d, t ): _description_
        Returns:
            X ( b, d, t ): _description_
        """
        return self.ff(self.attn(X.permute(0, 2, 1))).permute(0, 2, 1)


class Downsample1D(nn.Module):
    def __init__(self, D2: int) -> None:
        super().__init__()

        self.rearrange = Rearrange("b c (t s) -> b (c s) t", s=2)
        self.conv = nn.Conv1d(D2 * 2, D2, 1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.shape[-1] % 2 != 0:
            X = F.pad(X, (0, 1), "constant", 0)

        return self.conv(self.rearrange(X))


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
                bias=True,  # args.biases.linear_reduc_time,
            ),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.layers(X)  # ( b, F * temporal_multiplier )


class TemporalAggregation(nn.Module):
    def __init__(
        self,
        temporal_dim: int,
        embed_dim: Optional[int] = None,
        mode: str = "affine",
        multiplier: int = 1,
        args=None,  # FIXME
    ) -> None:
        super().__init__()

        if embed_dim is None:
            embed_dim = args.D3

        # FIXME: Other than affine may not be working

        if mode == "original":
            assert args is not None
            self.layers = OriginalAggregator(args, temporal_dim, multiplier)
        else:
            """Modified from: https://ai.meta.com/static-resource/image-decoding"""
            self.layers = nn.Sequential()

            # NOTE: conv_final corresponds to linear projection in the paper as long as the kernel size and stride are 1
            self.layers.add_module(
                "linear_projection",
                nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1),
            )

            if mode == "affine":
                self.layers.add_module("temporal_agg", nn.Linear(temporal_dim, multiplier))  # fmt: skip
            elif mode == "pool":
                self.layers.add_module("temporal_agg", nn.AdaptiveAvgPool1d(multiplier))
            else:
                raise NotImplementedError()

            # NOTE: MLP projectors are provided for CLIP and MSE
            self.layers.add_module(
                "mlp_projector",
                nn.Sequential(
                    Rearrange("b d t -> b (d t)"),
                    nn.Linear(embed_dim * multiplier, embed_dim * multiplier),
                    nn.GELU(),
                    Rearrange("b (d t) -> b d t", d=embed_dim),
                ),
            )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.layers(X)  # ( b, F * multiplier )


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, t: int = 1) -> None:
        super().__init__()

        self.net = nn.Sequential(
            Rearrange("b d t -> b (t d)"),
            nn.Linear(in_dim * t, in_dim * t // 2),
            nn.LayerNorm([in_dim * t // 2]),
            nn.GELU(),
            nn.Linear(in_dim * t // 2, out_dim * t),
            Rearrange("b (t d) -> b t d", d=out_dim),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X)


class BrainEncoderBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def encode(
        self,
        X: torch.Tensor,
        subject_idxs: Optional[torch.Tensor],
        return_mse: bool = True,
        normalize: bool = False,
        swap_dims: bool = False,
        stats: Optional[Tuple[float]] = None,
        device=None,
    ) -> torch.Tensor:
        if device is not None:
            orig_device = X.device
            X, subject_idxs = X.to(device), subject_idxs.to(device)

        single = X.dim == 2
        if single:
            X = X.unsqueeze(0)

            if subject_idxs is not None:
                subject_idxs = subject_idxs.unsqueeze(0)

        Z = self(X, subject_idxs)
        Z = Z["Z_mse"] if return_mse else Z["Z_clip"]

        _, t, d = Z.shape
        Z = rearrange(Z, "b t d -> b (t d)")

        if normalize:
            Z /= Z.norm(dim=-1, keepdim=True)

        if stats is not None:
            # Inverse normalization
            Z = (Z - Z.mean()) / Z.std()
            mean, std = stats
            Z = Z * std + mean

        Z = rearrange(Z, "b (t d) -> b t d", d=d)

        if swap_dims:
            Z = rearrange(Z, "b t d -> b d t")

        if device is not None:
            Z = Z.to(orig_device)

        if single:
            Z = Z.squeeze(0)

        return Z


class BrainEncoder(BrainEncoderBase):
    def __init__(
        self,
        args,
        subjects: Union[int, List[str]],
        temporal_dim: Optional[int] = None,
        # unknown_subject: bool = False,
    ) -> None:
        super().__init__()

        # Parameters
        self.ignore_subjects: bool = args.ignore_subjects or args.dann or subjects == 1
        self.init_temporal_dim: int = int(args.seq_len * args.resample_freq) if temporal_dim is None else temporal_dim  # fmt: skip
        self.vq = args.vq
        self.vae: Optional[str] = args.vae
        self.sample_l: int = args.sample_l
        # self.unknown_subject = unknown_subject

        D1, D2, D3, F, K = args.D1, args.D2, args.D3, args.F, args.K
        F_mse: int = args.F_mse
        num_clip_tokens: int = args.num_clip_tokens
        num_blocks: int = args.num_blocks
        num_subjects: int = subjects if isinstance(subjects, int) else len(subjects)
        layout: Union[ch_locations_2d, DynamicChanLoc2d] = eval(args.layout)
        spatial_attention: bool = args.spatial_attention
        pos_enc: str = args.pos_enc
        blocks: str = args.blocks
        conv_block_ksize: int = args.conv_block_ksize
        depthwise_ksize: int = args.depthwise_ksize
        downsample: Union[bool, List[bool]] = args.downsample
        temporal_agg: str = args.temporal_aggregation
        transformer_heads: int = args.transformer_heads
        p_drop: float = args.p_drop
        d_drop: float = args.d_drop
        drop_mode: str = args.drop_mode
        final_ksize: int = args.final_ksize
        final_stride: int = args.final_stride
        vq_aggregated: bool = args.vq_aggregated
        dann: bool = args.dann
        dann_scale: float = args.dann_scale
        vae_zdim: int = args.vae_zdim

        if isinstance(downsample, bool):
            downsample = [downsample] * num_blocks
        else:
            assert len(downsample) == num_blocks, "downsample and num_blocks should have the same length."  # fmt: skip

        if layout == ch_locations_2d:
            if self.ignore_subjects:
                self.spatial_attention = nn.Sequential(
                    SpatialAttention(layout(args), D1, K, d_drop),
                    nn.Conv1d(D1, D1, kernel_size=1, stride=1),
                )
            else:
                self.subject_block = SubjectBlock(args, num_subjects, layout(args))

        elif layout == DynamicChanLoc2d:
            assert isinstance(subjects, list), "subjects should be list of str when using DynamicChanLoc2d."  # fmt: skip
            assert not self.ignore_subjects, "Cannot ignore subjects when channel locations are different among them."  # fmt: skip

            if spatial_attention:
                self.subject_block = SubjectBlockSA(args, len(subjects), layout(args, subjects))  # fmt: skip
            else:
                self.subject_block = SubjectBlockConvDynamic(args, len(subjects), layout(args, subjects))  # fmt: skip
        else:
            raise TypeError(f"Unknown layout type: {layout}")

        if blocks in ["transformer", "conformer", "crate"]:
            self.tf_proj = nn.Conv1d(D1, D2, kernel_size=1)

            if is_in(pos_enc, "abs") and not (blocks == "conformer" and args.conformer_impl == 1):
                self.pos_enc = PositionalEncoding(self.init_temporal_dim, D2, pos_enc.split("_")[0])
                cprint("Putting PE at the beginning.", "yellow")
            else:
                cprint("No PE at the beginning.", "yellow")

        self.blocks = nn.Sequential()
        for k in range(num_blocks):
            if blocks == "dilated_conv":
                cprint(f"Block{k}: dilated_conv", "magenta")

                self.blocks.add_module(
                    f"block{k}",
                    ConvBlock(k, D1, D2, ksize=conv_block_ksize, drop_mode=drop_mode, p_drop=p_drop),  # fmt: skip
                )
            elif blocks == "inception":
                cprint(f"Block{k}: inception", "magenta")

                self.blocks.add_module(
                    f"block{k}",
                    Inception1DBlock(k, D1, D2, drop_mode=drop_mode, p_drop=p_drop),
                )
            elif blocks == "transformer":
                cprint(f"Block{k}: transformer", "magenta")

                self.blocks.add_module(
                    f"block{k}",
                    TransformerBlock(
                        D2, transformer_heads, self.init_temporal_dim, rel_pos=is_in(pos_enc, "rel"), p_drop=p_drop  # fmt: skip
                    ),
                )
            elif blocks == "conformer":
                # NOTE: .index only returns the first occurrence
                # attn_type = "espnet" if k == blocks.index("conformer") else None
                # cprint(f"Block{k}: conformer with PE {'off' if attn_type is None else pos_enc}", "magenta")  # fmt: skip
                cprint(f"Block{k}: conformer version {args.conformer_impl}", "magenta")

                if args.conformer_impl == 0:
                    self.blocks.add_module(
                        f"block{k}",
                        ConformerBlock(
                            D2, transformer_heads, self.init_temporal_dim, depthwise_ksize, pos_enc=pos_enc, p_drop=p_drop  # fmt: skip
                        ),
                    )
                elif args.conformer_impl == 1:
                    self.blocks.add_module(
                        f"block{k}",
                        nn.Sequential(
                            Rearrange("b d t -> b t d"),
                            ConformerBlock2(D2, transformer_heads),
                            Rearrange("b t d -> b d t"),
                        ),
                    )
            elif blocks == "crate":
                pe = pos_enc if k == blocks.index("crate") else None
                cprint(f"Block{k}: CRATE with pos_enc {pe}", "magenta")

                self.blocks.add_module(
                    f"block{k}",
                    CRATEBlock(D2, transformer_heads, p_drop=p_drop),
                )
            else:
                raise NotImplementedError()

            if downsample[k]:
                self.blocks.add_module(f"downsample{k}", Downsample1D(D2))

        self.conv_final = nn.Conv1d(
            in_channels=D2,
            out_channels=D3,
            kernel_size=final_ksize,
            stride=final_stride,
        )

        temporal_dim = conv_output_size(
            self.init_temporal_dim,
            ksize=final_ksize,
            stride=final_stride,
            repetition=3 if temporal_agg == "original" else 1,
            downsample=sum(downsample),
        )

        if temporal_agg is not None:
            self.temporal_aggregation = TemporalAggregation(
                temporal_dim, multiplier=num_clip_tokens, args=args
            )
        else:
            self.temporal_aggregation = None

        if self.vq is not None:
            dim = {"middle1": D1, "middle2": D2, "end": D3}[self.vq]
            self.vector_quantizer = get_vector_quantizer(args, dim)

            if vq_aggregated:
                assert not "middle" in self.vq, "Cannot aggregate time in the middle of the model."  # fmt: skip

                self.vector_quantizer = AggregatedVectorQuantizer(
                    args,
                    TemporalAggregation,
                    dim,
                    self.vector_quantizer,
                    temporal_dim=temporal_dim,
                )

                self.temporal_aggregation = None

        self.clip_head = MLPHead(in_dim=D3, out_dim=F, t=num_clip_tokens)
        self.mse_head = MLPHead(in_dim=D3, out_dim=F_mse, t=num_clip_tokens)

        if dann:
            self.dann_head = DANN(in_dim=D3, num_domains=num_subjects, scale=dann_scale)  # fmt: skip

        if self.vae:
            assert num_clip_tokens == 1, "Variational auto-encoding is only supported for single clip token."  # fmt: skip

            out_dim = vae_zdim + 1 if self.vae in ["vmf", "ps"] else vae_zdim * 2

            self.q_head = nn.Sequential(
                MLPHead(in_dim=D3, out_dim=out_dim, t=num_clip_tokens),
                Rearrange("b t d -> b (t d)"),
            )

    def forward(self, X: torch.Tensor, subject_idxs: Optional[torch.Tensor]) -> torch.Tensor:
        assert self.ignore_subjects or subject_idxs is not None, "You need to provide subject_idxs when it's not unknown subject."  # fmt: skip

        if hasattr(self, "subject_block"):
            X = self.subject_block(X, subject_idxs)
        else:
            X = self.spatial_attention(X)
        # ( b, D1, t )

        if self.vq == "middle1":
            X, vq_loss, perplexity = self.vector_quantizer(X)

        if hasattr(self, "tf_proj"):
            X = self.tf_proj(X)  # ( b, D2, t )

        if hasattr(self, "pos_enc"):
            X = self.pos_enc(X, transpose=True)

        X = self.blocks(X)  # ( b, D2, t )

        if self.vq == "middle2":
            X, vq_loss, perplexity = self.vector_quantizer(X)

        X = F.gelu(self.conv_final(X))

        if self.vq == "end":
            X, vq_loss, perplexity = self.vector_quantizer(X)

        if self.temporal_aggregation is not None:
            X = self.temporal_aggregation(X)

        ret_dict = {"Z_clip": self.clip_head(X), "Z_mse": self.mse_head(X)}

        if self.vae:
            Z_sample, q = self._reparameterize(X)
            ret_dict.update({"Z_sample": Z_sample, "q": q})

        if self.vq is not None:
            ret_dict.update({"vq_loss": vq_loss, "perplexity": perplexity})

        if hasattr(self, "dann_head"):
            subject_pred = self.dann_head(X)
            adv_loss = F.cross_entropy(subject_pred, subject_idxs.to(X.device))

            ret_dict.update({"adv_loss": adv_loss})

        return ret_dict

    def _reparameterize(self, X: torch.Tensor):
        if self.vae in ["vmf", "ps"]:
            mu, kappa = torch.tensor_split(self.q_head(X), [-1], dim=-1)
            # ( b, zdim ), ( b, 1 )

            mu = mu / mu.norm(dim=-1, keepdim=True)
            # + 1 to prevent collapsing
            kappa = F.softplus(kappa) + 1

            if self.vae == "vmf":
                q = VonMisesFisher(loc=mu, scale=kappa)
                Z_sample = q.rsample(torch.Size([self.sample_l]))

            elif self.vae == "ps":
                q = PowerSpherical(loc=mu, scale=rearrange(kappa, "b 1 -> (b 1)"))
                Z_sample = q.rsample((self.sample_l,))  # ( l, b, d )

        elif self.vae == "normal":
            mu, var = torch.tensor_split(self.q_head(X), 2, dim=-1)
            # ( b, zdim ), ( b, zdim )
            var = F.softplus(var)

            q = torch.distributions.Normal(loc=mu, scale=var)
            Z_sample = q.rsample((self.sample_l,))  # ( l, b, d )

        return Z_sample, q
