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

from fairseq.fairseq.modules.conformer_layer import ConformerEncoderLayer
from fairseq.fairseq.modules import RelPositionalEncoding

from nd.models.vector_quantizer import get_vector_quantizer, AggregatedVectorQuantizer
from nd.models.transformer import (
    SelfAttention,
    FeedForward,
    PreNorm,
    Residual,
    PositionalEncoding,  # positional_encoding,
    relative_positional_encoding,
)
from nd.models.dann import DANN
from nd.models.utils import DropBlock1D
from nd.models.subject_sa import SubjectBlockSA, SubjectBlockConvDynamic
from nd.utils.layout import ch_locations_2d, DynamicChanLoc2d
from nd.utils.train_utils import conv_output_size


def is_in(s: Optional[str], _s: str) -> bool:
    if s is None:
        return False
    else:
        return _s in s


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


class TransformerBlock(nn.Module):
    def __init__(
        self,
        k: int,
        D1: int,
        D2: int,
        n_heads: int,
        block_size: int,
        pos_enc: Optional[str] = None,
        p_drop: float = 0.1,
    ):
        super().__init__()

        emb_dim = D2

        if k == 0:
            self.proj = nn.Linear(D1, emb_dim)

        if is_in(pos_enc, "abs"):
            self.pos_enc = PositionalEncoding(
                block_size, emb_dim, pos_enc.split("_")[0]
            )
        elif pos_enc == "sine_rel":
            self.register_buffer(
                "pos_enc_k", relative_positional_encoding(block_size, self.d_qk)
            )
            # ( t, t, d_qk )
            self.register_buffer(
                "pos_enc_v", relative_positional_encoding(block_size, self.d_v)
            )
            # ( t, t, d_v )
        else:
            assert pos_enc is None, f"Unknown positional encoding type: {pos_enc}"

        self.attn = Residual(
            PreNorm(SelfAttention(emb_dim, n_heads, block_size), emb_dim)
        )

        self.mlp = FeedForward(emb_dim, ff_pdrop=p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X ( b, d, t ): _description_
        Returns:
            X ( b, d, t ): _description_
        """
        X = X.permute(0, 2, 1)

        if hasattr(self, "proj"):
            X = self.proj(X)

        if hasattr(self, "pos_enc"):
            X = self.pos_enc(X)

        if hasattr(self, "pos_enc_k"):
            X = self.attn(X, self.pos_enc_k, self.pos_enc_v)
        else:
            X = self.attn(X)

        X = self.mlp(X)

        return X.permute(0, 2, 1)


class ConformerBlock(nn.Module):
    def __init__(
        self,
        k: int,
        D1: int,
        D2: int,
        n_heads: int,
        depthwise_ksize: int,
        activation_fn: str = "swish",
        attn_type: Optional[str] = "espnet",
        pos_enc_type: str = "abs",
        temporal_dim: Optional[int] = None,
        p_drop: float = 0.1,
    ):
        super().__init__()

        emb_dim = D2

        if k == 0:
            self.proj = nn.Conv1d(D1, emb_dim, kernel_size=1)

        if pos_enc_type == "rel_pos":
            self.rel_pos = RelPositionalEncoding(temporal_dim, emb_dim)

        self.attn = ConformerEncoderLayer(
            embed_dim=emb_dim,
            ffn_embed_dim=emb_dim * 4,
            attention_heads=n_heads,
            dropout=p_drop,
            use_fp16=False,
            depthwise_conv_kernel_size=depthwise_ksize,
            activation_fn=activation_fn,
            attn_type=attn_type,
            pos_enc_type=pos_enc_type,
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, c, t ): _description_
        """
        if hasattr(self, "proj"):
            X = self.proj(X)

        X = rearrange(X, "b c t -> t b c")

        if hasattr(self, "rel_pos"):
            positions = self.rel_pos(X)
        else:
            positions = None

        X, _ = self.attn(X, encoder_padding_mask=None, position_emb=positions)

        return rearrange(X, "t b c -> b c t")


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
        args,
        temporal_dim: int,
        embed_dim: Optional[int] = None,
        multiplier: int = 1,
    ) -> None:
        super().__init__()

        if embed_dim is None:
            embed_dim = args.D3

        # FIXME: Other than affine may not be working

        if args.temporal_aggregation == "original":
            self.layers = OriginalAggregator(args, temporal_dim, multiplier)
        else:
            """Modified from: https://ai.meta.com/static-resource/image-decoding"""
            self.layers = nn.Sequential()

            # NOTE: conv_final corresponds to linear projection in the paper as long as the kernel size and stride are 1
            self.layers.add_module(
                "linear_projection",
                nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=1),
            )

            if args.temporal_aggregation == "affine":
                self.layers.add_module("temporal_agg", nn.Linear(temporal_dim, multiplier))  # fmt: skip
            elif args.temporal_aggregation == "pool":
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
    def __init__(self, in_dim: int, out_dim: int, t: int) -> None:
        super().__init__()

        self.net = nn.Sequential(
            Rearrange("b d t -> b (d t)"),
            nn.LayerNorm([in_dim * t]),
            nn.GELU(),
            nn.Linear(in_dim * t, out_dim * t),
            Rearrange("b (d t) -> b d t", d=out_dim),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X)


class BrainEncoder(nn.Module):
    def __init__(
        self, args, subjects: Union[int, List[str]], unknown_subject: bool = False
    ) -> None:
        super().__init__()

        # Parameters
        self.vq = args.vq
        self.ignore_subjects = args.ignore_subjects or args.dann
        self.unknown_subject = unknown_subject

        D1, D2, D3, F = args.D1, args.D2, args.D3, args.F
        init_temporal_dim: int = int(args.seq_len * args.brain_resample_sfreq)
        num_clip_tokens: int = args.num_clip_tokens
        num_blocks: int = args.num_blocks
        num_subjects: int = subjects if isinstance(subjects, int) else len(subjects)
        layout: Union[ch_locations_2d, DynamicChanLoc2d] = eval(args.layout)
        spatial_attention: bool = args.spatial_attention
        pos_enc: str = args.pos_enc
        blocks: Union[str, List[str]] = args.blocks
        conv_block_ksize: int = args.conv_block_ksize
        depthwise_ksize: int = args.depthwise_ksize
        downsample: Union[bool, List[bool]] = args.downsample
        temporal_agg: str = args.temporal_aggregation
        transformer_heads: int = args.transformer_heads
        p_drop: float = args.p_drop
        drop_mode: str = args.drop_mode
        final_ksize: int = args.final_ksize
        final_stride: int = args.final_stride
        vq_aggregated: bool = args.vq_aggregated
        dann: bool = args.dann
        dann_scale: float = args.dann_scale

        if isinstance(blocks, str):
            blocks = [blocks] * num_blocks
        else:
            if len(blocks) != num_blocks:
                num_blocks = len(blocks)
                cprint(f"Updating num_blocks to {num_blocks}.", "yellow")

        if isinstance(downsample, bool):
            downsample = [downsample] * num_blocks
        else:
            assert len(downsample) == num_blocks, "downsample and num_blocks should have the same length."  # fmt: skip

        if layout == ch_locations_2d:
            self.subject_block = SubjectBlock(
                args, num_subjects if not self.ignore_subjects else 1, layout(args)
            )

        elif layout == DynamicChanLoc2d:
            assert isinstance(subjects, list), "subjects should be list of str when using DynamicChanLoc2d."  # fmt: skip
            assert not self.ignore_subjects, "Cannot ignore subjects when channel locations are different among them."  # fmt: skip

            if spatial_attention:
                self.subject_block = SubjectBlockSA(args, len(subjects), layout(args, subjects))  # fmt: skip
            else:
                self.subject_block = SubjectBlockConvDynamic(args, len(subjects), layout(args, subjects))  # fmt: skip
        else:
            raise TypeError

        block_args = {"D1": D1, "D2": D2, "p_drop": p_drop}
        self.blocks = nn.Sequential()

        for k, block in enumerate(blocks):
            if block == "dilated_conv":
                cprint(f"Block{k}: dilated_conv", "magenta")

                self.blocks.add_module(
                    f"block{k}",
                    ConvBlock(
                        k,
                        ksize=conv_block_ksize,
                        drop_mode=drop_mode,
                        **block_args,
                    ),
                )
            elif block == "inception":
                cprint(f"Block{k}: inception", "magenta")

                self.blocks.add_module(
                    f"block{k}",
                    Inception1DBlock(k, drop_mode=drop_mode, **block_args),
                )
            elif block == "transformer":
                pe = pos_enc if k == blocks.index("transformer") else None
                cprint(f"Block{k}: transformer with pos_enc {pe}", "magenta")

                self.blocks.add_module(
                    f"block{k}",
                    TransformerBlock(
                        k,
                        n_heads=transformer_heads,
                        # TODO: TransformerBlocks after downsampling with ConvBlocks
                        block_size=init_temporal_dim,
                        pos_enc=pe,
                        **block_args,
                    ),
                )
            elif block == "conformer":
                pe = {"sine_abs": "abs", "sine_rel": "rel_pos", "rotary": "rope"}[pos_enc]  # fmt: skip
                cprint(f"Block{k}: conformer", "magenta")

                self.blocks.add_module(
                    f"block{k}",
                    ConformerBlock(
                        k,
                        n_heads=transformer_heads,
                        depthwise_ksize=depthwise_ksize,
                        pos_enc_type=pe,
                        temporal_dim=init_temporal_dim,
                        **block_args,
                    ),
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
            init_temporal_dim,
            ksize=final_ksize,
            stride=final_stride,
            repetition=3 if temporal_agg == "original" else 1,
            downsample=sum(downsample),
        )

        if temporal_agg is not None:
            self.temporal_aggregation = TemporalAggregation(
                args, temporal_dim, multiplier=num_clip_tokens
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

        if dann:
            self.dann_head = DANN(in_dim=D3, num_domains=num_subjects, scale=dann_scale)  # fmt: skip

        self.clip_head = MLPHead(in_dim=D3, out_dim=F, t=num_clip_tokens)
        self.mse_head = MLPHead(in_dim=D3, out_dim=F, t=num_clip_tokens)

    def forward(
        self, X: torch.Tensor, subject_idxs: Optional[torch.Tensor]
    ) -> torch.Tensor:
        assert self.unknown_subject or subject_idxs is not None, "You need to provide subject_idxs when it's not unknown subject."  # fmt: skip

        X = self.subject_block(
            X, subject_idxs if not self.ignore_subjects else torch.zeros_like(subject_idxs)  # fmt: skip
        )

        if self.vq == "middle1":
            X, vq_loss, perplexity = self.vector_quantizer(X)

        X = self.blocks(X)

        if self.vq == "middle2":
            X, vq_loss, perplexity = self.vector_quantizer(X)

        X = F.gelu(self.conv_final(X))

        if self.vq == "end":
            X, vq_loss, perplexity = self.vector_quantizer(X)

        if self.temporal_aggregation is not None:
            X = self.temporal_aggregation(X)

        Z_clip = self.clip_head(X)
        Z_mse = self.mse_head(X)

        ret_dict = {"Z_clip": Z_clip, "Z_mse": Z_mse}

        if self.vq is not None:
            ret_dict.update({"vq_loss": vq_loss, "perplexity": perplexity})

        if hasattr(self, "dann_head"):
            subject_pred = self.dann_head(X)
            adv_loss = F.cross_entropy(subject_pred, subject_idxs.to(X.device))

            ret_dict.update({"adv_loss": adv_loss})

        return ret_dict

    def encode(
        self,
        X: torch.Tensor,
        subject_idxs: Optional[torch.Tensor],
        return_mse: bool = True,
        normalize: bool = True,
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

        _, d, t = Z.shape
        Z = rearrange(Z, "b d t -> b (d t)")

        if normalize:
            Z /= Z.norm(dim=-1, keepdim=True)

        if stats is not None:
            # Inverse normalization
            Z = (Z - Z.mean()) / Z.std()
            mean, std = stats
            Z = Z * std + mean

        Z = rearrange(Z, "b (d t) -> b d t", d=d)

        if swap_dims:
            Z = rearrange(Z, "b d t -> b t d")

        if device is not None:
            Z = Z.to(orig_device)

        if single:
            Z = Z.squeeze(0)

        return Z
