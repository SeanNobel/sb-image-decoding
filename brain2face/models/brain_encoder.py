import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Optional, Union, Callable, List, Tuple
from termcolor import cprint

from brain2face.utils.layout import ch_locations_2d, DynamicChanLoc2d
from brain2face.utils.train_utils import conv_output_size


class SpatialAttention(nn.Module):
    """Same as SpatialAttentionVer2, but a little more concise"""

    def __init__(self, args, loc: np.ndarray):
        super(SpatialAttention, self).__init__()

        # vectorize of k's and l's
        a = []
        for k in range(args.K):
            for l in range(args.K):
                a.append((k, l))
        a = torch.tensor(a)
        k, l = a[:, 0], a[:, 1]

        # vectorize x- and y-positions of the sensors
        x, y = loc[:, 0], loc[:, 1]

        # make a complex-valued parameter, reshape k,l into one dimension
        self.z = nn.Parameter(torch.rand(size=(args.D1, args.K**2), dtype=torch.cfloat))

        # NOTE: pre-compute the values of cos and sin (they depend on k, l, x and y which repeat)
        phi = (
            2 * torch.pi * (torch.einsum("k,x->kx", k, x) + torch.einsum("l,y->ly", l, y))
        )  # torch.Size([1024, 60]))
        self.register_buffer("cos", torch.cos(phi))
        self.register_buffer("sin", torch.sin(phi))

        # self.spatial_dropout = SpatialDropoutX(args)
        self.spatial_dropout = SpatialDropout(loc, args.d_drop)

    def forward(self, X):
        """X: (batch_size, num_channels, T)"""

        # NOTE: do hadamard product and and sum over l and m (i.e. m, which is l X m)
        re = torch.einsum("jm, me -> je", self.z.real, self.cos)  # torch.Size([270, 60])
        im = torch.einsum("jm, me -> je", self.z.imag, self.sin)
        a = re + im
        # essentially (unnormalized) weights with which to mix input channels into ouput channels
        # ( D1, num_channels )

        # NOTE: to get the softmax spatial attention weights over input electrodes,
        # we don't compute exp, etc (as in the eq. 5), we take softmax instead:
        SA_wts = F.softmax(a, dim=-1)  # each row sums to 1
        # ( D1, num_channels )

        # NOTE: drop some channels within a d_drop of the sampled channel
        dropped_X = self.spatial_dropout(X)

        # NOTE: each output is a diff weighted sum over each input channel
        return torch.einsum("oi,bit->bot", SA_wts, dropped_X)


class SpatialDropout(nn.Module):
    """Using same drop center for all samples in batch"""

    def __init__(self, loc, d_drop):
        super(SpatialDropout, self).__init__()
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
        super(SubjectSpatialAttention, self).__init__()

        self.num_channels = loc.shape[0]

        self.spatial_attention = SpatialAttention(args, loc)

        self.conv = nn.Conv1d(
            in_channels=args.D1,
            out_channels=args.D1,
            kernel_size=1,
            stride=1,
            bias=args.biases.conv_subj_sa,
        )
        # self.conv2 = nn.Conv1d(
        #     in_channels=args.D1, out_channels=args.D1, kernel_size=1, stride=1, bias=False
        # )

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
        super(SubjectBlock, self).__init__()

        self.num_subjects = num_subjects
        self.D1 = args.D1
        self.K = args.K
        self.spatial_attention = SpatialAttention(args, loc)
        self.conv = nn.Conv1d(
            in_channels=self.D1, out_channels=self.D1, kernel_size=1, stride=1
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
        super(SubjectBlockConvDynamic, self).__init__()

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
        )  # ( B, 270, 256 )

        return X


class SubjectBlockSA(nn.Module):
    """Applies Spatial Attention to each subject separately"""

    def __init__(self, args, num_subjects: int, layouts: DynamicChanLoc2d) -> None:
        super(SubjectBlockSA, self).__init__()

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
            )  # ( B, 270, 256 )

        X = self.conv(X)

        return X


class ConvBlock(nn.Module):
    def __init__(self, k: int, D1: int, D2: int, ksize: int = 3) -> None:
        super(ConvBlock, self).__init__()

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

        return X  # ( B, 320, 256 )


class BrainEncoder(nn.Module):
    def __init__(
        self,
        args,
        subject_names: List[str],
        layout: Union[Callable, DynamicChanLoc2d] = ch_locations_2d,
        unknown_subject: bool = False,
    ) -> None:
        super(BrainEncoder, self).__init__()

        self.D1 = args.D1
        self.D2 = args.D2
        self.F = args.F
        self.K = args.K

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

        self.conv_blocks = nn.Sequential()
        for k in range(5):
            self.conv_blocks.add_module(
                f"conv{k}", ConvBlock(k, self.D1, self.D2, args.ksizes.conv_block)
            )

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

    def forward(
        self, X: torch.Tensor, subject_idxs: Optional[torch.Tensor]
    ) -> torch.Tensor:
        assert (
            self.unknown_subject or subject_idxs is not None
        ), "You need to provide subject_idxs when it's not unknown subject."

        X = self.subject_block(X, subject_idxs)
        X = self.conv_blocks(X)
        X = F.gelu(self.conv_final1(X))
        X = F.gelu(self.conv_final2(X))
        return X
    
    
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeds: int, embed_dim: int, beta: float) -> None:
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
        self.beta = beta

        self.embedding = nn.Embedding(self.num_embeds, self.embed_dim)
        self.embedding.weight.data.uniform_(-1 / self.num_embeds, 1 / self.num_embeds)

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        Args:
            z_e ( b, embed_dim, t' ): 
        Returns:
            loss : torch.Tensor (, )
                目的関数の再構成誤差以外の部分．
            z_q : torch.Tensor ( b, embed_dim, h', w' )
                離散化された潜在変数．
        """
        z_e = z_e.permute(0, 2, 1).contiguous() # ( b, t', embed_dim )
        z_e_shape = z_e.shape

        z_e_flat = z_e.view(-1, self.embed_dim) # ( b * t', embed_dim )

        # L2 distances
        distances = (
            torch.sum(z_e_flat**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_e_flat, self.embedding.weight.T)
        )
        # ( b * t', num_embeds )

        # 最も距離の近いembedding vectorのインデックス
        encoding_indices = torch.argmin(distances, dim=1)
        # ( b * t', )

        encodings = F.one_hot(encoding_indices, num_classes=self.num_embeds).to(torch.float32)
        # ( b * t', num_embeds )

        z_q = torch.matmul(encodings, self.embedding.weight)
        # ( b * t', embed_dim )
        z_q = z_q.view(z_e_shape) # ( b, t', embed_dim )

        e_latent_loss = F.mse_loss(z_q.detach(), z_e)
        q_latent_loss = F.mse_loss(z_q, z_e.detach())
        # Regularization loss
        reg_loss = q_latent_loss + self.beta * e_latent_loss

        # Straight-through estimator
        z_q = z_e + (z_q - z_e).detach()

        z_q = z_q.permute(0, 2, 1).contiguous() # ( b, embed_dim, t' )

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return z_q, reg_loss, perplexity
    
    
class BrainEncoderVQ(nn.Module):
    def __init__(
        self,
        args,
        subject_names: List[str] = None,
        layout: Union[Callable, DynamicChanLoc2d] = ch_locations_2d,
        unknown_subject: bool = False,
    ) -> None:
        super().__init__()
        
        self.brain_encoder = BrainEncoder(
            args,
            subject_names=subject_names,
            layout=layout,
            unknown_subject=unknown_subject,
        )
        
        self.vector_quantizer = VectorQuantizer(
            num_embeds=args.vq.num_embeds,
            embed_dim=args.F,
            beta=args.vq.beta,
        )
        
    def forward(
        self, X: torch.Tensor, subject_idxs: Optional[torch.Tensor]
    ) -> torch.Tensor:
        X = self.brain_encoder(X, subject_idxs) # ( b, F, t' )
        
        X_q, reg_loss, perplexity = self.vector_quantizer(X) # ( b, F, t' )
        
        return X_q, reg_loss, perplexity


class BrainEncoderReduceTime(nn.Module):
    def __init__(
        self,
        args,
        subject_names: List[str] = None,
        layout: Union[Callable, DynamicChanLoc2d] = ch_locations_2d,
        vq: bool = False,
        unknown_subject: bool = False,
        time_multiplier: int = 1,
    ) -> None:
        """
        Args:
            time_multiplier:
        """
        super(BrainEncoderReduceTime, self).__init__()
        
        self.vq = vq
        if vq:
            self.brain_encoder = BrainEncoderVQ(
                args,
                subject_names=subject_names,
                layout=layout,
                unknown_subject=unknown_subject,
            )
        else:
            self.brain_encoder = BrainEncoder(
                args,
                subject_names=subject_names,
                layout=layout,
                unknown_subject=unknown_subject,
            )

        self.conv1 = nn.Conv1d(
            in_channels=args.F,
            out_channels=args.F,
            kernel_size=args.final_ksize,
            stride=args.final_stride,
        )
        self.conv2 = nn.Conv1d(
            in_channels=args.F,
            out_channels=args.F,
            kernel_size=args.final_ksize,
            stride=args.final_stride,
        )

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(
            in_features=args.F
            * (
                conv_output_size(
                    int(args.seq_len * args.brain_resample_sfreq),
                    ksize=args.final_ksize,
                    stride=args.final_stride,
                    repetition=4,
                )
            ),
            out_features=args.F * time_multiplier,
            bias=args.biases.linear_reduc_time,
        )
        self.activation = args.head_activation

    def forward(
        self, X: torch.Tensor, subject_idxs: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if self.vq:
            X, reg_loss, perplexity = self.brain_encoder(X, subject_idxs)
        else:
            X = self.brain_encoder(X, subject_idxs)

        X = self.conv1(X)
        X = self.conv2(X)
        X = self.linear(self.flatten(X))

        if self.activation:
            X = F.gelu(X)

        if self.vq and self.training:
            return X, reg_loss, perplexity
        else:
            return X