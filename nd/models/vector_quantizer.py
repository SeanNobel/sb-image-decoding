import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from termcolor import cprint

from vqtorch.nn import (
    VectorQuant,
    AffineTransform,
    GroupVectorQuant,
    ResidualVectorQuant,
)

from vector_quantize_pytorch import (
    ResidualVQ,
    GroupedResidualVQ,
    FSQ,
    LFQ,
    ResidualLFQ,
    GroupedResidualLFQ,
    ResidualFSQ,
    GroupedResidualFSQ,
)


def get_vector_quantizer(args, dim: int) -> nn.Module:
    """_summary_
    Args:
        args (_type_): _description_
        dim (int): Embedding dimension.
    Returns:
        nn.Module: _description_
    """
    if args.vq_type == "original":
        return VectorQuantizer(
            num_embeds=args.vq_num_embeds,
            embed_dim=dim,
            affine_lr=args.vq_affine_lr,
            use_ema=args.vq_use_ema,
            alpha=args.vq_alpha,
            beta=args.vq_beta,
            gamma=args.vq_gamma,
            epsilon=args.vq_epsilon,
        )
    elif args.vq_type.startswith("vqt_"):  # vqtorch
        vq_args = {
            "feature_size": dim,
            "num_codes": args.vq_num_embeds,
            "beta": 1 - args.vq_beta,
            "kmeans_init": args.vq_kmeans_init,
            "norm": args.vq_norm,
            "cb_norm": args.vq_cb_norm,
            "affine_lr": args.vq_affine_lr,
            "sync_nu": args.vq_sync_nu,
            "replace_freq": args.vq_replace_freq,
            "dim": 1,
        }

        if args.vq_alternate:
            vq_args.update(
                {
                    "beta": 1.0,  # Overwrite beta
                    "inplace_optimizer": lambda *args, **kwargs: torch.optim.SGD(
                        *args, **kwargs, lr=50.0, momentum=0.9
                    ),
                }
            )

        if args.vq_type.endswith("_affine"):
            vq = VectorQuant(**vq_args)
        elif args.vq_type.endswith("_group"):
            vq = GroupVectorQuant(groups=args.vq_groups, share=args.vq_share, **vq_args)
        elif args.vq_type.endswith("_residual"):
            vq = ResidualVectorQuant(groups=args.vq_groups, share=args.vq_share, **vq_args)  # fmt: skip
        else:
            raise NotImplementedError(f"vq_type {args.vq_type} not implemented.")

        return VQtorchWrapper(vq, args.vq_alpha)

    elif args.vq_type.startswith("lu_"):  # lucidrains
        vq_args = {}

        if "fsq" in args.vq_type:
            vq_args.update({"levels": [8, 5, 5, 5]})
        elif "lfq" in args.vq_type:
            pass
        else:
            vq_args.update({"dim": dim, "codebook_size": args.vq_num_embeds})

        if "res" in args.vq_type:
            vq_args.update({"num_quantizers": args.vq_num_quantizers})

        if "group" in args.vq_type:
            vq_args.update({"groups": args.vq_groups})

        if args.vq_type.endswith("_residual"):
            vq = ResidualVQ(**vq_args)
        elif args.vq_type.endswith("_groupres"):
            vq = GroupedResidualVQ(**vq_args)
        elif args.vq_type.endswith("_fsq"):
            vq = FSQ(**vq_args)
        elif args.vq_type.endswith("_lfq"):
            vq = LFQ(**vq_args)
        # elif args.vq_type.endswith("_resfsq"):
        #     vq = ResidualFSQ(**vq_args)
        # elif args.vq_type.endswith("_reslfq"):
        #     vq = ResidualLFQ(**vq_args)
        # elif args.vq_type.endswith("_groupresfsq"):
        #     vq = GroupedResidualFSQ(dim=args.vq_groups * 4, **vq_args)
        # elif args.vq_type.endswith("_groupreslfq"):
        #     vq = GroupedResidualLFQ(**vq_args)
        else:
            raise NotImplementedError(f"vq_type {args.vq_type} not implemented.")

        return LucidrainsWrapper(vq, args.vq_alpha, args.vq_groups)


class VQtorchWrapper(nn.Module):
    def __init__(self, vq: VectorQuant, alpha: float) -> None:
        super().__init__()

        self.vq = vq
        self.alpha = alpha

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor]:
        z_q, vq_returns = self.vq(z_e)

        return z_q, self.alpha * vq_returns["loss"], vq_returns["perplexity"]


class LucidrainsWrapper(nn.Module):
    def __init__(self, vq: ResidualVQ, alpha: float, codebook_size: int) -> None:
        super().__init__()

        self.vq = vq
        self.alpha = alpha
        self.codebook_size = codebook_size

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor]:
        """_summary_
        Args:
            z_e ( b, f, t ): _description_
        Returns:
            Tuple[torch.Tensor]: _description_
        """
        z_q, indices, loss = self.vq(z_e.permute(0, 2, 1).contiguous())

        z_q = z_q.permute(0, 2, 1).contiguous()

        # FIXME: It's unsure whether taking loss of the last layer is correct.
        loss = loss.squeeze()
        if loss.ndim > 0:
            loss = loss[-1]

        # FIXME: indices returned has the same shape as q in vqtorch but not sure if I can calculate perplexity in the same way.
        # !!!! Also if I calculate perplexity this way, CUDA device-side assertion error occurs !!!!
        # e_mean = F.one_hot(indices, num_classes=self.codebook_size).view(-1, self.codebook_size).float().mean(0)  # fmt: skip
        # perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        # TODO: Calculate perplexity.
        perplexity = torch.tensor(0.0)

        return z_q, self.alpha * loss, perplexity


class VectorQuantizer(nn.Module):
    def __init__(
        self,
        num_embeds: int,
        embed_dim: int,
        affine_lr: float = 0.0,
        use_ema: bool = False,
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

        if affine_lr > 0:
            assert not use_ema, "EMA is not compatible with affine transform."

            self.affine_transform = AffineTransform(
                self.embed_dim,
                use_running_statistics=False,
                lr_scale=affine_lr,
                num_groups=1,
            )

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

        if hasattr(self, "affine_transform"):
            self.affine_transform.update_running_statistics(z_e, self.embedding.weight)
            codebook = self.affine_transform(self.embedding.weight)
        else:
            codebook = self.embedding.weight

        z_e_flat = z_e.view(-1, self.embed_dim)  # ( b * t', embed_dim )

        # L2 distances
        distances = (
            torch.sum(z_e_flat**2, dim=1, keepdim=True)
            + torch.sum(codebook**2, dim=1)
            - 2 * torch.matmul(z_e_flat, codebook.T)
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

        z_q = torch.matmul(encodings, codebook).view(z_e_shape)
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
            vq_loss = self.beta * e_latent_loss
        else:
            q_latent_loss = F.mse_loss(z_q, z_e.detach())
            vq_loss = (1 - self.beta) * q_latent_loss + self.beta * e_latent_loss

        # Straight-through estimator
        z_q = z_e + (z_q - z_e).detach()

        z_q = z_q.permute(0, 2, 1).contiguous()  # ( b, embed_dim, t' )

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return z_q, self.alpha * vq_loss, perplexity
