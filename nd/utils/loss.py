import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import kl_divergence
from einops import rearrange
import matplotlib.pyplot as plt
from tqdm import tqdm
from termcolor import cprint
from typing import Optional, Union, List

from nd.utils.power_spherical import PowerSpherical
from nd.utils.power_spherical import HypersphericalUniform as HypersphericalUniformPS
from nd.utils.von_mises_fisher import VonMisesFisher
from nd.utils.von_mises_fisher import HypersphericalUniform as HypersphericalUniformVMF


def calc_similarity(
    Z: torch.Tensor, Y: torch.Tensor, sequential: bool, pbar: bool = True
) -> torch.Tensor:
    batch_size, _size = len(Z), len(Y)

    Z = Z.contiguous().view(batch_size, -1)
    Y = Y.contiguous().view(_size, -1)

    # NOTE: avoid CUDA out of memory like this
    if sequential:
        Z = Z / Z.norm(dim=-1, keepdim=True)
        Y = Y / Y.norm(dim=-1, keepdim=True)

        similarity = torch.empty(batch_size, _size).to(device=Z.device)

        if pbar:
            pbar = tqdm(total=batch_size, desc="Similarity matrix of test size")

        for i in range(batch_size):
            # similarity[i] = (Z[i] @ Y.T) / torch.clamp((Z[i].norm() * Y.norm(dim=1)), min=1e-8)
            similarity[i] = Z[i] @ Y.T

            if pbar:
                pbar.update(1)
    else:
        Z = rearrange(Z, "b f -> b 1 f")
        Y = rearrange(Y, "b f -> 1 b f")
        similarity = F.cosine_similarity(Y, Z, dim=-1)

    torch.cuda.empty_cache()

    return similarity


def top_k_accuracy(k: int, similarity: torch.Tensor, labels: torch.Tensor):
    """_summary_

    Args:
        k (int): _description_
        similarity ( b, 2400 ): _description_
        labels ( b, ): _description_

    Returns:
        _type_: _description_
    """
    # if k == 1:
    #     return (similarity.argmax(axis=1) == diags).to(torch.float).mean().item()  # fmt: skip
    # else:
    return np.mean(
        [
            label in row
            for row, label in zip(
                torch.topk(similarity, k, dim=1, largest=True)[1], labels
            )
        ]
    )


def off_diag(mat: torch.Tensor) -> torch.Tensor:
    """Returns off-diagonal elements of a square matrix.
    Args:
        mat ( b, b ): _description_
    Returns:
        off_diag ( b * (b - 1), ): _description_
    """
    b = mat.shape[0]

    return torch.cat(
        [torch.diag(mat, i) for i in range(1, b)]
        + [torch.diag(mat, -i) for i in range(1, b)]
    )


def build_clip(args, dataset, device):
    if args.loss == "clip":
        loss_func = CLIPLoss(args).to(device)
    elif args.loss == "variationalclip":
        loss_func = VariationalCLIPLoss(args, beta=args.kl_beta).to(device)
    elif args.loss == "klclip":
        loss_func = KLRegCLIPLoss(args, alpha=args.klclip_alpha).to(device)
    elif args.loss == "orclip":
        loss_func = OrthoRegCLIPLoss(args, alpha=args.orclip_alpha).to(device)
    elif args.loss == "leclip":
        loss_func = LargeEntropyCLIPLoss(args, alpha=args.leclip_alpha).to(device)
    elif args.loss == "adaptiveclip":
        loss_func = AdaptiveCLIPLoss(args).to(device)
    elif args.loss == "apclip":
        loss_func = AdditionalPositivesCLIPLoss(args).to(device)
    elif args.loss == "cosfaceclip":
        loss_func = CosFaceCLIPLoss(args).to(device)
    elif args.loss == "arcfaceclip":
        loss_func = ArcFaceCLIPLoss(args).to(device)
    elif args.loss == "amclip":
        loss_func = AdaptiveMarginCLIPLoss(args).to(device)
    elif args.loss == "circleclip":
        loss_func = CircleCLIPLoss(args).to(device)
    elif args.loss == "geomclip":
        loss_func = GeometricCLIPLoss(args).to(device)
    elif args.loss == "nnclip":
        loss_func = NearestNeighborCLIPLoss(args).to(device)
    elif args.loss == "clipclasscosface":
        loss_func = CLIPWithClassCosFaceLoss(
            args,
            dataset.num_categories,
            dataset.num_high_categories if args.use_high_categories else None,
        ).to(device)
    elif args.loss == "clipclasscircle":
        loss_func = CLIPWithClassCircleLoss(
            args,
            dataset.num_categories,
            dataset.num_high_categories if args.use_high_categories else None,
        ).to(device)
    else:
        raise ValueError(f"Invalid loss function: {args.loss}")

    return loss_func


class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()

        self.reduction = reduction

    def forward(self, logits, targets):
        """I'm makeing my own BCE because torch BCE turned out not to work as I expected.
        Accepts soft targets.
        Args:
            logits ( b, b ): _description_
            targets ( b, b ) or ( b, ): _description_
        Returns:
            _type_: _description_
        """
        probs = F.softmax(logits, dim=1)

        if targets.ndim == 1:
            targets = F.one_hot(targets, num_classes=logits.shape[1]).to(torch.float32)
        # FIXME
        elif torch.equal(targets, torch.eye(targets.shape[0], device=targets.device)):
            pass
        else:
            targets = F.softmax(targets, dim=1)

        bce = F.binary_cross_entropy(probs, targets, reduction="none").sum(dim=1)

        if self.reduction == "mean":
            return bce.mean()
        elif self.reduction == "sum":
            return bce.sum()
        else:
            return bce


class VariationalLowerBound(nn.Module):
    def __init__(self, args, device: str) -> None:
        super().__init__()

        # FIXME: They should be the same but haven't checked yet.
        if args.vae == "ps":
            self.p = HypersphericalUniformPS(dim=args.vae_zdim - 1)
        elif args.vae == "vmf":
            self.p = HypersphericalUniformVMF(dim=args.vae_zdim - 1, device=device)
        elif args.vae == "normal":
            self.p = torch.distributions.Normal(loc=0.0, scale=1.0)
        else:
            raise ValueError(f"Invalid hypersphere: {args.hypersphere}")

        self.beta = args.kl_beta
        self.reduction = args.reduction

    def forward(
        self,
        X_recon: torch.Tensor,
        X: torch.Tensor,
        q: Union[PowerSpherical, List[PowerSpherical]],
    ) -> torch.Tensor:
        """_summary_
        Args:
            X_recon ( l, b, c, t ): _description_
            X ( b, c, t ): _description_
        Returns:
            torch.Tensor: _description_
        """
        X_recon = rearrange(X_recon, "l b c t -> l b (c t)")
        X = rearrange(X, "b c t -> b (c t)")

        recon_loss = torch.stack(
            [F.mse_loss(Xr, X, reduction=self.reduction) for Xr in X_recon]
        ).mean()

        if isinstance(q, list):
            kl_loss = torch.cat([kl_divergence(q_, self.p) for q_ in q]).mean()
        else:
            kl_loss = kl_divergence(q, self.p).mean()

        return recon_loss + self.beta * kl_loss, recon_loss, kl_loss


class CLIPLoss(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.compute_similarity = nn.CosineSimilarity(dim=-1)

        if args.use_negative:
            self.ce = BinaryCrossEntropyLoss(reduction=args.reduction)
        else:
            self.ce = nn.CrossEntropyLoss(reduction=args.reduction)

        # Temperature (scaler)
        self.temp = nn.Parameter(torch.tensor([float(args.clip_temp_init)]))
        self.temp_min = args.clip_temp_min
        self.temp_max = args.clip_temp_max
        if not args.clip_temp_learn:
            self.temp.requires_grad = False

    def forward(self, x, y, fast=True, return_logits=False):
        batch_size = x.size(0)
        assert batch_size > 1, "Batch size must be greater than 1."
        targets = torch.arange(batch_size, requires_grad=False).long().to(device=x.device)  # fmt: skip

        if not fast:
            # less efficient way
            x_ = rearrange(x, "b f t -> 1 b (f t)")
            y_ = rearrange(y, "b f t -> b 1 (f t)")
            logits = self.compute_similarity(x_, y_)  # s
        else:
            # fast way
            x = x.reshape(batch_size, -1)
            y = y.reshape(batch_size, -1)

            # NOTE: scale the embeddings to unit norm
            x = x / x.norm(dim=-1, keepdim=True)
            y = y / y.norm(dim=-1, keepdim=True)

            # get dot products
            logits = torch.matmul(x, y.T)  # ( b, b )

        # FIXME: Probably exp is not needed, but keeping it for consistency.
        logits *= torch.exp(self.temp)

        # NOTE: as in https://arxiv.org/abs/2103.00020
        loss = (self.ce(logits, targets) + self.ce(logits.T, targets)) / 2

        if return_logits:
            return logits, loss
        else:
            return loss

    def clamp_params(self):
        if not (self.temp_min is None and self.temp_max is None):
            self.temp.data.clamp_(min=self.temp_min, max=self.temp_max)


class VariationalCLIPLoss(CLIPLoss):
    def __init__(self, args, beta: float) -> None:
        super().__init__(args)

        self.beta = beta
        self.p = HypersphericalUniformPS(dim=args.F)

    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        q: Union[PowerSpherical, List[PowerSpherical]],
    ) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, 1, d ): _description_
            Y ( b, 1, d ): _description_
        Returns:
            torch.Tensor: _description_
        """
        b = X.shape[0]
        targets = torch.eye(b, device=X.device, requires_grad=False)

        X = X.reshape(b, -1)
        Y = Y.reshape(b, -1)

        X = X / X.norm(dim=-1, keepdim=True)
        Y = Y / Y.norm(dim=-1, keepdim=True)

        similarity = torch.matmul(X, Y.T)

        similarity *= torch.exp(self.temp)
        clip_loss = (self.ce(similarity, targets) + self.ce(similarity.T, targets)) / 2

        if isinstance(q, list):
            reg_loss = torch.cat([kl_divergence(q_, self.p) for q_ in q]).mean()
        else:
            reg_loss = kl_divergence(q, self.p).mean()

        return clip_loss + self.beta * reg_loss


class OrthoRegCLIPLoss(CLIPLoss):
    """Regularizes with the orthogonality of the similarity matrix."""

    def __init__(self, args, alpha=0.1) -> None:
        super().__init__(args)

        self.alpha = alpha

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        b = X.shape[0]
        targets = torch.eye(b, device=X.device, requires_grad=False)

        X = X.reshape(b, -1)
        Y = Y.reshape(b, -1)

        X = X / X.norm(dim=-1, keepdim=True)
        Y = Y / Y.norm(dim=-1, keepdim=True)

        similarity = torch.matmul(X, Y.T)

        reg_loss = F.l1_loss(similarity, targets, reduction="none").mean()

        similarity *= torch.exp(self.temp)
        clip_loss = (self.ce(similarity, targets) + self.ce(similarity.T, targets)) / 2

        return clip_loss + self.alpha * reg_loss


class KLRegCLIPLoss(CLIPLoss):
    """Regularizes with the KL divergence of X and Y."""

    def __init__(self, args, alpha=0.1) -> None:
        super().__init__(args)

        self.alpha = alpha

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        b = X.shape[0]
        targets = torch.eye(b, device=X.device, requires_grad=False)

        X = X.reshape(b, -1)
        Y = Y.reshape(b, -1)

        reg_loss = F.kl_div(
            X.log_softmax(dim=0), Y.softmax(dim=0), reduction="batchmean"
        )

        X = X / X.norm(dim=-1, keepdim=True)
        Y = Y / Y.norm(dim=-1, keepdim=True)

        similarity = torch.matmul(X, Y.T)

        similarity *= torch.exp(self.temp)
        clip_loss = (self.ce(similarity, targets) + self.ce(similarity.T, targets)) / 2

        return clip_loss + self.alpha * reg_loss


class LargeEntropyCLIPLoss(CLIPLoss):
    def __init__(self, args, alpha=0.1) -> None:
        super().__init__(args)

        self.alpha = alpha

        self.impl_type = args.impl_type

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        b = X.shape[0]
        targets = torch.arange(b, requires_grad=False).long().to(X.device)

        X = X.reshape(b, -1)
        Y = Y.reshape(b, -1)

        X = X / X.norm(dim=-1, keepdim=True)
        Y = Y / Y.norm(dim=-1, keepdim=True)

        similarity = torch.matmul(X, Y.T)

        if self.impl_type == 0:
            entropy_loss = -(self._entropy(X) + self._entropy(Y))

        elif self.impl_type == 1:
            entropy_loss = -(self._perplexity(X) + self._perplexity(Y))

        elif self.impl_type == 2:
            entropy_loss = -(self._entropy(similarity) + self._entropy(similarity.T))

        elif self.impl_type == 3:
            entropy_loss = -(
                self._perplexity(similarity) + self._perplexity(similarity.T)
            )

        elif self.impl_type == 4:
            entropy_loss = -(
                self._angular_entropy(similarity) + self._angular_entropy(similarity.T)
            )
        else:
            raise NotImplementedError

        similarity *= torch.exp(self.temp)
        clip_loss = (self.ce(similarity, targets) + self.ce(similarity.T, targets)) / 2

        return clip_loss + self.alpha * entropy_loss

    @staticmethod
    def _entropy(x: torch.Tensor) -> torch.Tensor:
        avg_probs = torch.softmax(x, dim=-1).mean(dim=0)

        return -torch.sum(avg_probs * torch.log(avg_probs + 1e-7))

    @staticmethod
    def _perplexity(x: torch.Tensor) -> torch.Tensor:
        avg_probs = torch.softmax(x, dim=-1).mean(dim=0)

        return -torch.sum(avg_probs * torch.log(avg_probs + 1e-7), dim=-1).exp().sum()

    @staticmethod
    def _angular_entropy(cosine: torch.Tensor) -> torch.Tensor:
        angles = torch.acos(cosine.clamp(-1 + 1e-7, 1 - 1e-7))

        avg_probs = torch.softmax(angles, dim=-1).mean(dim=0)

        return -torch.sum(avg_probs * torch.log(avg_probs + 1e-7))


class AdditionalPositivesCLIPLoss(CLIPLoss):
    def __init__(self, args) -> None:
        super().__init__(args)

        self.threshold = args.positive_threshold

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        b = X.shape[0]

        X = X.reshape(b, -1)
        Y = Y.reshape(b, -1)

        X = X / X.norm(dim=-1, keepdim=True)
        Y = Y / Y.norm(dim=-1, keepdim=True)

        similarity = torch.matmul(X, Y.T)

        similarity *= torch.exp(self.temp)

        guidance = torch.matmul(Y, Y.T)
        targets = torch.where(guidance > self.threshold, 1.0, 0.0)

        return (self.ce(similarity, targets) + self.ce(similarity.T, targets)) / 2


class AdaptiveCLIPLoss(CLIPLoss):
    def __init__(self, args) -> None:
        super().__init__(args)
        assert args.use_negative, "Not using negatives makes this loss same as the original CLIP loss."  # fmt: skip

        self.margin_n = args.clip_margin_init

        self.impl_type = args.impl_type

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        b = X.shape[0]

        X = X.reshape(b, -1)
        Y = Y.reshape(b, -1)

        X = X / X.norm(dim=-1, keepdim=True)
        Y = Y / Y.norm(dim=-1, keepdim=True)

        guidance = torch.matmul(Y, Y.T)
        eye = torch.eye(b, device=X.device, requires_grad=False)

        if self.impl_type == 0:
            targets = torch.where(eye == 1.0, guidance, guidance - off_diag(guidance).mean())  # fmt: skip
        elif self.impl_type == 1:
            targets = torch.where(eye == 1.0, guidance, guidance - off_diag(guidance).median())  # fmt: skip
        elif self.impl_type == 2:
            targets = torch.where(eye == 1.0, guidance, guidance - off_diag(guidance).max())  # fmt: skip
        elif self.impl_type == 3:
            targets = torch.where(eye == 1.0, guidance, guidance - off_diag(guidance).min())  # fmt: skip

        similarity = torch.matmul(X, Y.T)

        similarity *= torch.exp(self.temp)

        return (self.ce(similarity, targets) + self.ce(similarity.T, targets)) / 2


class CosFaceCLIPLoss(CLIPLoss):
    def __init__(self, args) -> None:
        super().__init__(args)

        self.margin = args.clip_margin_init

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        b = X.shape[0]
        targets = torch.eye(b, device=X.device, requires_grad=False)

        X = X.reshape(b, -1)
        Y = Y.reshape(b, -1)

        X = X / X.norm(dim=-1, keepdim=True)
        Y = Y / Y.norm(dim=-1, keepdim=True)

        similarity = torch.matmul(X, Y.T)

        similarity -= self.margin * targets

        similarity *= torch.exp(self.temp)

        return (self.ce(similarity, targets) + self.ce(similarity.T, targets)) / 2


class ArcFaceCLIPLoss(CLIPLoss):
    """Based on https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py"""

    def __init__(self, args, easy_margin: bool = False) -> None:
        super().__init__(args)

        self.easy_margin = easy_margin

        m: float = args.clip_margin_init
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        b = X.shape[0]
        targets = torch.eye(b, device=X.device, requires_grad=False)

        X = X.reshape(b, -1)
        Y = Y.reshape(b, -1)

        X = X / X.norm(dim=-1, keepdim=True)
        Y = Y / Y.norm(dim=-1, keepdim=True)

        similarity = torch.matmul(X, Y.T)

        similarity = self._add_angular_margin(similarity, targets)

        similarity *= torch.exp(self.temp)

        return (self.ce(similarity, targets) + self.ce(similarity.T, targets)) / 2

    def _add_angular_margin(self, cosine, targets) -> torch.Tensor:
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2)).clamp(0, 1)

        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        return (targets * phi) + ((1.0 - targets) * cosine)


class AdaptiveMarginCLIPLoss(CLIPLoss):
    """Applies margins on negative pairs."""

    def __init__(self, args) -> None:
        super().__init__(args)

        self.margin = nn.Parameter(torch.tensor([float(args.clip_margin_init)]))
        self.margin_min = args.clip_margin_min
        self.margin_max = args.clip_margin_max
        if not args.clip_margin_learn:
            self.margin.requires_grad = False

        self.impl_type = args.impl_type

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        b = X.shape[0]
        targets = torch.arange(b, device=X.device, requires_grad=False).long()

        X = X.reshape(b, -1)
        Y = Y.reshape(b, -1)

        X = X / X.norm(dim=-1, keepdim=True)
        Y = Y / Y.norm(dim=-1, keepdim=True)

        similarity = torch.matmul(X, Y.T)

        # guidance = torch.matmul(Y, Y.T)

        # similarity -= guidance * self.margin

        similarity = self._add_angular_margin(similarity)  # ( b, b )

        similarity *= torch.exp(self.temp)

        return (self.ce(similarity, targets) + self.ce(similarity.T, targets)) / 2

    def _add_angular_margin(self, cosine: torch.Tensor) -> torch.Tensor:
        angles = torch.acos(cosine.clamp(-1, 1))  # ( b, b )

        margin = self._margin(cosine.shape[0])

        cosine_plus_margin = torch.cos(angles + margin)
        cosine = torch.cos(angles)

        # Keep the cost function monotonically decreasing
        return torch.where(
            angles <= np.pi - margin,
            cosine_plus_margin,
            cosine - margin * torch.sin(margin),
        )

    def _margin(self, b: int) -> torch.Tensor:
        # to_margin = -torch.eye(b, device=self.margin.device) + 1
        to_margin = torch.eye(b, device=self.margin.device, requires_grad=False)

        return self.margin * to_margin


class CircleCLIPLoss(CLIPLoss):
    def __init__(self, args) -> None:
        super().__init__(args)

        self.m = args.circle_relax
        self.o_p = 1 + self.m
        self.o_n = -self.m
        self.margin_p = 1 - self.m
        self.margin_n = self.m

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        b = X.shape[0]

        targets = torch.arange(b, device=X.device, requires_grad=False)
        targets_onehot = F.one_hot(targets, num_classes=b).to(torch.float32)

        X = X.reshape(b, -1)
        Y = Y.reshape(b, -1)

        X = X / X.norm(dim=-1, keepdim=True)
        Y = Y / Y.norm(dim=-1, keepdim=True)

        similarity = torch.matmul(X, Y.T)  # ( b, b )

        similarity -= self._margin(targets_onehot)
        similarity *= self._scaling(similarity, targets_onehot)
        similarity *= torch.exp(self.temp)

        return (self.ce(similarity, targets) + self.ce(similarity.T, targets)) / 2

    def _scaling(
        self, similarity: torch.Tensor, targets_onehot: torch.Tensor
    ) -> torch.Tensor:
        """self.o_p = 1.2, self.o_n = -0.2
        Args:
            similarity ( b, b ): _description_
            target_onehot ( b, b ): Float32 one-hot encoded.
        Returns:
            penalty ( b, b ): _description_
        """
        return torch.where(
            targets_onehot == 1,
            torch.relu(self.o_p - similarity),
            torch.relu(similarity - self.o_n),
        )

    def _margin(self, targets_onehot: torch.Tensor) -> torch.Tensor:
        """self.margin_p = 0.6, self.margin_n = 0.4
        Args:
            targets_onehot ( b, b ): Float32 one-hot encoded.
        Returns:
            margin ( b, b ): _description_
        """
        return torch.where(targets_onehot == 1, self.margin_p, self.margin_n)


class GeometricCLIPLoss(CLIPLoss):
    def __init__(self, args) -> None:
        super().__init__(args)

        self.ce = nn.CrossEntropyLoss(reduction=args.reduction)
        self.bce = BinaryCrossEntropyLoss(args.reduction)

        self.m = args.circle_relax

        self.impl_type = args.impl_type

    def forward(self, Y: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, (t,) f ): _description_
            Y ( b, (t,) f ): _description_
        Returns:
            torch.Tensor: _description_
        """
        b = X.shape[0]

        X = X.reshape(b, -1)
        Y = Y.reshape(b, -1)

        X = X / X.norm(dim=-1, keepdim=True)
        Y = Y / Y.norm(dim=-1, keepdim=True)

        similarity = torch.matmul(X, Y.T)  # ( b, b )
        guidance = torch.matmul(Y, Y.T)  # ( b, b )

        targets = torch.eye(b, device=X.device, requires_grad=False)  # ( b, b )
        # torch.arange(b, device=X.device, requires_grad=False)

        if self.impl_type in [0, 1, 2]:
            similarity *= torch.exp(self.temp)

        # Original CLIP loss for comparison
        if self.impl_type == 0:
            loss = (self.ce(similarity, targets) + self.ce(similarity.T, targets)) / 2

        # Simply CLIP loss with soft targets
        elif self.impl_type == 1:
            soft_targets = F.softmax(guidance, dim=1)  # ( b, b )

            loss = (self.ce(similarity, soft_targets) + self.ce(similarity.T, soft_targets)) / 2  # fmt: skip

        # BCE-CLIP with soft targets
        elif self.impl_type == 2:
            loss = (self.bce(similarity, guidance) + self.bce(similarity.T, guidance)) / 2  # fmt: skip

        # CircleLoss-like
        elif self.impl_type in [3, 4]:
            if self.impl_type == 3:
                guidance = torch.where(targets == 1, guidance, 1 - guidance)

            similarity -= self._margin(targets, m=guidance)
            similarity *= self._scaling(similarity, targets, m=guidance)
            similarity *= torch.exp(self.temp)

            loss = (self.bce(similarity, targets) + self.bce(similarity.T, targets)) / 2  # fmt: skip

        # Only margin (CosFace-like loss)
        else:
            if self.impl_type == 5:
                margin = guidance

            elif self.impl_type == 6:
                margin = 1 - guidance

            elif self.impl_type == 7:
                margin = self.m - guidance

            elif self.impl_type == 8:
                margin = torch.where(targets == 1, guidance, 1 - guidance)

            elif self.impl_type == 9:
                margin = torch.where(targets == 1, 1 - guidance, guidance)

            elif self.impl_type == 10:
                margin = torch.where(targets == 1, guidance, self.m - guidance)

            elif self.impl_type == 11:
                margin = torch.where(targets == 1, self.m - guidance, guidance)

            # Margin only for positive pairs.
            elif self.impl_type == 12:
                margin = torch.where(targets == 1, guidance, 0)

            elif self.impl_type == 13:
                margin = torch.where(targets == 1, 1 - guidance, 0)

            elif self.impl_type == 14:
                margin = torch.where(targets == 1, self.m - guidance, 0)

            # Margin only for negative pairs.
            elif self.impl_type == 15:
                margin = torch.where(targets == 1, 0, guidance)

            elif self.impl_type == 16:
                margin = torch.where(targets == 1, 0, 1 - guidance)

            elif self.impl_type == 17:
                margin = torch.where(targets == 1, 0, self.m - guidance)

            elif self.impl_type == 18:
                margin = torch.where(targets == 1, -guidance, 0)

            elif self.impl_type == 19:
                margin = -guidance

            similarity -= margin

            similarity *= self.temp

            loss = (self.bce(similarity, targets) + self.bce(similarity.T, targets)) / 2  # fmt: skip

        return loss

    @staticmethod
    def _margin(targets, m):
        """_summary_
        Args:
            targets ( b, b ): Float32 one-hot encoded.
            m ( b, b ): Relaxation margin.
        Returns:
            torch.Tensor: _description_
        """
        margin_p = 1 - m
        margin_n = m

        return torch.where(targets == 1, margin_p, margin_n)

    @staticmethod
    def _scaling(similarity, targets, m):
        """_summary_
        Args:
            similarity ( b, b ): _description_
            targets ( b, b ): Float32 one-hot encoded.
            m ( b, b ): Relaxation margin.
        Returns:
            _type_: _description_
        """
        o_p = 1 + m
        o_n = -m

        return torch.where(
            targets == 1, torch.relu(o_p - similarity), torch.relu(similarity - o_n)
        )


class CLIPWithClassCosFaceLoss(CLIPLoss):
    def __init__(
        self, args, n_classes, n_high_categories: Optional[int] = None
    ) -> None:
        super().__init__(args)

        self.alpha = args.cosface_alpha
        self.n_classes = n_classes
        self.n_high_categories = n_high_categories

        # Centers of the classes
        self.W = nn.Parameter(torch.Tensor(n_classes, args.F))
        self.W.data.normal_()

        if n_high_categories is not None:
            self.W_high = nn.Parameter(torch.Tensor(n_high_categories, args.F))
            self.W_high.data.normal_()

        # Cosine margin
        self.margin = nn.Parameter(torch.tensor([float(args.clip_margin_init)]))
        self.margin_min = args.clip_margin_min
        self.margin_max = args.clip_margin_max
        if not args.clip_margin_learn:
            self.margin.requires_grad = False

    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        classes: Optional[torch.Tensor] = None,
        high_categories: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, f ): _description_
            Y ( b, f ): _description_
            classes ( b, ): Elements are integers [0, n_classes - 1].
                            If None (evaluation), super().forward() is called.
        Returns:
            torch.Tensor: _description_
        """
        loss = super().forward(X, Y)

        if classes is not None:
            loss += self.alpha * self.metric_loss(
                self.W, X, Y, classes.to(X.device), self.n_classes
            )

        if high_categories is not None:
            loss += self.alpha * self.metric_loss(
                self.W_high, X, Y, high_categories.to(X.device), self.n_high_categories
            )

        return loss

    def metric_loss(
        self,
        W: torch.Tensor,
        X: torch.Tensor,
        Y: torch.Tensor,
        classes: torch.Tensor,
        n_classes: int,
    ):
        classes_onehot = F.one_hot(classes, n_classes).to(torch.float32)

        W = W / W.norm(dim=-1, keepdim=True)
        X = X / X.norm(dim=-1, keepdim=True)
        Y = Y / Y.norm(dim=-1, keepdim=True)

        sim_x = torch.matmul(X, W.T)
        sim_y = torch.matmul(Y, W.T)

        sim_x = (sim_x - self._margin(classes_onehot)) * self._scaling(sim_x, classes_onehot)  # fmt: skip
        sim_y = (sim_y - self._margin(classes_onehot)) * self._scaling(sim_y, classes_onehot)  # fmt: skip

        return (
            self.cross_entropy(sim_x, classes) + self.cross_entropy(sim_y, classes)
        ) / 2

    def _scaling(self, *args, **kwargs) -> torch.Tensor:
        # FIXME: Probably exp is not needed, but keeping it for consistency.
        return self.temp.exp()

    def _margin(self, classes: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            classes ( b, n_classes ): Float32 one-hot encoded.
        Returns:
            margin ( b, n_classes ): _description_
        """
        return self.margin * classes

    def clamp_params(self) -> None:
        super().clamp_params()

        if not (self.margin_min is None and self.margin_max is None):
            self.margin.data.clamp_(min=self.margin_min, max=self.margin_max)


class CLIPWithClassCircleLoss(CLIPWithClassCosFaceLoss):
    def __init__(self, args, n_classes, n_high_categories) -> None:
        super().__init__(args, n_classes, n_high_categories)

        self.m = args.clip_margin_init

        self.margin_p = 1 - self.m
        self.margin_n = self.m
        self.o_p = 1 + self.m
        self.o_n = -self.m

    def _scaling(self, similarity: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            similarity ( b, n_classes ): _description_
            classes ( b, n_classes ): Float32 one-hot encoded.
        Returns:
            penalty ( b, n_classes ): _description_
        """
        return torch.where(
            classes == 1,
            torch.relu(self.o_p - similarity),
            torch.relu(similarity - self.o_n),
        )

    def _margin(self, classes: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            classes ( b, n_classes ): Float32 one-hot encoded.
        Returns:
            margin ( b, n_classes ): _description_
        """
        return torch.where(classes == 1, self.margin_p, self.margin_n)

    def clamp_params(self) -> None:
        pass


class NearestNeighborCLIPLoss(nn.Module):
    def __init__(self, args):
        super().__init__()

        # CLIP related
        self.cross_entropy = nn.CrossEntropyLoss(reduction=args.reduction)

        self.temp = nn.Parameter(torch.tensor([float(args.clip_temp_init)]))
        if not args.clip_temp_learn:
            self.temp.requires_grad = False

        # Nearest Neighbor related
        self.k = args.nnclip_k
        self.symmetric = args.nnclip_symmetric
        self.support_size = args.nnclip_support_size
        self.alpha = args.nnclip_alpha

        self.support_set_x = torch.randn(self.support_size, args.F)
        if self.symmetric:
            self.support_set_y = torch.randn(self.support_size, args.F)

        self.bce_logits = nn.BCEWithLogitsLoss(reduction=args.reduction)

    def forward(self, Y: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            X ( b, f ): _description_
            Y ( b, f ): _description_

        Returns:
            torch.Tensor: _description_
        """
        b, f = X.shape

        targets = torch.arange(b, dtype=torch.long, device=X.device, requires_grad=False)  # fmt: skip
        X = X / X.norm(dim=-1, keepdim=True)
        Y = Y / Y.norm(dim=-1, keepdim=True)

        logits = torch.matmul(X, Y.T) * torch.exp(self.temp)

        clip_loss = (self.cross_entropy(logits, targets) + self.cross_entropy(logits.T, targets)) / 2  # fmt: skip

        # -------------------------
        #   Nearest Neighbor Loss
        # -------------------------
        nnclip_loss = self._calc_nnclip_loss(X, self.support_set_x.to(X.device))
        if self.symmetric:
            nnclip_loss = (
                nnclip_loss + self._calc_nnclip_loss(Y, self.support_set_y.to(Y.device))
            ) / 2

        self._update_support_set(X, Y)

        return clip_loss + self.alpha * nnclip_loss

    def _calc_nnclip_loss(self, X, support_set):
        similarity = calc_similarity(X, support_set, sequential=True, pbar=False)
        # ( b, support_size )
        topk = torch.topk(similarity, self.k, dim=1)[1]  # ( b, k )
        targets = F.one_hot(topk, num_classes=self.support_size)
        targets = targets.sum(dim=1).to(torch.float32)  # ( b, support_size )

        similarity = similarity * torch.exp(self.temp)

        return self.bce_logits(similarity, targets)

    @torch.no_grad()
    def _update_support_set(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """_summary_

        Args:
            X ( b, f ): _description_
            Y ( b, f ): _description_
        """
        device = self.support_set_x.device

        self.support_set_x = torch.cat([self.support_set_x, X.to(device)], dim=0)[
            -self.support_size :
        ]
        if self.symmetric:
            self.support_set_y = torch.cat([self.support_set_y, Y.to(device)], dim=0)[
                -self.support_size :
            ]


# BinaryCrossEntropyLoss.forward()

#     probs_diag = torch.diag(probs)
#     targets_diag = torch.diag(targets)
#     ce = -(targets_diag * probs_diag.log()).mean()
#     # cprint((ce, F.cross_entropy(logits, targets)), "yellow")
#     assert torch.isclose(ce, F.cross_entropy(logits, targets)).item()
#     assert (targets_diag == 1).all()

#     probs_offdiag = self._off_diag(probs)
#     targets_offdiag = self._off_diag(targets)
#     assert torch.isclose(probs.sum(), probs_diag.sum() + probs_offdiag.sum()).item()
#     assert torch.isclose(targets.sum(), targets_diag.sum() + targets_offdiag.sum()).item()  # fmt: skip
#     assert (targets_offdiag == 0).all()

#     bce3 = ce - ((1 - targets_offdiag) * (1 - probs_offdiag).log()).mean()

#     print(bce.mean(), bce2.mean(), bce3)
#     sys.exit()


# class MSELoss(nn.Module):
#     def __init__(self, reduction: str = "mean") -> None:
#         super().__init__()

#         self.reduction = reduction

#     def forward(self, X: torch.Tensor, Y: torch.Tensor, *_):
#         b = X.shape[0]

#         return F.mse_loss(X.reshape(b, -1), Y.reshape(b, -1), reduction=self.reduction)


# class HypersphericalKLLoss(nn.Module):
#     def __init__(
#         self, dim: int, beta: float = 1.0, l: int = 1, reduction: str = "batchmean"
#     ) -> None:
#         super().__init__()

#         self.beta = beta
#         self.l = l
#         self.reduction = reduction

#         self.q = HypersphericalUniform(dim=dim)

#     def forward(self, X, Y, kappa):
#         """_summary_
#         Args:
#             X ( b, 1, d ): _description_
#             Y ( b, 1, d ): _description_
#             kappa ( b, 1, 1 ): _description_
#         """
#         X = rearrange(X, "b t d -> b (t d)")
#         Y = rearrange(Y, "b t d -> b (t d)")
#         kappa = rearrange(kappa, "b t 1 -> (b t 1)")

#         p = PowerSpherical(loc=X, scale=kappa)
#         X = p.rsample((self.l,))  # ( l, b, d )

#         reg_loss = kl_divergence(p, self.q)  # .mean()
#         print(reg_loss.mean(), reg_loss.sum())
#         kl1_loss = F.kl_div(X.log_softmax(dim=-1), Y.softmax(dim=-1), reduction=self.reduction)  # fmt: skip
#         kl2_loss = F.kl_div(Y.log_softmax(dim=-1), X.softmax(dim=-1), reduction=self.reduction)  # fmt: skip
#         sys.exit()

#         return self.beta * (reg_loss + kl1_loss + kl2_loss)
