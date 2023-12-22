import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm
from termcolor import cprint
from typing import Optional


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


class CLIPLoss(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.compute_similarity = nn.CosineSimilarity(dim=-1)

        if args.push_negative:
            self.cross_entropy = nn.BCELoss(reduction=args.reduction)
        else:
            self.cross_entropy = nn.CrossEntropyLoss(reduction=args.reduction)

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
        loss = (self.cross_entropy(logits, targets) + self.cross_entropy(logits.t(), targets)) / 2  # fmt: skip

        if return_logits:
            return logits, loss
        else:
            return loss

    def clamp_params(self):
        if not (self.temp_min is None and self.temp_max is None):
            self.temp.data.clamp_(min=self.temp_min, max=self.temp_max)


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


class CosFaceCLIPLoss(CLIPLoss):
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
            self.cross_entropy(sim_x, classes) + self.cross_entropy(sim_y, classes) / 2
        )

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


class CircleCLIPLoss(CosFaceCLIPLoss):
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
