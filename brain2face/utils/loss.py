import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm


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
    def __init__(self, args):
        super().__init__()
        self.compute_similarity = nn.CosineSimilarity(dim=-1)
        self._criterion = nn.CrossEntropyLoss(reduction=args.reduction)

        self.temp = nn.Parameter(torch.tensor([float(args.clip_temp_init)]))
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
            logits = torch.matmul(x, y.T)

        # scale by temperature
        logits *= torch.exp(self.temp)

        # NOTE: as in https://arxiv.org/abs/2103.00020
        loss = (self._criterion(logits, targets) + self._criterion(logits.t(), targets)) / 2  # fmt: skip

        if return_logits:
            return logits, loss
        else:
            return loss


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

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
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
