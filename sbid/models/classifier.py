import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from einops import rearrange
from termcolor import cprint
from typing import List, Optional
import gc

from sbid.utils.loss import top_k_accuracy


class DiagonalClassifier(nn.Module):
    def __init__(self, loss_func, topk: List[int] = [1, 10]):
        super().__init__()

        self.loss_func = loss_func
        self.topk = topk

        # NOTE: Do we need to adjust the accuracies for the dataset size?
        self.factor = 1

    @torch.no_grad()
    def forward(
        self, Z: torch.Tensor, Y: torch.Tensor, sequential=False, return_pred=False
    ) -> torch.Tensor:
        batch_size = Z.size(0)

        diags = torch.arange(batch_size).to(device=Z.device)

        similarity = self.loss_func.similarity(Z, Y, sequential, mult_temp=False)
        # ( b, b )

        topk_accs = np.array([top_k_accuracy(k, similarity, diags) for k in self.topk])

        # NOTE: max similarity of speech and M/EEG representations is expected for corresponding windows
        # _top1accuracy = (
        #     (similarity.argmax(axis=1) == diags).to(torch.float).mean().item()
        # )

        if return_pred:
            cprint(similarity.argmax(axis=1).shape, "cyan")
            cprint(Y.shape, "cyan")

            return topk_accs, similarity.argmax(axis=1).cpu()
        else:
            return topk_accs, similarity


class LabelClassifier(nn.Module):
    def __init__(self, loss_func, dataset, topk: List[int] = [1, 5], device="cuda"):
        super().__init__()

        self.loss_func = loss_func
        self.topk = topk

        test_y_idxs = dataset.y_idxs[dataset.test_idxs].numpy()
        # ( 9600, )
        # NOTE: torch.unique has no return_index option
        test_y_idxs, arg_unique = np.unique(test_y_idxs, return_index=True)
        arg_unique = torch.from_numpy(arg_unique)
        # ( 2400, )
        self.test_y_idxs = torch.from_numpy(test_y_idxs)

        self.Y = torch.index_select(dataset.test_Y, 0, arg_unique)
        # ( 2400, F )
        self.categories = torch.index_select(dataset.test_categories, 0, arg_unique)

        self.Y = self.Y.to(device)
        self.categories = self.categories.to(device)
        self.test_y_idxs = self.test_y_idxs.to(device)

    @torch.no_grad()
    def forward(
        self,
        Z: torch.Tensor,
        y_idxs: torch.Tensor,
        y_encoder: Optional[nn.Module],
        sequential: bool = False,
        return_sim: bool = False,
    ) -> np.ndarray:
        """_summary_
        Args:
            Z ( b=9600, F ): _description_
            y_idxs ( b=9600, ): _description_
        Returns:
            torch.Tensor: _description_
        """
        Y = y_encoder.encode(self.Y) if y_encoder is not None else self.Y

        similarity = self.loss_func.similarity(Z, Y, sequential, mult_temp=False)
        # ( 2400, b )
        if return_sim:
            return similarity

        labels = y_idxs == self.test_y_idxs.unsqueeze(1)  # ( 2400, b )
        assert torch.all(labels.sum(dim=0) == 1)
        labels = labels.to(int).argmax(dim=0)  # ( b, )

        topk_accs = np.array([top_k_accuracy(k, similarity, labels) for k in self.topk])

        return topk_accs
