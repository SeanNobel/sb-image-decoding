import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from einops import rearrange
from termcolor import cprint
from typing import List

from brain2face.utils.train_utils import sequential_apply


class DiagonalClassifier(nn.Module):
    def __init__(self, topk: List[int] = [1, 10]):
        super().__init__()

        self.topk = topk

        # NOTE: Do we need to adjust the accuracies for the dataset size?
        self.factor = 1  # self.batch_size / 241

    @torch.no_grad()
    def forward(
        self, Z: torch.Tensor, Y: torch.Tensor, sequential=False, return_pred=False
    ) -> torch.Tensor:
        batch_size = Z.size(0)

        diags = torch.arange(batch_size).to(device=Z.device)

        Z = Z.contiguous().view(batch_size, -1)
        Y = Y.contiguous().view(batch_size, -1)

        Z = Z / Z.norm(dim=-1, keepdim=True)
        Y = Y / Y.norm(dim=-1, keepdim=True)

        # NOTE: avoid CUDA out of memory like this
        if sequential:
            similarity = torch.empty(batch_size, batch_size).to(device=Z.device)

            pbar = tqdm(total=batch_size, desc="Similarity matrix of test size")

            for i in range(batch_size):
                # similarity[i] = (Z[i] @ Y.T) / torch.clamp(
                #     (Z[i].norm() * Y.norm(dim=1)), min=1e-8
                # )
                similarity[i] = Z[i] @ Y.T

                pbar.update(1)

            similarity = similarity.T

            torch.cuda.empty_cache()
        else:
            Z = rearrange(Z, "b f -> 1 b f")
            Y = rearrange(Y, "b f -> b 1 f")
            similarity = F.cosine_similarity(Z, Y, dim=-1)  # ( B, B )

        topk_accs = np.array(
            [self.top_k_accuracy(k, similarity, diags) for k in self.topk]
        )

        # NOTE: max similarity of speech and M/EEG representations is expected for corresponding windows
        # _top1accuracy = (
        #     (similarity.argmax(axis=1) == diags).to(torch.float).mean().item()
        # )
        # cprint(f"{topk_accs[0]}, {_top1accuracy}", "cyan")

        if return_pred:
            cprint(similarity.argmax(axis=1).shape, "cyan")
            cprint(Y.shape, "cyan")

            return topk_accs, similarity.argmax(axis=1).cpu()
        else:
            return topk_accs, similarity

    @staticmethod
    def top_k_accuracy(k: int, similarity: torch.Tensor, diags: torch.Tensor):
        # if k == 1:
        #     return (similarity.argmax(axis=1) == diags).to(torch.float).mean().item()  # fmt: skip
        # else:
        return np.mean(
            [
                label in row
                for row, label in zip(
                    torch.topk(similarity, k, dim=1, largest=True)[1], diags
                )
            ]
        )


# WIP
class LabelClassifier(nn.Module):
    def __init__(self, dataset, topk: List[int] = [1, 5], device="cuda"):
        super().__init__()

        self.topk = topk

        test_y_idxs = dataset.y_idxs[dataset.test_idxs].numpy()
        test_y_idxs, arg_unique = np.unique(test_y_idxs, return_index=True)

        test_y_list = np.take(dataset.y_list, dataset.test_idxs).take(arg_unique)

        Y = [
            Image.open(y).convert("RGB")
            for y in tqdm(test_y_list, desc="Loading test images for classification.")
        ]
        Y = torch.stack(
            [
                dataset.preprocess(y)
                for y in tqdm(Y, desc="Preprocessing test images for classification.")
            ]
        )
        Y = sequential_apply(
            Y,
            dataset.clip_model.encode_image,
            batch_size=32,
            device=device,
            desc="Encoding test images for classification.",
        )

        self.Y = Y / Y.norm(dim=-1, keepdim=True)
        print(self.Y.shape)

    @torch.no_grad()
    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        batch_size = Z.size(0)

        Z = Z.contiguous().view(batch_size, -1)

        Z = Z / Z.norm(dim=-1, keepdim=True)

        similarity = Z @ self.Y.T

        print(similarity.shape)
        sys.exit()

        topk_accs = np.array([self.top_k_accuracy(k, similarity, Y) for k in self.topk])

        return topk_accs
