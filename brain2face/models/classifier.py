import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from einops import rearrange
from termcolor import cprint


class Classifier(nn.Module):
    # NOTE: experimental

    def __init__(self, args):
        super(Classifier, self).__init__()

        # NOTE: Do we need to adjust the accuracies for the dataset size?
        self.factor = 1  # self.batch_size / 241

    @torch.no_grad()
    def forward(
        self, Z: torch.Tensor, Y: torch.Tensor, sequential=False, return_pred=False
    ) -> torch.Tensor:
        batch_size = Z.size(0)
        diags = torch.arange(batch_size).to(device=Z.device)
        x = Z.view(batch_size, -1)
        y = Y.view(batch_size, -1)

        # NOTE: avoid CUDA out of memory like this
        if sequential:
            similarity = torch.empty(batch_size, batch_size).to(device=Z.device)

            pbar = tqdm(total=batch_size, desc="Similarity matrix of test size")

            for i in range(batch_size):
                similarity[i] = (x[i] @ y.T) / torch.clamp(
                    (x[i].norm() * y.norm(dim=1)), min=1e-8
                )

                pbar.update(1)

            similarity = similarity.T

        else:
            x_ = rearrange(x, "b f -> 1 b f")
            y_ = rearrange(y, "b f -> b 1 f")
            similarity = F.cosine_similarity(x_, y_, dim=-1)  # ( B, B )

        # NOTE: max similarity of speech and M/EEG representations is expected for corresponding windows
        top1accuracy = (similarity.argmax(axis=1) == diags).to(torch.float).mean().item()
        try:
            top10accuracy = np.mean(
                [
                    label in row
                    for row, label in zip(
                        torch.topk(similarity, 10, dim=1, largest=True)[1], diags
                    )
                ]
            )
        except:
            print(similarity.size())

            raise

        if return_pred:
            cprint(similarity.argmax(axis=1).shape, "cyan")
            cprint(Y.shape, "cyan")
            return (
                top1accuracy,
                top10accuracy,
                similarity.argmax(axis=1).cpu(),
            )

        else:
            return top1accuracy, top10accuracy, similarity
