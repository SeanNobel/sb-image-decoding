import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ISTA(nn.Module):
    """Sparse Coding Proximal Step"""

    def __init__(self, dim, dropout=0.0, step_size=0.1):
        super().__init__()

        self.weight = nn.Parameter(torch.Tensor(dim, dim))

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.weight)

        self.step_size = step_size
        self.lambd = 0.1

    def forward(self, x):
        # compute D^T * D * x
        x1 = F.linear(x, self.weight, bias=None)
        grad_1 = F.linear(x1, self.weight.t(), bias=None)

        # compute D^T * x
        grad_2 = F.linear(x, self.weight.t(), bias=None)

        # compute negative gradient update: step_size * (D^T * x - D^T * D * x)
        grad_update = self.step_size * (grad_2 - grad_1) - self.step_size * self.lambd

        output = F.relu(x + grad_update)

        return output


class MSSA(nn.Module):
    """Multi-Head Subspace Self-Attention"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()

        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.qkv = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        w = rearrange(self.qkv(x), "b n (h d) -> b h n d", h=self.heads)

        dots = torch.matmul(w, w.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, w)

        out = rearrange(out, "b h n d -> b n (h d)")

        return self.to_out(out)
