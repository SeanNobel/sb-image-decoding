import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from termcolor import cprint


class SelfAttention(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        n_heads: int,
        block_size: int,
        h_qk: int = 8,
        h_v: int = 8,
        do_mask: bool = False,
        attn_pdrop: float = 0.1,
        proj_pdrop: float = 0.1,
        return_attn: bool = False,
    ):
        super().__init__()

        self.n_head = n_heads
        self.d_qk = emb_dim // h_qk
        self.d_v = emb_dim // h_v
        self.return_attn = return_attn

        self.query = nn.Linear(emb_dim, self.n_head * self.d_qk)
        self.key = nn.Linear(emb_dim, self.n_head * self.d_qk)
        self.value = nn.Linear(emb_dim, self.n_head * self.d_v)

        self.attn_drop = nn.Dropout(attn_pdrop)
        self.proj_drop = nn.Dropout(proj_pdrop)

        self.proj = nn.Linear(self.n_head * self.d_v, emb_dim)

        self.do_mask = do_mask
        if self.do_mask:
            self.register_buffer(
                name="mask",
                tensor=torch.tril(torch.ones(block_size, block_size)).view(
                    1, 1, block_size, block_size
                ),
            )

    def forward(
        self,
        X: torch.Tensor,
        pos_enc_k: Optional[torch.Tensor] = None,
        pos_enc_v: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, t, d ): _description_
        """
        assert not (pos_enc_k is None) ^ (pos_enc_v is None), "pos_enc_k and pos_enc_v must be both None or both not None"  # fmt: skip

        b, t, d = X.size()

        Q = self.query(X)  # ( b, t, n_head * d_qk )
        K = self.key(X)  # ( b, t, n_head * d_qk )
        V = self.value(X)  # ( b, t, n_head * d_v )

        # Multi-Head
        Q = Q.view(b, t, self.n_head, self.d_qk).transpose(1, 2)
        # ( b, n_heads, t, d_k )
        K = K.view(b, t, self.n_head, self.d_qk).transpose(1, 2)
        # ( b, n_heads, t, d_k )
        V = V.view(b, t, self.n_head, self.d_v).transpose(1, 2)
        # ( b, n_heads, t, d_v )

        att = Q @ K.transpose(-2, -1)
        # ( b, n_heads, t, d_k ) x ( b, n_heads, d_k, t ) -> ( b, n_heads, t, t )

        if pos_enc_k is not None:
            rel_qk_att = Q.transpose(1, 2) @ pos_enc_k.transpose(1, 2)
            # ( b, t, n_heads, d_qk ) x ( t, d_qk, t ) -> ( b, t, n_heads, t )
            att = att + rel_qk_att.transpose(1, 2)

        att = att / math.sqrt(K.shape[-1])

        if self.do_mask:
            att = att.masked_fill(self.mask[:, :, :t, :t] == 0, float("-inf"))

        att = F.softmax(att, dim=-1)  # ( b, n_heads, t, t )

        att = self.attn_drop(att)

        y = att @ V
        # ( b, n_heads, t, t ) x ( b, n_heads, t, d_v ) -> ( b, n_heads, t, d_v )

        if pos_enc_v is not None:
            rel_qkv_att = att.transpose(1, 2) @ pos_enc_v
            # ( b, t, n_heads, t ) x ( t, t, d_v ) -> ( b, t, n_heads, d_v )
            y = y + rel_qkv_att.transpose(1, 2)

        y = y.transpose(1, 2).contiguous().view(b, t, self.n_head * self.d_v)
        # ( b, n_heads, t, d_v ) -> ( b, t, n_heads, d_v ) -> ( b, t, n_heads * d_v )

        y = self.proj_drop(self.proj(y))  # ( b, t, emb_dim )

        if self.return_attn:
            return y, att
        else:
            return y


class PreNorm(nn.Module):
    def __init__(self, fn, n_embd):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(n_embd)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class FeedForward(nn.Module):
    def __init__(self, emb_dim: int, mult: int = 4, residual: bool = True, ff_pdrop: float = 0.1):
        super().__init__()

        self.net = PreNorm(
            nn.Sequential(
                nn.Linear(emb_dim, mult * emb_dim),
                nn.GELU(),
                nn.Linear(mult * emb_dim, emb_dim),
                nn.Dropout(ff_pdrop),
            ),
            emb_dim,
        )

        if residual:
            self.net = Residual(self.net)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: ( batch_size, seq_len, emb_dim )
        Returns:
            X: ( batch_size, seq_len, emb_dim )
        """
        return self.net(X)


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        block_size: int,
        emb_dim: int,
        pos_enc: str = "sine",
        learn_pos_scale: bool = False,
    ):
        super().__init__()

        self.block_size = block_size
        self.emb_dim = emb_dim
        self.pos_enc = pos_enc

        if pos_enc == "learn":
            self.pe = nn.Parameter(torch.zeros(block_size, emb_dim))

        elif pos_enc == "sine":
            assert (emb_dim % 2 == 0), "Cannot use sin/cos positional encoding with odd dim"  # fmt: skip

            self.register_buffer("dims", torch.arange(0, emb_dim, 2, dtype=torch.float32))
            self.register_buffer(
                "position", torch.arange(block_size, dtype=torch.float32).unsqueeze(1)
            )
            self.scale = torch.tensor([1.0])
            if learn_pos_scale:
                self.scale = nn.Parameter(self.scale)
        else:
            raise NotImplementedError

    def forward(self, X: torch.Tensor, transpose=False) -> torch.Tensor:
        if self.pos_enc == "learn":
            pe = self.pe
        elif self.pos_enc == "sine":
            # NOTE: Keeping PE method in forward for scale learning.
            pe = self._sinusoidal_pe(X.device)

        if transpose:
            pe = pe.T

        return X + pe.unsqueeze(0)

    def _sinusoidal_pe(self, device) -> torch.Tensor:
        pe = torch.zeros(self.block_size, self.emb_dim, device=device)

        div_term = torch.exp(
            self.dims * -(torch.log(self.scale.to(device) * 10000.0) / self.emb_dim)
        )

        pe[:, 0::2] = torch.sin(self.position * div_term)
        pe[:, 1::2] = torch.cos(self.position * div_term)

        return pe


def relative_positional_encoding(
    block_size: int, emb_dim: int, max_position: int = 256
) -> torch.Tensor:
    """Works only with self-attention. Needs block_size_q and block_size_k to work with cross-attention.
    Args:
        block_size (int): _description_
        emb_dim (int): _description_
        max_position (int, optional): _description_. Defaults to 256.
    Returns:
        pe ( block_size, block_size, emb_dim )
    """
    pe = torch.zeros(block_size, block_size, emb_dim)

    position = torch.arange(block_size)
    position = (position[None] - position[:, None]).unsqueeze(-1)  # ( t, t, 1 )
    position = position.clamp(-max_position, max_position) + max_position

    div_term = torch.exp(
        torch.arange(0, emb_dim, 2, dtype=torch.float) * -(math.log(10000.0) / emb_dim)
    )
    # ( d / 2 )

    pe[:, :, 0::2] = torch.sin(position.float() * div_term)  # ( t, t, d / 2 )
    pe[:, :, 1::2] = torch.cos(position.float() * div_term)

    return pe
