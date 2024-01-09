import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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

        # Multi-Head AttentionアウトプットのFeedForward
        self.proj = nn.Linear(self.n_head * self.d_v, emb_dim)

        self.do_mask = do_mask
        if self.do_mask:
            # torch.trilは行列の右上三角部分をゼロにして返します（予測するトークンの右側をマスク）
            # nn.Moduleのregister_bufferは, モデルのパラメータとならないtensorを追加するのに使われます.
            self.register_buffer(
                name="mask",
                tensor=torch.tril(torch.ones(block_size, block_size)).view(
                    1, 1, block_size, block_size
                ),
            )

        # FIXME: for visualization but there would be smarter ways.
        self.att = None

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            x (torch.Tensor): ( B, T, C )
                B: バッチサイズ batch_size
                T: シークエンスの長さ. コンテクストサイズ （block_size, または上のn）よりも小さくないといけない.
                C: Embedding空間の次元数. 上の説明のd_model
            layer_past (_type_, optional): _description_. Defaults to None.
        Returns:
            _type_: _description_
        """
        B, T, C = X.size()

        # Que, Key, Valueをそれぞれの全結合層で計算
        Q = self.query(X)  # ( B, T, n_head * d_qk )
        K = self.key(X)  # ( B, T, n_head * d_qk )
        V = self.value(X)  # ( B, T, n_head * d_v )

        # Multi-Head
        Q = Q.view(B, T, self.n_head, self.d_qk).transpose(1, 2)
        # ( B, n_heads, T, d_k )
        K = K.view(B, T, self.n_head, self.d_qk).transpose(1, 2)
        # ( B, n_heads, T, d_k )
        V = V.view(B, T, self.n_head, self.d_v).transpose(1, 2)
        # ( B, n_heads, T, d_v )

        # QとKの行列積をとり, sqrt(d_k)でスケール
        # ( B, n_heads, T, d_k ) x ( B, n_heads, d_k, T ) -> ( B, n_heads, T, T )
        att = (Q @ K.transpose(-2, -1)) * (1.0 / math.sqrt(K.size(-1)))

        # Attention matrixの右上三角部分をマスク
        if self.do_mask:
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))

        att = F.softmax(att, dim=-1)  # ( B, n_heads, T, T )

        # 正則化
        att = self.attn_drop(att)

        if self.return_attn:
            return att

        # VとのMatMul
        # ( B, n_heads, T, T ) x ( B, n_heads, T, d_v ) -> ( B, n_heads, T, d_v )
        y = att @ V

        # 各headからの出力を結合する.
        # ( B, n_heads, T, d_v ) -> ( B, T, n_heads, d_v ) -> ( B, T, n_heads * d_v )
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.d_v)

        # Attentionアウトプットの全結合層
        y = self.proj_drop(self.proj(y))  # ( B, T, emb_dim )

        self.att = att.detach().cpu().numpy()

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
    def __init__(
        self, emb_dim: int, mult: int = 4, residual: bool = True, ff_pdrop: float = 0.1
    ):
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
