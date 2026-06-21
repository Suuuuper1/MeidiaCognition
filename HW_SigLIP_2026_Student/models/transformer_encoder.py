import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 128):
        super().__init__()
        position = torch.arange(max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerTextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        max_len: int = 32,
        padding_idx: int = 0,
        pool_type: str = "mean",
    ):
        super().__init__()
        self.padding_idx = padding_idx
        self.pool_type = pool_type
        self.max_len = max_len
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.positional_encoding = PositionalEncoding(embed_dim, max_len=max_len + 1)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.proj = nn.Linear(embed_dim, embed_dim)

        # cls 池化
        if pool_type == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def _pool_text(self, x, padding_mask):
        if self.pool_type == "mean":
            valid = (~padding_mask).unsqueeze(-1).float()
            return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

        if self.pool_type == "max":
            # pad 位置填一个很小的数，再对序列维做 max
            x_masked = x.masked_fill(padding_mask.unsqueeze(-1), -1e4)
            return x_masked.max(dim=1).values

        if self.pool_type == "cls":
            return x[:, 0, :]

    def forward(self, input_ids):
        padding_mask = input_ids.eq(self.padding_idx)
        x = self.token_embedding(input_ids)

        if self.pool_type == "cls":
            batch_size = x.size(0)
            cls_vec = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_vec, x], dim=1)
            cls_pad = torch.zeros(batch_size, 1, dtype=torch.bool, device=input_ids.device)
            padding_mask = torch.cat([cls_pad, padding_mask], dim=1)

        x = self.positional_encoding(x)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        pooled = self._pool_text(x, padding_mask)
        return self.proj(pooled)
