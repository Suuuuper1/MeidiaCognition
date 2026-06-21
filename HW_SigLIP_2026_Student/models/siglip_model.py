import torch.nn as nn

from .resnet_custom import ResNet18
from .transformer_encoder import TransformerTextEncoder


class MLPTextEncoder(nn.Module):
    """Small text baseline for quick comparisons against the Transformer."""

    def __init__(self, vocab_size: int, embed_dim: int = 256, padding_idx: int = 0):
        super().__init__()
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, input_ids):
        padding_mask = input_ids.eq(self.padding_idx)
        valid = (~padding_mask).unsqueeze(-1).float()
        x = self.embedding(input_ids)
        pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return self.net(pooled)


class SigLIPModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        image_width: int = 32,
        text_encoder: str = "transformer",
        max_len: int = 32,
        pool_type: str = "mean",
    ):
        super().__init__()
        self.image_encoder = ResNet18(width=image_width, output_dim=embed_dim)
        if text_encoder == "transformer":
            self.text_encoder = TransformerTextEncoder(
                vocab_size,
                embed_dim=embed_dim,
                max_len=max_len,
                pool_type=pool_type,
            )
        elif text_encoder == "mlp":
            self.text_encoder = MLPTextEncoder(vocab_size, embed_dim=embed_dim)
        else:
            raise ValueError(f"Unsupported text_encoder: {text_encoder}")

    def forward(self, images, input_ids):
        image_embeds = self.image_encoder(images)
        text_embeds = self.text_encoder(input_ids)
        return image_embeds, text_embeds
