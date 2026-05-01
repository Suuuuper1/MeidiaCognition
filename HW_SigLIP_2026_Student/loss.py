import torch
import torch.nn as nn
import torch.nn.functional as F


class SigLIPLoss(nn.Module):
    """Sigmoid pairwise image-text loss used by SigLIP.

    For a batch of B aligned image/text pairs, the diagonal pairs are positives
    and all off-diagonal pairs are negatives.
    """

    def __init__(self, init_logit_scale: float = 10.0, init_logit_bias: float = -10.0):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(init_logit_scale)))
        self.logit_bias = nn.Parameter(torch.tensor(init_logit_bias, dtype=torch.float32))

    def forward(self, image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)

        # Pairwise cosine similarity: [B, D] x [D, B] -> [B, B]
        logits = image_embeds @ text_embeds.t()
        logit_scale = self.logit_scale.exp()
        logits = logits * logit_scale + self.logit_bias

        batch_size = logits.size(0)
        labels = torch.full_like(logits, -1.0)
        labels.fill_diagonal_(1.0)

        # softplus(x) == log(1 + exp(x)) for numerical stability.
        # Implements: log(1 + exp(s_ij * (-t * z_ij + b)))
        # where logits = t * z_ij + b, thus -logits below.
        loss = F.softplus(-labels * logits)
        return loss.sum(dim=-1).mean()
