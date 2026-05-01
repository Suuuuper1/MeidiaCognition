"""
SigLIP 预训练模型 Wrapper
=========================
加载 google/siglip-base-patch16-224，将视觉 / 文本编码器拆分为独立模块。
"""

import torch
import torch.nn as nn
import logging
from PIL import Image
from typing import List, Tuple, Optional
from torchvision import transforms

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_NAME, DEVICE

logger = logging.getLogger(__name__)


class SigLIPWrapper:

    def __init__(self, model_name: str = MODEL_NAME, device: str = DEVICE):
        self.device = device

        self.vision_model: Optional[nn.Module] = None
        self.text_model:   Optional[nn.Module] = None
        self.transform     = None
        self.tokenizer     = None
        self.embed_dim:    Optional[int] = None
        self.vision_proj:  Optional[nn.Module] = None
        self.text_proj:    Optional[nn.Module] = None

        self.logit_scale = nn.Parameter(torch.tensor(4.6052))
        self.logit_bias  = nn.Parameter(torch.tensor(-10.0))

        logger.info(f"加载 SigLIP 预训练权重：{model_name}")
        self._load_models(model_name)
        logger.info(f"SigLIPWrapper 初始化完成 | embed_dim={self.embed_dim} | device={self.device}")

    def _load_models(self, model_name: str) -> None:
        from transformers import AutoProcessor, AutoModel

        processor  = AutoProcessor.from_pretrained(model_name)
        full_model = AutoModel.from_pretrained(model_name)

        self.vision_model = full_model.vision_model.to(self.device).eval()
        self.text_model   = full_model.text_model.to(self.device).eval()

        if hasattr(full_model, "visual_projection"):
            self.vision_proj = full_model.visual_projection.to(self.device)
        if hasattr(full_model, "text_projection"):
            self.text_proj = full_model.text_projection.to(self.device)

        self.logit_scale = nn.Parameter(full_model.logit_scale.detach().clone().to(self.device))
        self.logit_bias  = nn.Parameter(full_model.logit_bias.detach().clone().to(self.device))

        img_cfg = processor.image_processor
        size    = img_cfg.size.get("height", 224)
        mean    = getattr(img_cfg, "image_mean", [0.5, 0.5, 0.5])
        std     = getattr(img_cfg, "image_std",  [0.5, 0.5, 0.5])

        self.transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        self.tokenizer = processor.tokenizer

        if self.vision_proj is not None:
            self.embed_dim = self.vision_proj.out_features
        else:
            self.embed_dim = full_model.config.vision_config.hidden_size

    @torch.no_grad()
    def encode_images(self, images: List[Image.Image], batch_size: int = 32) -> torch.Tensor:
        all_embeds = []
        for i in range(0, len(images), batch_size):
            batch   = images[i: i + batch_size]
            tensors = torch.stack(
                [self.transform(img.convert("RGB")) for img in batch]
            ).to(self.device)

            feats = self.vision_model(tensors)
            if isinstance(feats, (tuple, list)):
                feats = feats[0]
            feats = feats.pooler_output if hasattr(feats, "pooler_output") else feats
            if feats.dim() > 2:
                feats = feats.flatten(1)
            if self.vision_proj is not None:
                feats = self.vision_proj(feats)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_embeds.append(feats)
        return torch.cat(all_embeds, dim=0)

    @torch.no_grad()
    def encode_texts(self, texts: List[str], batch_size: int = 64) -> torch.Tensor:
        all_embeds = []
        for i in range(0, len(texts), batch_size):
            batch  = texts[i: i + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=64,
            ).to(self.device)

            out = self.text_model(**inputs)

            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                feats = out.pooler_output
            elif hasattr(out, "last_hidden_state"):
                feats = out.last_hidden_state[:, 0, :]
            else:
                raw   = out[0] if isinstance(out, (tuple, list)) else out
                feats = raw.mean(dim=1) if raw.dim() == 3 else raw

            if self.text_proj is not None:
                feats = self.text_proj(feats)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_embeds.append(feats)
        return torch.cat(all_embeds, dim=0)

    @torch.no_grad()
    def compute_cosine_similarity(
        self, image_embeds: torch.Tensor, text_embeds: torch.Tensor
    ) -> torch.Tensor:
        return torch.matmul(image_embeds, text_embeds.T)

    @torch.no_grad()
    def compute_similarity(
        self, image_embeds: torch.Tensor, text_embeds: torch.Tensor
    ) -> torch.Tensor:
        logits = torch.matmul(image_embeds, text_embeds.T)
        logits = logits * self.logit_scale.exp() + self.logit_bias
        return torch.sigmoid(logits)

    def compute_similarity_with_grad(
        self, image_embeds: torch.Tensor, text_embeds: torch.Tensor
    ) -> torch.Tensor:
        logits = torch.matmul(image_embeds, text_embeds.T)
        logits = logits * self.logit_scale.exp() + self.logit_bias
        return torch.sigmoid(logits)

    def get_logit_params(self) -> Tuple[float, float]:
        return self.logit_scale.item(), self.logit_bias.item()

    def get_embed_dim(self) -> int:
        return self.embed_dim

    @property
    def vision_encoder(self) -> nn.Module:
        return self.vision_model
