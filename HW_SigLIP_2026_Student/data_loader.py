import os
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from utils import read_caption_file


def build_transform(image_size: int = 224, train: bool = True, use_aug: bool = False):
    norm = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    if not train:
        return T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                norm,
            ]
        )

    steps = [T.Resize((image_size, image_size))]
    if use_aug:
        steps.append(T.RandomHorizontalFlip(p=0.5))
        # 颜色抖动
        steps.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
    steps.extend([T.ToTensor(), norm])
    return T.Compose(steps)


class Flickr8kDataset(Dataset):
    def __init__(self, image_root: str, captions_file: str, tokenizer, transform=None):
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.transform = transform or build_transform(train=True)
        self.rows = read_caption_file(captions_file)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image_path = os.path.join(self.image_root, row["image_id"])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        input_ids = torch.tensor(self.tokenizer.encode(row["caption"]), dtype=torch.long)
        return {
            "image": image,
            "input_ids": input_ids,
            "image_id": row["image_id"],
            "caption": row["caption"],
        }


class SyntheticPairDataset(Dataset):
    """Tiny deterministic dataset for code sanity checks without Flickr8k images."""

    def __init__(self, tokenizer, size: int = 256, image_size: int = 64, num_concepts: int = 32):
        self.tokenizer = tokenizer
        self.size = size
        self.image_size = image_size
        self.num_concepts = num_concepts

    @staticmethod
    def captions(num_concepts: int = 32):
        return [f"synthetic object {idx}" for idx in range(num_concepts)]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        concept = idx % self.num_concepts
        generator = torch.Generator().manual_seed(concept)
        image = torch.rand((3, self.image_size, self.image_size), generator=generator)
        image[0].mul_((concept + 1) / self.num_concepts)
        image[1].mul_(1.0 - concept / self.num_concepts)
        caption = f"synthetic object {concept}"
        input_ids = torch.tensor(self.tokenizer.encode(caption), dtype=torch.long)
        return {
            "image": image,
            "input_ids": input_ids,
            "image_id": f"synthetic_{concept:03d}.jpg",
            "caption": caption,
        }
