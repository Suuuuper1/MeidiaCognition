"""
Flickr8K 数据集加载器
==========================
优先从 HuggingFace (tsystems/flickr8k) 自动下载，
若设置了本地目录则优先使用本地文件。

本地目录格式（标准 Flickr8K）：
    Flickr8k/
    ├── Images/
    ├── Flickr8k.token.txt  （或 captions.txt）
    ├── Flickr_8k.trainImages.txt
    ├── Flickr_8k.devImages.txt
    └── Flickr_8k.testImages.txt

用法：
    from data.flickr8k_loader import load_flickr8k, get_random_batch
    samples = load_flickr8k(split="train", n_samples=1000)
"""

import os
import random
import logging
from pathlib import Path
from PIL import Image
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

_HF_DATASET = "tsystems/flickr8k"

_DEFAULT_DATA_DIR = os.environ.get(
    "FLICKR8K_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Flickr8k"),
)

_SPLIT_FILES = {
    "train": "Flickr_8k.trainImages.txt",
    "val":   "Flickr_8k.devImages.txt",
    "test":  "Flickr_8k.testImages.txt",
}


def load_flickr8k(
    split: str = "train",
    data_dir: Optional[str] = None,
    n_samples: Optional[int] = None,
    seed: int = 42,
) -> List[Dict]:
    """
    加载 Flickr8K 数据集。

    加载顺序：
    1. 若本地目录（data_dir 或 FLICKR8K_DIR）存在，使用本地文件
    2. 否则从 HuggingFace (tsystems/flickr8k) 自动下载（约 1 GB）

    Returns:
        每项包含 {'image': PIL.Image, 'caption': str,
                   'all_captions': List[str], 'idx': int}
    """
    root = Path(data_dir or _DEFAULT_DATA_DIR)

    if root.exists():
        logger.info(f"使用本地 Flickr8K 数据集：{root}")
        return _load_local(root, split, n_samples, seed)

    logger.info(f"本地目录不存在，从 HuggingFace 加载 {_HF_DATASET} ...")
    result = _load_hf(n_samples, seed)
    if result is not None:
        return result

    raise RuntimeError(
        f"Flickr8K 加载失败。\n"
        f"  本地目录不存在：{root}\n"
        f"  HuggingFace 下载也失败。\n"
        f"请确保网络可访问 HuggingFace，或设置 FLICKR8K_DIR 环境变量指向本地数据目录。"
    )


def _load_hf(n_samples: Optional[int], seed: int) -> Optional[List[Dict]]:
    """从 HuggingFace tsystems/flickr8k 加载（只有 train split，8091 张图像）。"""
    try:
        from datasets import load_dataset
        ds = load_dataset(_HF_DATASET, split="train")

        indices = list(range(len(ds)))
        if n_samples and n_samples < len(indices):
            rng = random.Random(seed)
            indices = rng.sample(indices, n_samples)

        samples = []
        for new_idx, orig_idx in enumerate(indices):
            item = ds[orig_idx]
            raw_cap = item.get("captions") or item.get("caption") or [""]
            if isinstance(raw_cap, str):
                raw_cap = [raw_cap]
            primary = item.get("query") or raw_cap[0]
            samples.append({
                "image":        item["image"].convert("RGB"),
                "caption":      primary,
                "all_captions": raw_cap,
                "idx":          new_idx,
            })

        logger.info(f"Flickr8K 从 HuggingFace 加载完成：{len(samples)} 个样本")
        return samples

    except Exception as e:
        logger.warning(f"HuggingFace 加载失败：{e}")
        return None


def _load_local(root: Path, split: str, n_samples: Optional[int], seed: int) -> List[Dict]:
    """从本地标准目录格式加载。"""
    images_dir = root / "Images"
    token_file = root / "Flickr8k.token.txt"
    split_file = root / _SPLIT_FILES.get(split, _SPLIT_FILES["train"])

    if not token_file.exists():
        alt_token = root / "captions.txt"
        if alt_token.exists():
            token_file = alt_token
        else:
            raise FileNotFoundError(
                f"找不到标注文件：{token_file}（也尝试了 captions.txt）"
            )

    captions_map: Dict[str, List[str]] = {}

    if token_file.name == "captions.txt":
        with open(token_file, encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            comma_idx = line.index(",")
            fname = line[:comma_idx].strip()
            cap   = line[comma_idx + 1:].strip()
            captions_map.setdefault(fname, []).append(cap)
    else:
        with open(token_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                key, cap = line.split("\t", 1)
                fname = key.split("#")[0]
                captions_map.setdefault(fname, []).append(cap)

    if split_file.exists():
        with open(split_file, encoding="utf-8") as f:
            filenames = [ln.strip() for ln in f if ln.strip()]
    else:
        logger.warning(f"split 文件不存在：{split_file}，使用所有图像")
        filenames = list(captions_map.keys())

    if n_samples and n_samples < len(filenames):
        rng = random.Random(seed)
        filenames = rng.sample(filenames, n_samples)

    samples, missing = [], 0
    for idx, fname in enumerate(filenames):
        img_path = images_dir / fname
        if not img_path.exists():
            missing += 1
            continue
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            missing += 1
            continue
        all_caps = captions_map.get(fname, [""])
        samples.append({
            "image":        img,
            "caption":      all_caps[0],
            "all_captions": all_caps,
            "idx":          idx,
        })

    if missing:
        logger.warning(f"跳过 {missing} 张图像（文件缺失或损坏）")
    logger.info(f"Flickr8K [{split}] 本地加载完成：{len(samples)} 个样本")
    return samples


def get_random_batch(samples: List[Dict], n: int) -> List[Dict]:
    """从 samples 中随机抽取 n 个，不足则全部返回。"""
    return random.sample(samples, min(n, len(samples)))
