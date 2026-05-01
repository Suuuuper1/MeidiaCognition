import csv
import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import torch


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SimpleTokenizer:
    """A compact word-level tokenizer for the Flickr8k captions."""

    def __init__(self, captions: Iterable[str], min_freq: int = 2, max_len: int = 32):
        self.max_len = max_len
        self.word2idx: Dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.idx2word: List[str] = [PAD_TOKEN, UNK_TOKEN]
        self.build_vocab(captions, min_freq)

    def build_vocab(self, captions: Iterable[str], min_freq: int) -> None:
        counter = Counter()
        for caption in captions:
            counter.update(self.tokenize(caption))
        for word, freq in counter.most_common():
            if freq >= min_freq and word not in self.word2idx:
                self.word2idx[word] = len(self.idx2word)
                self.idx2word.append(word)

    def tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-z0-9']+", text.lower())

    def encode(self, text: str) -> List[int]:
        tokens = self.tokenize(text)
        token_ids = [self.word2idx.get(token, self.word2idx[UNK_TOKEN]) for token in tokens]
        token_ids = token_ids[: self.max_len]
        token_ids += [self.word2idx[PAD_TOKEN]] * (self.max_len - len(token_ids))
        return token_ids

    def __len__(self) -> int:
        return len(self.idx2word)


def read_caption_file(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            image_id = row[0].strip()
            caption = ",".join(row[1:]).strip()
            if image_id and caption:
                rows.append({"image_id": image_id, "caption": caption})
    return rows


@dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)
