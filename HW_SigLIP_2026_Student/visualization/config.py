import os
import torch

MODEL_NAME = "google/siglip-base-patch16-224"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR  = "./cache"

HEATMAP_MAX_N = 16   # 相似度矩阵最大图文对数
N_SAMPLES     = 1000 # 从数据集中最多加载的样本数
