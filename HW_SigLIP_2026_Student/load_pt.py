#!/usr/bin/env python3
"""
将 .pt (PyTorch) 文件的内容转换为 JSON Lines (.jsonl) 文件。

用法:
    python pt_to_jsonl.py <input.pt> [output.jsonl]

如果未指定输出文件名，则默认输出到与输入文件同目录下的 <input>.jsonl。
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import numpy as np


def make_serializable(obj):
    """将 PyTorch 张量、NumPy 数组等递归转换为 Python 原生可序列化类型。"""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [make_serializable(item) for item in obj]
    if isinstance(obj, dict):
        return {key: make_serializable(val) for key, val in obj.items()}
    # 其他类型保持原样（int, float, str, bool, None）
    return obj


def convert_to_jsonl(data, output_path):
    """
    将加载的数据转换为 JSON Lines 格式并写入文件。
    逻辑：
    - 如果 data 是 list -> 每个元素写一行。
    - 如果 data 是 dict：
        a) 如果所有 value 都是等长的 list，则按索引转换成 dict 列表（类似记录集）。
        b) 否则将整个 dict 当作一个对象，写一行。
    - 其他类型 -> 整个对象写一行。
    """
    lines = []

    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        # 检查是否可以转换为记录列表（如 {“id”: [1,2], “text”: [“a”,“b”]}）
        list_values = {k: v for k, v in data.items() if isinstance(v, list)}
        if list_values:
            # 确保所有 list 长度一致
            length = len(next(iter(list_values.values())))
            if all(len(v) == length for v in list_values.values()):
                # 按索引重组
                records = []
                for i in range(length):
                    record = {}
                    for k, v in data.items():
                        if k in list_values:
                            record[k] = v[i]
                        else:
                            record[k] = v  # 标量字段所有行共享
                    records.append(record)
            else:
                # 长度不一致，当作单个对象
                records = [data]
        else:
            records = [data]
    else:
        records = [data]

    # 写入 JSON Lines
    with open(output_path, 'w', encoding='utf-8') as f:
        for rec in records:
            serializable = make_serializable(rec)
            f.write(json.dumps(serializable, ensure_ascii=False) + '\n')

    print(f"成功转换 {len(records)} 条记录 -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="将 .pt 文件转换为 .jsonl")
    parser.add_argument("input", type=str, help="输入的 .pt 文件路径")
    parser.add_argument("output", nargs="?", default=None, help="输出的 .jsonl 文件路径（可选）")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"错误: 文件 '{input_path}' 不存在。", file=sys.stderr)
        sys.exit(1)

    # 确定输出路径
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix(".jsonl")

    print(f"加载 {input_path} ...")
    try:
        data = torch.load(input_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"加载 .pt 文件失败: {e}", file=sys.stderr)
        sys.exit(1)

    convert_to_jsonl(data, output_path)


if __name__ == "__main__":
    main()