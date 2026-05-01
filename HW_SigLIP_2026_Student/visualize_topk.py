import argparse
import os
from typing import List

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data_loader import build_transform
from models import SigLIPModel
from utils import SimpleTokenizer, read_caption_file


class ImageOnlyDataset(Dataset):
    def __init__(self, image_root: str, image_ids: List[str], image_size: int = 224):
        self.image_root = image_root
        self.image_ids = image_ids
        self.transform = build_transform(image_size=image_size, train=False)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_root, image_id)
        image = Image.open(image_path).convert("RGB")
        return {"image": self.transform(image), "image_id": image_id}


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize text-to-image Top-K retrieval results.")
    parser.add_argument("--data-dir", default="Flickr8k", help="Directory containing images/ and caption split files.")
    parser.add_argument("--checkpoint", required=True, help="Path to best_siglip.pt or latest_siglip.pt.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--num-queries", type=int, default=8, help="How many captions to visualize.")
    parser.add_argument("--query-captions", nargs="*", default=None, help="Optional custom text queries.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", default="outputs/topk_vis")
    return parser.parse_args()


def load_checkpoint(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def rebuild_tokenizer_for_eval(captions, checkpoint, max_len):
    tokenizer = SimpleTokenizer(captions, min_freq=1, max_len=max_len)
    word2idx = checkpoint.get("tokenizer_word2idx")
    if not word2idx:
        return tokenizer
    tokenizer.word2idx = dict(word2idx)
    tokenizer.idx2word = [""] * len(tokenizer.word2idx)
    for word, idx in tokenizer.word2idx.items():
        tokenizer.idx2word[idx] = word
    return tokenizer


@torch.no_grad()
def encode_images(model, image_root, image_ids, image_size, batch_size, device):
    dataset = ImageOnlyDataset(image_root=image_root, image_ids=image_ids, image_size=image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    embeds = []
    ordered_image_ids = []
    for batch in tqdm(loader, desc="encode-images", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        image_embeds = model.image_encoder(images)
        image_embeds = F.normalize(image_embeds, dim=-1)
        embeds.append(image_embeds.cpu())
        ordered_image_ids.extend(batch["image_id"])
    return torch.cat(embeds, dim=0), ordered_image_ids


def draw_query_panel(query_caption: str, ranked_image_ids: List[str], image_root: str, output_path: str):
    tile_w, tile_h = 224, 224
    header_h = 88
    gap = 8
    num_cols = len(ranked_image_ids)
    canvas_w = gap + num_cols * (tile_w + gap)
    canvas_h = header_h + gap + tile_h + gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(250, 250, 250))
    draw = ImageDraw.Draw(canvas)

    draw.text((12, 12), "Text Query:", fill=(0, 0, 0))
    draw.text((12, 34), query_caption, fill=(30, 30, 30))

    for rank, image_id in enumerate(ranked_image_ids, start=1):
        x0 = gap + (rank - 1) * (tile_w + gap)
        y0 = header_h
        image = Image.open(os.path.join(image_root, image_id)).convert("RGB").resize((tile_w, tile_h))
        canvas.paste(image, (x0, y0))
        draw.rectangle([(x0, y0), (x0 + tile_w, y0 + tile_h)], outline=(30, 160, 80), width=2)
        draw.text((x0 + 6, y0 + 6), f"Top-{rank}", fill=(255, 255, 255))
        draw.text((x0 + 6, y0 + tile_h - 20), image_id, fill=(255, 255, 255))

    canvas.save(output_path)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    ckpt_args = checkpoint.get("args", {})
    text_encoder = ckpt_args.get("text_encoder", "transformer")
    embed_dim = ckpt_args.get("embed_dim", 256)
    image_width = ckpt_args.get("image_width", 32)
    max_len = ckpt_args.get("max_len", 32)

    split_file = os.path.join(args.data_dir, f"{args.split}_captions.txt")
    rows = read_caption_file(split_file)
    if len(rows) == 0:
        raise RuntimeError(f"No captions found in {split_file}")

    tokenizer = rebuild_tokenizer_for_eval((row["caption"] for row in rows), checkpoint, max_len=max_len)
    model = SigLIPModel(
        vocab_size=len(tokenizer),
        embed_dim=embed_dim,
        image_width=image_width,
        text_encoder=text_encoder,
        max_len=max_len,
    )
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()

    unique_image_ids = list(dict.fromkeys([row["image_id"] for row in rows]))
    image_root = os.path.join(args.data_dir, "images")
    image_embeds, image_ids = encode_images(
        model=model,
        image_root=image_root,
        image_ids=unique_image_ids,
        image_size=args.image_size,
        batch_size=args.batch_size,
        device=device,
    )

    if args.query_captions and len(args.query_captions) > 0:
        query_captions = args.query_captions
    else:
        query_captions = [row["caption"] for row in rows[: args.num_queries]]

    with open(os.path.join(args.output_dir, "topk_results.txt"), "w", encoding="utf-8") as f:
        for query_idx, caption in enumerate(query_captions, start=1):
            input_ids = torch.tensor([tokenizer.encode(caption)], dtype=torch.long, device=device)
            text_embed = model.text_encoder(input_ids)
            text_embed = F.normalize(text_embed, dim=-1).cpu()
            scores = (text_embed @ image_embeds.t()).squeeze(0)
            k = min(args.topk, scores.size(0))
            top_indices = scores.topk(k).indices.tolist()
            ranked_image_ids = [image_ids[idx] for idx in top_indices]

            f.write(f"[Query {query_idx}] {caption}\n")
            for rank, image_id in enumerate(ranked_image_ids, start=1):
                f.write(f"  Top-{rank}: {image_id}\n")
            f.write("\n")

            panel_path = os.path.join(args.output_dir, f"query_{query_idx:02d}.png")
            draw_query_panel(caption, ranked_image_ids, image_root=image_root, output_path=panel_path)

    print(f"Top-K visualization saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
