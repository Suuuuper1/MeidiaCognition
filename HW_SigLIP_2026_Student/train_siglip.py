import argparse
import os
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data_loader import Flickr8kDataset, SyntheticPairDataset, build_transform
from loss import SigLIPLoss
from models import SigLIPModel
from training_utils import get_lr, init_weights, set_optimizer_lr
from utils import AverageMeter, SimpleTokenizer, read_caption_file, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a small SigLIP model on Flickr8k.")
    parser.add_argument("--data-dir", default="Flickr8k")
    parser.add_argument("--output-dir", default="outputs/siglip_resnet_transformer")
    parser.add_argument("--text-encoder", choices=["transformer", "mlp"], default="transformer")
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--use-cosine-lr", action="store_true")
    parser.add_argument("--pool-type", choices=["mean", "max", "cls"], default="mean")
    parser.add_argument("--custom-init", action="store_true")
    parser.add_argument("--data-aug",action="store_true")
    parser.add_argument("--optimized",action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-eval", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", default="")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def apply_optimized_defaults(args):
    if not args.optimized:
        return args
    if args.epochs == 10:
        args.epochs = 40
    if args.image_width == 32:
        args.image_width = 48
    if args.pool_type == "mean":
        args.pool_type = "cls"
    if args.lr == 3e-4:
        args.lr = 1e-4
    if args.warmup_epochs == 0:
        args.warmup_epochs = 3
    if not args.use_cosine_lr:
        args.use_cosine_lr = True
    if not args.custom_init:
        args.custom_init = True
    return args


def maybe_subset(dataset, limit):
    if limit and limit > 0:
        return Subset(dataset, range(min(limit, len(dataset))))
    return dataset


def build_dataloaders(args):
    if args.dry_run:
        captions = SyntheticPairDataset.captions()
        tokenizer = SimpleTokenizer(captions, min_freq=1, max_len=args.max_len)
        train_dataset = SyntheticPairDataset(tokenizer, size=256, image_size=64)
        val_dataset = SyntheticPairDataset(tokenizer, size=64, image_size=64)
        test_dataset = SyntheticPairDataset(tokenizer, size=64, image_size=64)
        args.image_size = 64
    else:
        all_rows = read_caption_file(os.path.join(args.data_dir, "captions.txt"))
        tokenizer = SimpleTokenizer((row["caption"] for row in all_rows), min_freq=args.min_freq, max_len=args.max_len)
        image_root = os.path.join(args.data_dir, "images")
        train_dataset = Flickr8kDataset(
            image_root=image_root,
            captions_file=os.path.join(args.data_dir, "train_captions.txt"),
            tokenizer=tokenizer,
            transform=build_transform(args.image_size, train=True, use_aug=args.data_aug),
        )
        val_dataset = Flickr8kDataset(
            image_root=image_root,
            captions_file=os.path.join(args.data_dir, "val_captions.txt"),
            tokenizer=tokenizer,
            transform=build_transform(args.image_size, train=False),
        )
        test_dataset = Flickr8kDataset(
            image_root=image_root,
            captions_file=os.path.join(args.data_dir, "test_captions.txt"),
            tokenizer=tokenizer,
            transform=build_transform(args.image_size, train=False),
        )

    train_dataset = maybe_subset(train_dataset, args.limit_train)
    val_dataset = maybe_subset(val_dataset, args.limit_eval)
    test_dataset = maybe_subset(test_dataset, args.limit_eval)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return tokenizer, train_loader, val_loader, test_loader


def train_one_epoch(model, criterion, loader, optimizer, device):
    model.train()
    criterion.train()
    meter = AverageMeter()
    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        image_embeds, text_embeds = model(images, input_ids)
        loss = criterion(image_embeds, text_embeds)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        meter.update(loss.item(), images.size(0))
    return meter.avg


@torch.no_grad()
def evaluate_loss(model, criterion, loader, device):
    model.eval()
    criterion.eval()
    meter = AverageMeter()
    for batch in tqdm(loader, desc="eval-loss", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        image_embeds, text_embeds = model(images, input_ids)
        loss = criterion(image_embeds, text_embeds)
        meter.update(loss.item(), images.size(0))
    return meter.avg


@torch.no_grad()
def evaluate_retrieval(model, loader, device, topk=(1, 3, 5, 10)):
    model.eval()
    all_image_embeds = []
    all_text_embeds = []
    all_image_ids = []

    for batch in tqdm(loader, desc="eval-retrieval", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        image_embeds, text_embeds = model(images, input_ids)
        all_image_embeds.append(image_embeds.cpu())
        all_text_embeds.append(text_embeds.cpu())
        all_image_ids.extend(batch["image_id"])

    if not all_image_embeds:
        return {f"t2i_R@{k}": 0.0 for k in topk} | {f"i2t_R@{k}": 0.0 for k in topk}

    image_embeds = F.normalize(torch.cat(all_image_embeds, dim=0), dim=-1)
    text_embeds = F.normalize(torch.cat(all_text_embeds, dim=0), dim=-1)

    # Group by image_id.
    # Keep one visual embedding per unique image and map each caption to its GT image index.
    unique_image_embeds = []
    imageid_to_unique_idx = {}
    caption_to_image_idx = []
    image_to_caption_indices = defaultdict(list)

    for caption_idx, image_id in enumerate(all_image_ids):
        if image_id not in imageid_to_unique_idx:
            imageid_to_unique_idx[image_id] = len(unique_image_embeds)
            unique_image_embeds.append(image_embeds[caption_idx])
        gt_image_idx = imageid_to_unique_idx[image_id]
        caption_to_image_idx.append(gt_image_idx)
        image_to_caption_indices[gt_image_idx].append(caption_idx)

    unique_image_embeds = torch.stack(unique_image_embeds, dim=0)
    caption_to_image_idx = torch.tensor(caption_to_image_idx, dtype=torch.long)
    num_images = unique_image_embeds.size(0)
    num_captions = text_embeds.size(0)

    metrics = {}

    # Text -> Image retrieval
    sim_t2i = text_embeds @ unique_image_embeds.t()
    for k in topk:
        kk = min(k, num_images)
        top_indices = sim_t2i.topk(kk, dim=1).indices
        correct = top_indices.eq(caption_to_image_idx.unsqueeze(1)).any(dim=1).float().mean().item()
        metrics[f"t2i_R@{k}"] = correct

    # Image -> Text retrieval
    sim_i2t = unique_image_embeds @ text_embeds.t()
    for k in topk:
        kk = min(k, num_captions)
        top_indices = sim_i2t.topk(kk, dim=1).indices
        hit = []
        for image_idx in range(num_images):
            gt_caption_indices = set(image_to_caption_indices[image_idx])
            pred_caption_indices = set(top_indices[image_idx].tolist())
            hit.append(1.0 if gt_caption_indices.intersection(pred_caption_indices) else 0.0)
        metrics[f"i2t_R@{k}"] = sum(hit) / max(1, len(hit))

    return metrics


def save_checkpoint(path, model, criterion, optimizer, tokenizer, epoch, metrics, args, best_val_r1=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "criterion": criterion.state_dict(),
            "optimizer": optimizer.state_dict(),
            "tokenizer_word2idx": tokenizer.word2idx,
            "metrics": metrics,
            "best_val_r1": best_val_r1,
            "args": vars(args),
        },
        path,
    )


def torch_load_checkpoint(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def apply_resume_model_args(args, checkpoint):
    checkpoint_args = checkpoint.get("args", {})
    for name in ("text_encoder", "embed_dim", "image_width", "max_len", "pool_type"):
        if name in checkpoint_args and getattr(args, name) != checkpoint_args[name]:
            print(
                f"resume overrides --{name.replace('_', '-')}={getattr(args, name)} "
                f"with checkpoint value {checkpoint_args[name]}"
            )
            setattr(args, name, checkpoint_args[name])


def restore_tokenizer(tokenizer, checkpoint):
    word2idx = checkpoint.get("tokenizer_word2idx")
    if not word2idx:
        return tokenizer
    tokenizer.word2idx = dict(word2idx)
    tokenizer.idx2word = [""] * len(tokenizer.word2idx)
    for word, idx in tokenizer.word2idx.items():
        tokenizer.idx2word[idx] = word
    return tokenizer


def load_training_state(checkpoint, model, criterion, optimizer, device):
    model.load_state_dict(checkpoint["model"])
    criterion.load_state_dict(checkpoint["criterion"])
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)


def main():
    args = parse_args()
    args = apply_optimized_defaults(args)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = torch_load_checkpoint(args.resume, map_location="cpu")
        apply_resume_model_args(args, resume_checkpoint)

    tokenizer, train_loader, val_loader, test_loader = build_dataloaders(args)
    if resume_checkpoint is not None:
        tokenizer = restore_tokenizer(tokenizer, resume_checkpoint)

    model = SigLIPModel(
        vocab_size=len(tokenizer),
        embed_dim=args.embed_dim,
        image_width=args.image_width,
        text_encoder=args.text_encoder,
        max_len=args.max_len,
        pool_type=args.pool_type,
    ).to(device)
    if args.custom_init:
        init_weights(model)
    criterion = SigLIPLoss().to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print(
        f"device={device} vocab_size={len(tokenizer)} text_encoder={args.text_encoder} "
        f"image_width={args.image_width} pool_type={args.pool_type} data_aug={args.data_aug} optimized={args.optimized}"
    )
    start_epoch = 1
    best_val_r1 = -1.0
    if args.resume:
        load_training_state(resume_checkpoint, model, criterion, optimizer, device)
        start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1
        best_val_r1 = resume_checkpoint.get(
            "best_val_r1",
            resume_checkpoint.get("metrics", {}).get("t2i_R@1", best_val_r1),
        )
        print(
            f"resumed_from={args.resume} checkpoint_epoch={start_epoch - 1} "
            f"next_epoch={start_epoch} best_t2i_R@1={best_val_r1 * 100:.2f}"
        )

    for epoch in range(start_epoch, args.epochs + 1):
        if args.use_cosine_lr:
            cur_lr = get_lr(args.lr, epoch, args.warmup_epochs, args.epochs, args.min_lr)
            set_optimizer_lr(optimizer, cur_lr)
        train_loss = train_one_epoch(model, criterion, train_loader, optimizer, device)
        val_loss = evaluate_loss(model, criterion, val_loader, device)
        val_metrics = evaluate_retrieval(model, val_loader, device)
        val_r1 = val_metrics["t2i_R@1"]
        scale = criterion.logit_scale.exp().item()
        bias = criterion.logit_bias.item()
        cur_lr = optimizer.param_groups[0]["lr"]
        metric_text = " ".join([f"{name}={value * 100:.2f}" for name, value in val_metrics.items()])
        print(
            f"epoch={epoch:03d} lr={cur_lr:.2e} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"scale={scale:.2f} bias={bias:.2f} {metric_text}"
        )
        if val_r1 > best_val_r1:
            best_val_r1 = val_r1
            save_checkpoint(
                os.path.join(args.output_dir, "best_siglip.pt"),
                model,
                criterion,
                optimizer,
                tokenizer,
                epoch,
                val_metrics,
                args,
                best_val_r1=best_val_r1,
            )
        save_checkpoint(
            os.path.join(args.output_dir, "latest_siglip.pt"),
            model,
            criterion,
            optimizer,
            tokenizer,
            epoch,
            val_metrics,
            args,
            best_val_r1=best_val_r1,
        )

    test_loss = evaluate_loss(model, criterion, test_loader, device)
    test_metrics = evaluate_retrieval(model, test_loader, device)
    metric_text = " ".join([f"{name}={value * 100:.2f}" for name, value in test_metrics.items()])
    print(f"final test_loss={test_loss:.4f} {metric_text}")


if __name__ == "__main__":
    main()
