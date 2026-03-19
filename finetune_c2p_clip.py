#!/usr/bin/env python3
"""
Fine-tune C2P-CLIP on OpenFake dataset (modern generators).

Architecture: CLIP ViT-L/14 backbone (frozen) + Linear(768->1) head (trainable).
Training data: OpenFake subset (21 generator families, ~4345 images).

Usage:
    python finetune_c2p_clip.py
    python finetune_c2p_clip.py --epochs 10 --lr 1e-3 --unfreeze-layers 2
    python finetune_c2p_clip.py --full  # unfreeze entire backbone (slow, needs more VRAM)
"""

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import CLIPModel, CLIPProcessor

WEIGHTS_DIR = Path(__file__).parent.parent / "weights"
OPENFAKE_DIR = Path(__file__).parent / "openfake"
RESULTS_DIR = Path(__file__).parent / "results"
IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")


class C2PClipWrapper(nn.Module):
    """C2P-CLIP: CLIP ViT-L/14 + Linear(768->1) classification head."""

    def __init__(self, clip_model):
        super().__init__()
        self.model = clip_model
        self.model.fc = nn.Linear(768, 1)

    def forward(self, pixel_values):
        outputs = self.model.vision_model(pixel_values=pixel_values)
        pooled = self.model.visual_projection(outputs.pooler_output)
        logit = self.model.fc(pooled)
        return logit


class OpenFakeDataset(Dataset):
    """Load OpenFake images with real/fake labels."""

    def __init__(self, root_dir, processor, limit_per_family=0):
        self.processor = processor
        self.samples = []

        for family_dir in sorted(root_dir.iterdir()):
            if not family_dir.is_dir():
                continue
            family = family_dir.name
            is_fake = 0 if family == "real" else 1
            files = sorted([f for f in family_dir.iterdir() if f.suffix.lower() in IMG_EXTS])
            if limit_per_family > 0:
                files = files[:limit_per_family]
            for f in files:
                self.samples.append((str(f), is_fake, family))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, family = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            inputs = self.processor(images=img, return_tensors="pt")
            pixel_values = inputs["pixel_values"].squeeze(0)
        except Exception:
            pixel_values = torch.zeros(3, 224, 224)
        return pixel_values, label, family


def run_evaluation(model, dataloader, device):
    """Run model on a dataloader. Returns metrics dict."""
    model.eval()
    all_labels = []
    all_scores = []
    all_families = []

    with torch.no_grad():
        for pixel_values, labels, families in dataloader:
            pixel_values = pixel_values.to(device)
            logits = model(pixel_values).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_labels.extend(labels.numpy())
            all_scores.extend(probs)
            all_families.extend(families)

    y_true = np.array(all_labels)
    y_scores = np.array(all_scores)

    try:
        auroc = roc_auc_score(y_true, y_scores)
    except ValueError:
        auroc = 0.0
    ap = average_precision_score(y_true, y_scores)
    acc = accuracy_score(y_true, y_scores > 0.5)

    # Per-family AUROC
    families_arr = np.array(all_families)
    real_mask = families_arr == "real"
    per_family = {}
    for fam in sorted(set(all_families) - {"real"}):
        fam_mask = families_arr == fam
        combined = real_mask | fam_mask
        yt = y_true[combined]
        ys = y_scores[combined]
        if len(np.unique(yt)) < 2:
            continue
        try:
            per_family[fam] = roc_auc_score(yt, ys)
        except ValueError:
            per_family[fam] = 0.0

    return {"auroc": auroc, "ap": ap, "accuracy": acc, "per_family": per_family}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--unfreeze-layers", type=int, default=0,
                        help="Number of ViT encoder layers to unfreeze from the top (0=head only)")
    parser.add_argument("--full", action="store_true",
                        help="Unfreeze entire backbone (needs ~12GB VRAM)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit images per generator family")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load base CLIP model
    clip_cache = str(WEIGHTS_DIR / "clip-vit-large-patch14")
    print("Loading CLIP ViT-L/14...")
    base_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", cache_dir=clip_cache)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", cache_dir=clip_cache)

    # Create wrapper and load pretrained C2P-CLIP weights
    model = C2PClipWrapper(base_model)
    pretrained_path = WEIGHTS_DIR / "c2p_clip_genimage.pth"
    if pretrained_path.exists():
        print(f"Loading pretrained C2P-CLIP weights from {pretrained_path}")
        state = torch.load(str(pretrained_path), map_location="cpu", weights_only=False)
        if isinstance(state, dict) and any(k.startswith("model.") for k in state.keys()):
            model.load_state_dict(state, strict=False)
        elif isinstance(state, dict) and "model" in state:
            model.load_state_dict(state["model"], strict=False)
        else:
            model.load_state_dict(state, strict=False)
        print("  Pretrained weights loaded successfully")
    else:
        print("  WARNING: No pretrained weights found, training from scratch")

    model.to(device)

    # Freeze backbone by default
    for param in model.model.vision_model.parameters():
        param.requires_grad = False
    for param in model.model.visual_projection.parameters():
        param.requires_grad = False

    # Unfreeze top N ViT layers if requested
    if args.full:
        print("Unfreezing ENTIRE backbone")
        for param in model.parameters():
            param.requires_grad = True
    elif args.unfreeze_layers > 0:
        layers = model.model.vision_model.encoder.layers
        n_layers = len(layers)
        unfreeze_from = max(0, n_layers - args.unfreeze_layers)
        print(f"Unfreezing top {args.unfreeze_layers} ViT layers (layers {unfreeze_from}-{n_layers-1})")
        for i in range(unfreeze_from, n_layers):
            for param in layers[i].parameters():
                param.requires_grad = True

    # Classification head is always trainable
    for param in model.model.fc.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

    # Load dataset
    print(f"\nLoading OpenFake dataset from {OPENFAKE_DIR}")
    dataset = OpenFakeDataset(OPENFAKE_DIR, processor, limit_per_family=args.limit)
    print(f"Total images: {len(dataset)}")

    families = defaultdict(int)
    for _, label, fam in dataset.samples:
        families[fam] += 1
    for fam, count in sorted(families.items()):
        tag = "real" if fam == "real" else "fake"
        print(f"  {fam:<20s} ({tag}): {count}")

    # Train/val split
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"\nTrain: {train_size}, Val: {val_size}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Optimizer + scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_auroc = 0.0
    best_epoch = 0
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "c2p_clip_finetuned.pth"

    print(f"\nTraining for {args.epochs} epochs, lr={args.lr}")
    print("=" * 60)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0

        for pixel_values, labels, _ in train_loader:
            pixel_values = pixel_values.to(device)
            labels = labels.float().to(device)

            logits = model(pixel_values).squeeze(-1)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = train_loss / n_batches

        # Validate
        metrics = run_evaluation(model, val_loader, device)

        lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1:2d}/{args.epochs} | loss={avg_loss:.4f} | "
              f"val_AUROC={metrics['auroc']:.4f} | val_AP={metrics['ap']:.4f} | "
              f"val_Acc={metrics['accuracy']:.4f} | lr={lr:.2e}")

        # Save best model
        if metrics["auroc"] > best_auroc:
            best_auroc = metrics["auroc"]
            best_epoch = epoch + 1
            torch.save(model.state_dict(), str(output_path))
            print(f"  >> New best! Saved to {output_path}")

    print(f"\n{'='*60}")
    print(f"Best: epoch {best_epoch}, AUROC={best_auroc:.4f}")
    print(f"Weights saved to: {output_path}")

    # Final evaluation
    print(f"\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(str(output_path), map_location=device, weights_only=False))
    final_metrics = run_evaluation(model, val_loader, device)
    print(f"\nFinal validation metrics:")
    print(f"  AUROC:    {final_metrics['auroc']:.4f}")
    print(f"  AP:       {final_metrics['ap']:.4f}")
    print(f"  Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"\n  Per-generator AUROC:")
    for fam, auroc_val in sorted(final_metrics["per_family"].items()):
        print(f"    {fam:<20s} {auroc_val:.4f}")

    # Save log
    log_path = RESULTS_DIR / "c2p_clip_finetune_log.json"
    with open(log_path, "w") as f:
        json.dump({
            "best_epoch": best_epoch,
            "best_auroc": best_auroc,
            "final_metrics": {k: v for k, v in final_metrics.items() if k != "per_family"},
            "per_family": final_metrics.get("per_family", {}),
            "args": vars(args),
        }, f, indent=2)
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()
