#!/usr/bin/env python3
"""
Evaluate pretrained vs finetuned C2P-CLIP on OpenFake dataset.
Computes AUROC, AP, Accuracy overall and per-generator-family AUROC.
"""

import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor

WEIGHTS_DIR = Path(__file__).parent.parent / "weights"
OPENFAKE_DIR = Path(__file__).parent / "openfake"
IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")

PRETRAINED_PATH = WEIGHTS_DIR / "c2p_clip_genimage.pth"
FINETUNED_PATH = Path(__file__).parent / "results" / "c2p_clip_finetuned.pth"


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

    def __init__(self, root_dir, processor):
        self.processor = processor
        self.samples = []
        for family_dir in sorted(root_dir.iterdir()):
            if not family_dir.is_dir():
                continue
            family = family_dir.name
            is_fake = 0 if family == "real" else 1
            files = sorted([f for f in family_dir.iterdir() if f.suffix.lower() in IMG_EXTS])
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
    """Run model on dataloader, return metrics."""
    model.eval()
    all_labels, all_scores, all_families = [], [], []

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

    auroc = roc_auc_score(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    acc = accuracy_score(y_true, y_scores > 0.5)

    # Per-family AUROC (each family vs real)
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
        per_family[fam] = roc_auc_score(yt, ys)

    return {"auroc": auroc, "ap": ap, "accuracy": acc, "per_family": per_family}


def load_model(clip_model_base, weights_path, device):
    """Create C2PClipWrapper and load weights."""
    import copy
    clip_copy = copy.deepcopy(clip_model_base)
    model = C2PClipWrapper(clip_copy)
    state = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=False)
    model.to(device)
    return model


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load base CLIP
    clip_cache = str(WEIGHTS_DIR / "clip-vit-large-patch14")
    print("Loading CLIP ViT-L/14 base...")
    base_clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", cache_dir=clip_cache)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", cache_dir=clip_cache)

    # Load dataset
    print(f"Loading OpenFake dataset from {OPENFAKE_DIR}")
    dataset = OpenFakeDataset(OPENFAKE_DIR, processor)
    print(f"Total images: {len(dataset)}")

    families = defaultdict(int)
    for _, label, fam in dataset.samples:
        families[fam] += 1
    n_real = families.pop("real", 0)
    n_fake = sum(families.values())
    print(f"  Real: {n_real}, Fake: {n_fake} ({len(families)} generators)")
    for fam, count in sorted(families.items()):
        print(f"    {fam:<25s} {count:4d}")

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    # --- Evaluate pretrained ---
    print(f"\n{'='*70}")
    print("Evaluating PRETRAINED C2P-CLIP (c2p_clip_genimage.pth)")
    print(f"{'='*70}")
    pretrained_model = load_model(base_clip, PRETRAINED_PATH, device)
    t0 = time.time()
    pretrained_metrics = run_evaluation(pretrained_model, dataloader, device)
    t_pre = time.time() - t0
    print(f"  Time: {t_pre:.1f}s")
    del pretrained_model
    torch.cuda.empty_cache()

    # --- Evaluate finetuned ---
    print(f"\n{'='*70}")
    print("Evaluating FINETUNED C2P-CLIP (c2p_clip_finetuned.pth)")
    print(f"{'='*70}")
    finetuned_model = load_model(base_clip, FINETUNED_PATH, device)
    t0 = time.time()
    finetuned_metrics = run_evaluation(finetuned_model, dataloader, device)
    t_ft = time.time() - t0
    print(f"  Time: {t_ft:.1f}s")
    del finetuned_model
    torch.cuda.empty_cache()

    # --- Comparison table ---
    print(f"\n{'='*70}")
    print("COMPARISON: Pretrained vs Finetuned C2P-CLIP on OpenFake")
    print(f"{'='*70}")
    print(f"{'Metric':<20s} {'Pretrained':>12s} {'Finetuned':>12s} {'Delta':>10s}")
    print("-" * 56)

    for metric_name in ["auroc", "ap", "accuracy"]:
        pre_val = pretrained_metrics[metric_name]
        ft_val = finetuned_metrics[metric_name]
        delta = ft_val - pre_val
        sign = "+" if delta >= 0 else ""
        print(f"{metric_name.upper():<20s} {pre_val:>12.4f} {ft_val:>12.4f} {sign}{delta:>9.4f}")

    print(f"\n{'Per-Generator AUROC':<20s} {'Pretrained':>12s} {'Finetuned':>12s} {'Delta':>10s}")
    print("-" * 56)

    all_families_sorted = sorted(
        set(list(pretrained_metrics["per_family"].keys()) +
            list(finetuned_metrics["per_family"].keys()))
    )
    for fam in all_families_sorted:
        pre_val = pretrained_metrics["per_family"].get(fam, float("nan"))
        ft_val = finetuned_metrics["per_family"].get(fam, float("nan"))
        delta = ft_val - pre_val
        sign = "+" if delta >= 0 else ""
        print(f"  {fam:<18s} {pre_val:>12.4f} {ft_val:>12.4f} {sign}{delta:>9.4f}")

    # Macro average per-family AUROC
    pre_family_vals = [v for v in pretrained_metrics["per_family"].values() if not np.isnan(v)]
    ft_family_vals = [v for v in finetuned_metrics["per_family"].values() if not np.isnan(v)]
    pre_macro = np.mean(pre_family_vals) if pre_family_vals else 0
    ft_macro = np.mean(ft_family_vals) if ft_family_vals else 0
    delta_macro = ft_macro - pre_macro
    sign = "+" if delta_macro >= 0 else ""
    print("-" * 56)
    print(f"  {'MACRO AVG':<18s} {pre_macro:>12.4f} {ft_macro:>12.4f} {sign}{delta_macro:>9.4f}")

    # Winners/losers
    print(f"\n{'='*70}")
    improved = [f for f in all_families_sorted
                if finetuned_metrics["per_family"].get(f, 0) > pretrained_metrics["per_family"].get(f, 0)]
    degraded = [f for f in all_families_sorted
                if finetuned_metrics["per_family"].get(f, 0) < pretrained_metrics["per_family"].get(f, 0)]
    print(f"Improved ({len(improved)}): {', '.join(improved)}")
    print(f"Degraded ({len(degraded)}): {', '.join(degraded)}")


if __name__ == "__main__":
    main()
