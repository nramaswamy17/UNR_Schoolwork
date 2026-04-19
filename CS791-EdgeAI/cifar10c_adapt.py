"""
cifar10c_adapt.py
-----------------
Phase 3: Domain shift adaptation on CIFAR-10C.

Loads the saved baseline checkpoint, injects the chosen adaptation method,
adapts on a held-out split, and evaluates on a disjoint eval split.

Usage:
    # LoRA adaptation, rank 4, late layers, severity 3, corruption=gaussian_noise
    python cifar10c_adapt.py --method lora --rank 4 --placement late \
        --corruption gaussian_noise --severity 3

    # Last-layer only
    python cifar10c_adapt.py --method head --corruption motion_blur --severity 5

    # BitFit
    python cifar10c_adapt.py --method bitfit --corruption brightness --severity 2

    # Full fine-tune
    python cifar10c_adapt.py --method full --corruption fog --severity 4

Download CIFAR-10C from:
    https://zenodo.org/record/2535967/files/CIFAR-10-C.tar
    tar -xf CIFAR-10-C.tar → produces ./data/CIFAR-10-C/*.npy
"""

import argparse
import copy
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.models import mobilenet_v3_small

# Reuse LoRA classes from lora_mobilenet.py
# (copy them here or import if in same package)
# ── paste / import LoRAConv2d, LoRALinear, inject_lora, freeze_base ──
from lora_mobilenet import (LoRAConv2d, LoRALinear,   # noqa: F401
                             inject_lora, freeze_base,
                             count_params)


# ─────────────────────────────────────────────
# CIFAR-10C corruptions available
# ─────────────────────────────────────────────

ALL_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness", "contrast",
    "elastic_transform", "pixelate", "jpeg_compression",
]

CIFAR10_CLASSES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class CIFAR10CDataset(Dataset):
    """
    Loads a single corruption type at a given severity level from CIFAR-10C .npy files.
    CIFAR-10C stores 10,000 images per severity (5 severities → 50,000 total per file).
    """
    def __init__(self, data_dir: str, corruption: str, severity: int,
                 transform=None):
        assert 1 <= severity <= 5, "Severity must be 1–5"
        data_dir = Path(data_dir)

        imgs_path   = data_dir / "CIFAR-10-C" / f"{corruption}.npy"
        labels_path = data_dir / "CIFAR-10-C" / "labels.npy"

        if not imgs_path.exists():
            raise FileNotFoundError(
                f"Could not find {imgs_path}\n"
                "Download CIFAR-10-C from https://zenodo.org/record/2535967"
            )

        all_imgs   = np.load(imgs_path)   # (50000, 32, 32, 3) uint8
        all_labels = np.load(labels_path) # (50000,) int64

        # Each severity occupies 10,000 consecutive samples
        start = (severity - 1) * 10_000
        end   = severity * 10_000
        self.imgs   = all_imgs[start:end]
        self.labels = all_labels[start:end]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.imgs[idx]  # HWC uint8
        # PIL-free path: convert numpy → tensor directly
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        if self.transform:
            img = self.transform(img)
        return img, int(self.labels[idx])


def get_cifar10c_loaders(data_dir, corruption, severity,
                          adapt_size, batch_size, seed=42):
    """
    Returns (adapt_loader, eval_loader) with disjoint samples.
    adapt_size: number of samples used for on-device tuning.
    eval_size:  remainder (up to 10,000 - adapt_size).
    """
    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # Upsample to 224 to match MobileNetV3 training resolution
    img_tf = T.Compose([T.Resize(224), normalize])

    full_ds = CIFAR10CDataset(data_dir, corruption, severity, transform=img_tf)

    n_total  = len(full_ds)          # 10,000
    n_adapt  = min(adapt_size, n_total // 2)
    n_eval   = n_total - n_adapt

    adapt_ds, eval_ds = random_split(
        full_ds, [n_adapt, n_eval],
        generator=torch.Generator().manual_seed(seed)
    )

    adapt_loader = DataLoader(adapt_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    eval_loader  = DataLoader(eval_ds,  batch_size=256,
                              shuffle=False, num_workers=4, pin_memory=True)
    return adapt_loader, eval_loader


# ─────────────────────────────────────────────
# Model loading helpers
# ─────────────────────────────────────────────

def load_baseline(ckpt_path: str, device) -> nn.Module:
    """Load the saved baseline MobileNetV3 checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device)
    model = mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 10)
    model.load_state_dict(ckpt["state_dict"])
    print(f"Loaded baseline — val acc: {ckpt['best_val_acc']:.4f}")
    return model.to(device)


def prepare_model(model: nn.Module, method: str, rank: int,
                  placement: str) -> nn.Module:
    """Apply adaptation strategy to a copy of the model."""
    model = copy.deepcopy(model)  # don't mutate the original

    if method == "lora":
        model = inject_lora(model, rank=rank, placement=placement)
        freeze_base(model)
    elif method == "head":
        for p in model.parameters():
            p.requires_grad = False
        for p in model.classifier.parameters():
            p.requires_grad = True
    elif method == "bitfit":
        for p in model.parameters():
            p.requires_grad = False
        for name, p in model.named_parameters():
            if "bias" in name:
                p.requires_grad = True
    elif method == "full":
        pass  # all params trainable
    else:
        raise ValueError(f"Unknown method: {method}")

    return model


# ─────────────────────────────────────────────
# Adaptation loop
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, n = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(1)
        correct += (preds == labels).sum().item()
        n += imgs.size(0)
    return correct / n


def adapt(model, adapt_loader, eval_loader, args, device):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.adapt_lr, weight_decay=1e-4
    )
    scaler = None  # AMP disabled — conflicts with manual conv2d in LoRAConv2d

    best_acc, best_state = 0., None
    history = []

    for epoch in range(1, args.adapt_epochs + 1):
        model.train()
        t0 = time.time()
        for imgs, labels in adapt_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            if scaler:
                with torch.autocast(device_type=device.type):
                    loss = criterion(model(imgs), labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = criterion(model(imgs), labels)
                loss.backward()
                optimizer.step()

        val_acc = evaluate(model, eval_loader, device)
        elapsed = time.time() - t0
        history.append({"epoch": epoch, "eval_acc": val_acc})
        print(f"  Adapt epoch {epoch:2d}/{args.adapt_epochs} | "
              f"eval acc {val_acc:.4f} | {elapsed:.1f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_acc, history


# ─────────────────────────────────────────────
# Per-class accuracy breakdown
# ─────────────────────────────────────────────

@torch.no_grad()
def per_class_accuracy(model, loader, device):
    model.eval()
    correct = torch.zeros(10)
    total   = torch.zeros(10)
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).argmax(1)
        for c in range(10):
            mask = labels == c
            correct[c] += (preds[mask] == c).sum().item()
            total[c]   += mask.sum().item()
    return {CIFAR10_CLASSES[i]: (correct[i] / total[i]).item()
            for i in range(10)}


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Corruption: {args.corruption} | Severity: {args.severity} | "
          f"Method: {args.method}" +
          (f" | rank={args.rank} placement={args.placement}"
           if args.method == "lora" else ""))

    # ── Load baseline ──
    baseline = load_baseline(args.baseline_ckpt, device)

    # ── Zero-shot evaluation (before adaptation) ──
    _, eval_loader = get_cifar10c_loaders(
        args.data_dir, args.corruption, args.severity,
        adapt_size=args.adapt_size, batch_size=args.batch_size
    )
    zero_shot_acc = evaluate(baseline, eval_loader, device)
    print(f"Zero-shot acc (no adaptation): {zero_shot_acc:.4f}")

    # ── Prepare model for adaptation ──
    model = prepare_model(baseline, args.method, args.rank, args.placement)
    model = model.to(device)  # re-send after injection; new LoRA params init on CPU
    total, trainable = count_params(model)
    print(f"Trainable params: {trainable:,} / {total:,} "
          f"({100*trainable/total:.2f}%)")

    # ── Build adapt / eval loaders ──
    adapt_loader, eval_loader = get_cifar10c_loaders(
        args.data_dir, args.corruption, args.severity,
        adapt_size=args.adapt_size, batch_size=args.batch_size
    )
    print(f"Adapt set: {len(adapt_loader.dataset):,} samples | "
          f"Eval set: {len(eval_loader.dataset):,} samples")

    # ── Adapt ──
    model, best_acc, history = adapt(
        model, adapt_loader, eval_loader, args, device
    )
    print(f"\nBest adapted eval acc: {best_acc:.4f}  "
          f"(Δ from zero-shot: {best_acc - zero_shot_acc:+.4f})")

    # ── Per-class breakdown ──
    cls_acc = per_class_accuracy(model, eval_loader, device)
    print("\nPer-class accuracy:")
    for cls, acc in cls_acc.items():
        print(f"  {cls:12s}: {acc:.4f}")

    # ── Save result ──
    tag = (f"{args.method}_r{args.rank}_{args.placement}"
           if args.method == "lora" else args.method)
    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "corruption":      args.corruption,
        "severity":        args.severity,
        "method":          args.method,
        "rank":            args.rank if args.method == "lora" else None,
        "placement":       args.placement if args.method == "lora" else None,
        "adapt_size":      args.adapt_size,
        "zero_shot_acc":   zero_shot_acc,
        "best_adapted_acc": best_acc,
        "delta":           best_acc - zero_shot_acc,
        "total_params":    total,
        "trainable_params": trainable,
        "per_class":       cls_acc,
        "history":         history,
    }
    import json
    result_path = out_dir / f"{args.corruption}_sev{args.severity}_{tag}.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResult saved → {result_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # Adaptation method
    p.add_argument("--method",    choices=["lora","head","bitfit","full"],
                   default="lora")
    p.add_argument("--rank",      type=int, default=4)
    p.add_argument("--placement", choices=["all","late","early","head"],
                   default="late")
    # Domain shift
    p.add_argument("--corruption", choices=ALL_CORRUPTIONS,
                   default="gaussian_noise")
    p.add_argument("--severity",   type=int, choices=[1,2,3,4,5], default=3)
    # Data
    p.add_argument("--adapt_size", type=int, default=500,
                   help="Samples used for adaptation (rest go to eval)")
    p.add_argument("--batch_size", type=int, default=64)
    # Training
    p.add_argument("--adapt_epochs", type=int,   default=10)
    p.add_argument("--adapt_lr",     type=float, default=1e-3)
    # Paths
    p.add_argument("--baseline_ckpt", default="checkpoints/mobilenetv3_baseline.pt")
    p.add_argument("--data_dir",      default="./data")
    p.add_argument("--save_dir",      default="./results")
    main(p.parse_args())