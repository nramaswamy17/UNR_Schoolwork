"""
lora_mobilenet.py
-----------------
Phase 1 + 2: MobileNetV3-Small baseline training on CIFAR-10
             with LoRA injection into pointwise convolutions
             and the classifier head.

Usage:
    # Train baseline (no LoRA)
    python lora_mobilenet.py --mode baseline

    # Train with LoRA (rank 4, late layers only)
    python lora_mobilenet.py --mode lora --rank 4 --placement late

    # Train with LoRA (rank 8, all eligible layers)
    python lora_mobilenet.py --mode lora --rank 8 --placement all
"""

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from torchvision.models import mobilenet_v3_small


# ─────────────────────────────────────────────
# 1. LoRA layer for nn.Conv2d (1x1 pointwise)
# ─────────────────────────────────────────────

class LoRAConv2d(nn.Module):
    """
    Wraps a frozen Conv2d with a low-rank update:
        W' = W + (B @ A) reshaped to [out, in, 1, 1]
    Only valid for 1x1 convolutions (pointwise).
    """
    def __init__(self, conv: nn.Conv2d, rank: int, alpha: float = 1.0):
        super().__init__()
        assert conv.kernel_size == (1, 1), "LoRAConv2d only supports 1x1 convolutions"

        self.conv = conv
        for p in self.conv.parameters():
            p.requires_grad = False  # freeze base weights

        out_ch, in_ch = conv.out_channels, conv.in_channels
        self.rank = rank
        self.scale = alpha / rank

        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.empty(rank, in_ch))
        self.lora_B = nn.Parameter(torch.zeros(out_ch, rank))

        # Kaiming init for A (like LoRA paper), B stays zero → ΔW=0 at init
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        # Base conv (frozen)
        base_out = self.conv(x)

        # LoRA path: compute ΔW = B @ A, reshape to conv weight shape
        delta_w = (self.lora_B @ self.lora_A)           # [out, in]
        delta_w = delta_w.view(*delta_w.shape, 1, 1)    # [out, in, 1, 1]

        lora_out = nn.functional.conv2d(
            x, delta_w.to(x.dtype),
            stride=self.conv.stride,
            padding=self.conv.padding,
            groups=self.conv.groups,
        )
        return base_out + self.scale * lora_out

    def extra_repr(self):
        o, i = self.conv.out_channels, self.conv.in_channels
        return f"in={i}, out={o}, rank={self.rank}, scale={self.scale:.3f}"


class LoRALinear(nn.Module):
    """
    Wraps a frozen Linear layer with a low-rank update:
        W' = W + scale * (B @ A)
    """
    def __init__(self, linear: nn.Linear, rank: int, alpha: float = 1.0):
        super().__init__()
        self.linear = linear
        for p in self.linear.parameters():
            p.requires_grad = False

        in_f, out_f = linear.in_features, linear.out_features
        self.rank = rank
        self.scale = alpha / rank

        self.lora_A = nn.Parameter(torch.empty(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        base_out = self.linear(x)
        lora_out = x @ self.lora_A.to(x.dtype).T @ self.lora_B.to(x.dtype).T
        return base_out + self.scale * lora_out


# ─────────────────────────────────────────────
# 2. LoRA injection into MobileNetV3
# ─────────────────────────────────────────────

def _is_pointwise_conv(m: nn.Module) -> bool:
    return (
        isinstance(m, nn.Conv2d)
        and m.kernel_size == (1, 1)
        and m.groups == 1
    )


def inject_lora(model: nn.Module, rank: int, placement: str = "all") -> nn.Module:
    """
    Replace eligible layers with their LoRA-wrapped equivalents.

    placement options:
        "all"   — all 1x1 convs + classifier Linear layers
        "late"  — only the last 3 InvertedResidual blocks + classifier
        "early" — only the first 3 InvertedResidual blocks
        "head"  — classifier Linear layers only
    """
    features = model.features  # nn.Sequential of InvertedResidual blocks
    n_blocks = len(features)

    if placement == "all":
        eligible_blocks = set(range(n_blocks))
        adapt_classifier = True
    elif placement == "late":
        eligible_blocks = set(range(n_blocks - 3, n_blocks))
        adapt_classifier = True
    elif placement == "early":
        eligible_blocks = set(range(3))
        adapt_classifier = False
    elif placement == "head":
        eligible_blocks = set()
        adapt_classifier = True
    else:
        raise ValueError(f"Unknown placement: {placement}")

    # Inject into selected feature blocks
    for idx in eligible_blocks:
        block = features[idx]
        for name, module in list(block.named_modules()):
            if _is_pointwise_conv(module):
                # Walk the path to replace in parent
                parts = name.split(".")
                parent = block
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], LoRAConv2d(module, rank=rank))

    # Inject into classifier head
    if adapt_classifier:
        classifier = model.classifier
        for i, layer in enumerate(classifier):
            if isinstance(layer, nn.Linear):
                classifier[i] = LoRALinear(layer, rank=rank)

    return model


def freeze_base(model: nn.Module):
    """Freeze all non-LoRA parameters."""
    for name, p in model.named_parameters():
        if "lora_" not in name:
            p.requires_grad = False


def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ─────────────────────────────────────────────
# 3. Data — CIFAR-10
# ─────────────────────────────────────────────

def get_cifar10_loaders(data_dir: str = "./data", batch_size: int = 128):
    # MobileNetV3 expects 224x224; CIFAR-10 is 32x32 → upscale
    train_tf = T.Compose([
        T.Resize(224),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2, 0.2),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    full_train = torchvision.datasets.CIFAR10(data_dir, train=True,
                                              download=True, transform=train_tf)
    test_set   = torchvision.datasets.CIFAR10(data_dir, train=False,
                                              download=True, transform=val_tf)

    # 90/10 train/val split
    n_val = int(0.1 * len(full_train))
    train_set, val_set = random_split(full_train, [len(full_train) - n_val, n_val],
                                      generator=torch.Generator().manual_seed(42))
    # Override val transform (random_split shares the dataset object, so wrap)
    val_set.dataset.transform = val_tf  # note: affects full dataset — fine for val

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=256,
                              shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=256,
                              shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────
# 4. Training loop
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss, correct, n = 0., 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler:
            with torch.autocast(device_type=device.type):
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += imgs.size(0)

    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0., 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += imgs.size(0)
    return total_loss / n, correct / n


# ─────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Model ──
    model = mobilenet_v3_small(weights="IMAGENET1K_V1")
    # Replace final classifier for CIFAR-10 (10 classes)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 10)

    if args.mode == "lora":
        print(f"Injecting LoRA (rank={args.rank}, placement={args.placement})")
        model = inject_lora(model, rank=args.rank, placement=args.placement)
        freeze_base(model)
    elif args.mode == "baseline":
        pass  # all params trainable
    elif args.mode == "head":
        freeze_base(model)  # freeze everything
        # Unfreeze classifier
        for p in model.classifier.parameters():
            p.requires_grad = True
    elif args.mode == "bitfit":
        freeze_base(model)
        for name, p in model.named_parameters():
            if "bias" in name:
                p.requires_grad = True

    total, trainable = count_params(model)
    print(f"Total params:     {total:,}")
    print(f"Trainable params: {trainable:,}  ({100*trainable/total:.2f}%)")

    model = model.to(device)

    # ── Data ──
    train_loader, val_loader, _ = get_cifar10_loaders(
        args.data_dir, args.batch_size
    )

    # ── Optimizer ──
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # ── Training ──
    best_val_acc, best_state = 0., None
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer,
                                          criterion, device, scaler)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train {tr_acc:.4f} | val {val_acc:.4f} | "
              f"loss {val_loss:.4f} | {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # ── Save ──
    tag = f"{args.mode}_r{args.rank}_{args.placement}" if args.mode == "lora" \
          else args.mode
    ckpt_path = save_dir / f"mobilenetv3_{tag}.pt"
    torch.save({
        "args": vars(args),
        "state_dict": best_state,
        "best_val_acc": best_val_acc,
        "total_params": total,
        "trainable_params": trainable,
    }, ckpt_path)
    print(f"\nSaved checkpoint → {ckpt_path}")
    print(f"Best val acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode",      choices=["baseline","lora","head","bitfit"],
                   default="baseline")
    p.add_argument("--rank",      type=int,   default=4)
    p.add_argument("--placement", choices=["all","late","early","head"],
                   default="all")
    p.add_argument("--epochs",    type=int,   default=30)
    p.add_argument("--batch_size",type=int,   default=128)
    p.add_argument("--lr",        type=float, default=1e-3)
    p.add_argument("--data_dir",  default="./data")
    p.add_argument("--save_dir",  default="./checkpoints")
    main(p.parse_args())