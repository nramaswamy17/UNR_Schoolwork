"""
benchmark.py
------------
Phase 5: Edge resource profiling for all adaptation methods.

Measures per-method:
  - Inference latency (batch=1, CPU, fixed threads)
  - Peak RAM during inference
  - Peak RAM during one adaptation step (backprop + optimizer)
  - Adapter parameter count + storage size on disk
  - Adaptation time per epoch

Usage:
    python benchmark.py                        # all methods + ranks
    python benchmark.py --methods lora head    # subset
"""

import argparse
import copy
import gc
import json
import os
import tempfile
import time
import tracemalloc
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import mobilenet_v3_small

from lora_mobilenet import inject_lora, freeze_base, count_params
from cifar10c_adapt import load_baseline, prepare_model

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

LORA_RANKS      = [1, 2, 4, 8, 16, 32]
LORA_PLACEMENT  = "late"
METHODS         = ["lora", "head", "bitfit", "full"]
CPU_THREADS     = 1   # simulate single-core edge device
N_WARMUP        = 10  # warmup forward passes before timing
N_MEASURE       = 50  # timed forward passes


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def make_dummy_batch(batch_size=1, device=torch.device("cpu")):
    """224x224 dummy input matching MobileNetV3 training resolution."""
    return torch.randn(batch_size, 3, 224, 224, device=device)


def adapter_storage_bytes(model: nn.Module) -> int:
    """Save only trainable params to a temp file, return file size in bytes."""
    trainable = {k: v for k, v in model.state_dict().items()
                 if any(k == n for n, p in model.named_parameters() if p.requires_grad)}
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp = f.name
    torch.save(trainable, tmp)
    size = os.path.getsize(tmp)
    os.unlink(tmp)
    return size


# ─────────────────────────────────────────────
# Latency benchmark (CPU, batch=1)
# ─────────────────────────────────────────────

def measure_latency(model: nn.Module, n_warmup=N_WARMUP, n_measure=N_MEASURE):
    """
    CPU-only, batch=1, fixed thread count.
    Returns (mean_ms, std_ms).
    """
    torch.set_num_threads(CPU_THREADS)
    model = model.cpu().eval()
    x = make_dummy_batch(batch_size=1, device=torch.device("cpu"))

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)

    times = []
    with torch.no_grad():
        for _ in range(n_measure):
            t0 = time.perf_counter()
            _ = model(x)
            times.append((time.perf_counter() - t0) * 1000)  # ms

    import statistics
    return statistics.mean(times), statistics.stdev(times)


# ─────────────────────────────────────────────
# Peak RAM — inference (tracemalloc, CPU)
# ─────────────────────────────────────────────

def measure_inference_ram(model: nn.Module):
    """Peak RAM (bytes) during a single forward pass on CPU."""
    model = model.cpu().eval()
    x = make_dummy_batch(batch_size=1, device=torch.device("cpu"))

    gc.collect()
    tracemalloc.start()
    with torch.no_grad():
        _ = model(x)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak  # bytes


# ─────────────────────────────────────────────
# Peak RAM — adaptation step (tracemalloc, CPU)
# ─────────────────────────────────────────────

def measure_adapt_ram(model: nn.Module):
    """
    Peak RAM (bytes) during one forward + backward + optimizer step on CPU.
    Uses a single dummy batch to simulate on-device adaptation.
    """
    model = model.cpu().train()
    x      = make_dummy_batch(batch_size=32, device=torch.device("cpu"))
    labels = torch.randint(0, 10, (32,))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )

    gc.collect()
    tracemalloc.start()
    optimizer.zero_grad()
    loss = criterion(model(x), labels)
    loss.backward()
    optimizer.step()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak  # bytes


# ─────────────────────────────────────────────
# Adaptation time per epoch
# ─────────────────────────────────────────────

def measure_adapt_time_per_epoch(model: nn.Module, n_batches=10, batch_size=32):
    """
    Wall-clock time for one epoch of adaptation on CPU.
    Uses dummy data; n_batches simulates a small on-device dataset.
    """
    model = model.cpu().train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )
    xs = [make_dummy_batch(batch_size) for _ in range(n_batches)]
    ys = [torch.randint(0, 10, (batch_size,)) for _ in range(n_batches)]

    t0 = time.perf_counter()
    for x, y in zip(xs, ys):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    return (time.perf_counter() - t0)  # seconds


# ─────────────────────────────────────────────
# Profile one model config
# ─────────────────────────────────────────────

def profile_model(label, model, total_params, trainable_params):
    print(f"  Profiling: {label}")

    lat_mean, lat_std = measure_latency(model)
    print(f"    Latency:        {lat_mean:.2f} ± {lat_std:.2f} ms")

    inf_ram = measure_inference_ram(model)
    print(f"    Inference RAM:  {inf_ram/1e6:.2f} MB")

    adapt_ram = measure_adapt_ram(model)
    print(f"    Adapt RAM:      {adapt_ram/1e6:.2f} MB")

    adapt_time = measure_adapt_time_per_epoch(model)
    print(f"    Adapt time/ep:  {adapt_time:.2f}s (10 batches × 32)")

    storage = adapter_storage_bytes(model)
    print(f"    Adapter storage:{storage/1e3:.1f} KB")

    return {
        "label":             label,
        "total_params":      total_params,
        "trainable_params":  trainable_params,
        "trainable_pct":     100 * trainable_params / total_params,
        "latency_mean_ms":   lat_mean,
        "latency_std_ms":    lat_std,
        "inference_ram_mb":  inf_ram / 1e6,
        "adapt_ram_mb":      adapt_ram / 1e6,
        "adapt_time_s":      adapt_time,
        "adapter_storage_kb": storage / 1e3,
    }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main(args):
    device = torch.device("cpu")  # edge benchmark is always CPU
    print("Edge benchmark — CPU only, single thread")

    baseline = load_baseline(args.baseline_ckpt, device)
    results  = []

    # ── Method comparison ──
    for method in args.methods:
        rank      = LORA_RANKS[2]  # rank 4 as canonical for method comparison
        placement = LORA_PLACEMENT
        model     = prepare_model(baseline, method, rank, placement).to(device)
        total, trainable = count_params(model)
        label = (f"lora_r{rank}_{placement}" if method == "lora" else method)
        results.append(profile_model(label, model, total, trainable))
        del model; gc.collect()

    # ── LoRA rank sweep ──
    if "lora" in args.methods:
        print("\nLoRA rank sweep:")
        for r in LORA_RANKS:
            model = prepare_model(baseline, "lora", r, LORA_PLACEMENT).to(device)
            total, trainable = count_params(model)
            label = f"lora_r{r}_{LORA_PLACEMENT}"
            if not any(res["label"] == label for res in results):
                results.append(profile_model(label, model, total, trainable))
            del model; gc.collect()

        print("\nLoRA placement sweep (rank=4):")
        for placement in ["early", "all", "head"]:  # "late" already done above
            model = prepare_model(baseline, "lora", LORA_RANKS[2], placement).to(device)
            total, trainable = count_params(model)
            label = f"lora_r{LORA_RANKS[2]}_{placement}"
            if not any(res["label"] == label for res in results):
                results.append(profile_model(label, model, total, trainable))
            del model; gc.collect()

    # ── Save ──
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Benchmark results → {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+",
                   default=METHODS,
                   choices=METHODS)
    p.add_argument("--baseline_ckpt", default="checkpoints/mobilenetv3_baseline.pt")
    p.add_argument("--out",           default="results/benchmark.json")
    main(p.parse_args())