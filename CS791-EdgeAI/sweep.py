"""
sweep.py
--------
Runs all ablation experiments and saves results to a single JSONL file.

Sweep covers:
  1. Method comparison       — all 4 methods on 5 corruptions × 5 severities
  2. LoRA rank ablation      — r in {1,2,4,8,16,32} on gaussian_noise sev3
  3. Layer placement ablation— all placements on gaussian_noise sev3

Usage:
    python sweep.py                          # full sweep (~hours)
    python sweep.py --smoke                  # 2 corruptions, sev 3 only, rank {4,8}
    python sweep.py --phase method           # method comparison only
    python sweep.py --phase rank             # rank ablation only
    python sweep.py --phase placement        # placement ablation only
"""

import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import mobilenet_v3_small

from lora_mobilenet import inject_lora, freeze_base, count_params
from cifar10c_adapt import (
    load_baseline, prepare_model, get_cifar10c_loaders,
    adapt, evaluate, per_class_accuracy, ALL_CORRUPTIONS
)

# ─────────────────────────────────────────────
# Sweep configuration
# ─────────────────────────────────────────────

# Subset of corruptions covering noise / blur / weather / digital
REPRESENTATIVE_CORRUPTIONS = [
    "gaussian_noise",   # noise
    "motion_blur",      # blur
    "fog",              # weather
    "brightness",       # lighting
    "jpeg_compression", # digital
]

ALL_SEVERITIES  = [1, 2, 3, 4, 5]
ALL_METHODS     = ["lora", "head", "bitfit", "full"]
ALL_RANKS       = [1, 2, 4, 8, 16, 32]
ALL_PLACEMENTS  = ["early", "late", "all", "head"]

# Canonical condition for ablations
CANON_CORRUPTION = "gaussian_noise"
CANON_SEVERITY   = 3
CANON_PLACEMENT  = "late"
CANON_RANK       = 4


# ─────────────────────────────────────────────
# Single experiment runner
# ─────────────────────────────────────────────

def run_experiment(baseline, cfg, device, args):
    """
    cfg keys: method, rank, placement, corruption, severity, adapt_size,
              adapt_epochs, adapt_lr, batch_size, data_dir
    Returns a result dict.
    """
    print(f"\n{'─'*60}")
    print(f"  corruption={cfg['corruption']}  sev={cfg['severity']}  "
          f"method={cfg['method']}"
          + (f"  rank={cfg['rank']}  placement={cfg['placement']}"
             if cfg['method'] == 'lora' else ""))
    print(f"{'─'*60}")

    # Zero-shot (eval only, no adaptation)
    _, eval_loader = get_cifar10c_loaders(
        cfg['data_dir'], cfg['corruption'], cfg['severity'],
        adapt_size=cfg['adapt_size'], batch_size=cfg['batch_size']
    )
    zero_shot_acc = evaluate(baseline, eval_loader, device)
    print(f"  Zero-shot: {zero_shot_acc:.4f}")

    # Prepare adapted model
    model = prepare_model(baseline, cfg['method'], cfg['rank'], cfg['placement'])
    model = model.to(device)
    total, trainable = count_params(model)

    # Build fresh loaders (reproducible split)
    adapt_loader, eval_loader = get_cifar10c_loaders(
        cfg['data_dir'], cfg['corruption'], cfg['severity'],
        adapt_size=cfg['adapt_size'], batch_size=cfg['batch_size']
    )

    # Adapt
    t0 = time.time()

    class _Args:
        pass
    a = _Args()
    a.adapt_epochs = cfg['adapt_epochs']
    a.adapt_lr     = cfg['adapt_lr']

    model, best_acc, history = adapt(model, adapt_loader, eval_loader, a, device)
    wall_time = time.time() - t0

    delta = best_acc - zero_shot_acc
    print(f"  Best acc: {best_acc:.4f}  (Δ={delta:+.4f})  "
          f"trainable={trainable:,}  time={wall_time:.1f}s")

    return {
        "corruption":        cfg['corruption'],
        "severity":          cfg['severity'],
        "method":            cfg['method'],
        "rank":              cfg['rank'] if cfg['method'] == 'lora' else None,
        "placement":         cfg['placement'] if cfg['method'] == 'lora' else None,
        "adapt_size":        cfg['adapt_size'],
        "adapt_epochs":      cfg['adapt_epochs'],
        "zero_shot_acc":     zero_shot_acc,
        "best_adapted_acc":  best_acc,
        "delta":             delta,
        "total_params":      total,
        "trainable_params":  trainable,
        "trainable_pct":     100 * trainable / total,
        "wall_time_s":       wall_time,
        "history":           history,
    }


# ─────────────────────────────────────────────
# Sweep builders
# ─────────────────────────────────────────────

def build_method_sweep(smoke=False):
    corruptions = ["gaussian_noise", "fog"] if smoke else REPRESENTATIVE_CORRUPTIONS
    severities  = [3] if smoke else ALL_SEVERITIES
    configs = []
    for corr in corruptions:
        for sev in severities:
            for method in ALL_METHODS:
                configs.append({
                    "method":     method,
                    "rank":       CANON_RANK,
                    "placement":  CANON_PLACEMENT,
                    "corruption": corr,
                    "severity":   sev,
                })
    return configs


def build_rank_sweep(smoke=False):
    ranks = [4, 8] if smoke else ALL_RANKS
    return [
        {
            "method":     "lora",
            "rank":       r,
            "placement":  CANON_PLACEMENT,
            "corruption": CANON_CORRUPTION,
            "severity":   CANON_SEVERITY,
        }
        for r in ranks
    ]


def build_placement_sweep(smoke=False):
    return [
        {
            "method":     "lora",
            "rank":       CANON_RANK,
            "placement":  pl,
            "corruption": CANON_CORRUPTION,
            "severity":   CANON_SEVERITY,
        }
        for pl in ALL_PLACEMENTS
    ]


# ─────────────────────────────────────────────
# Deduplication — skip already-done experiments
# ─────────────────────────────────────────────

def load_done(out_path: Path):
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add(_cfg_key(r))
                except Exception:
                    pass
    return done


def _cfg_key(cfg):
    return (cfg['corruption'], cfg['severity'], cfg['method'],
            cfg.get('rank'), cfg.get('placement'))


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    baseline = load_baseline(args.baseline_ckpt, device)

    # Build config list
    if args.phase == "method":
        configs = build_method_sweep(args.smoke)
    elif args.phase == "rank":
        configs = build_rank_sweep(args.smoke)
    elif args.phase == "placement":
        configs = build_placement_sweep(args.smoke)
    else:  # all
        configs  = build_method_sweep(args.smoke)
        configs += build_rank_sweep(args.smoke)
        configs += build_placement_sweep(args.smoke)

    # Fill in shared hyperparams
    shared = dict(
        adapt_size   = args.adapt_size,
        adapt_epochs = args.adapt_epochs,
        adapt_lr     = args.adapt_lr,
        batch_size   = args.batch_size,
        data_dir     = args.data_dir,
    )
    for c in configs:
        c.update(shared)

    # Output file (JSONL — one result per line, resumable)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path)

    total_cfgs = len(configs)
    skipped    = sum(1 for c in configs if _cfg_key(c) in done)
    print(f"\nTotal experiments: {total_cfgs}  |  "
          f"Already done: {skipped}  |  Remaining: {total_cfgs - skipped}")

    with open(out_path, "a") as f:
        for i, cfg in enumerate(configs, 1):
            if _cfg_key(cfg) in done:
                print(f"[{i}/{total_cfgs}] SKIP {_cfg_key(cfg)}")
                continue
            print(f"\n[{i}/{total_cfgs}]")
            try:
                result = run_experiment(baseline, cfg, device, args)
                f.write(json.dumps(result) + "\n")
                f.flush()
                done.add(_cfg_key(cfg))
            except Exception as e:
                print(f"  ERROR: {e} — skipping")

    print(f"\n✓ Sweep complete. Results → {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--phase",   choices=["method","rank","placement","all"],
                   default="all")
    p.add_argument("--smoke",   action="store_true",
                   help="Quick smoke test: 2 corruptions, sev3, ranks {4,8}")
    # Paths
    p.add_argument("--baseline_ckpt", default="checkpoints/mobilenetv3_baseline.pt")
    p.add_argument("--data_dir",      default="./data")
    p.add_argument("--out",           default="results/sweep.jsonl")
    # Hyperparams
    p.add_argument("--adapt_size",    type=int,   default=500)
    p.add_argument("--adapt_epochs",  type=int,   default=10)
    p.add_argument("--adapt_lr",      type=float, default=1e-3)
    p.add_argument("--batch_size",    type=int,   default=64)
    main(p.parse_args())