"""
analyze.py
----------
Phase 6: Load sweep.jsonl + benchmark.json, generate all paper figures.

Produces:
  figures/fig1_accuracy_vs_params.pdf       — Pareto: acc vs trainable params
  figures/fig2_accuracy_vs_latency.pdf      — Pareto: acc vs inference latency
  figures/fig3_severity_curves.pdf          — acc vs severity per method
  figures/fig4_rank_sweep.pdf               — acc + resource vs LoRA rank
  figures/fig5_placement_heatmap.pdf        — placement × corruption heatmap
  figures/fig6_corruption_breakdown.pdf     — per-corruption bar chart
  figures/fig7_per_class.pdf                — per-class accuracy radar (optional)

Usage:
    python analyze.py
    python analyze.py --sweep results/sweep.jsonl --bench results/benchmark.json
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────

def load_sweep(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    # Friendly label column
    def label(row):
        if row["method"] == "lora":
            return f"LoRA-r{int(row['rank'])} ({row['placement']})"
        return {"head": "Last-layer", "bitfit": "BitFit",
                "full": "Full FT"}[row["method"]]
    df["label"] = df.apply(label, axis=1)
    return df


def load_bench(path):
    with open(path) as f:
        data = json.load(f)
    return pd.DataFrame(data)


# ─────────────────────────────────────────────
# Plot style
# ─────────────────────────────────────────────

COLORS = {
    "lora":   "#2196F3",
    "head":   "#FF9800",
    "bitfit": "#4CAF50",
    "full":   "#F44336",
}
METHOD_LABELS = {
    "lora":   "LoRA",
    "head":   "Last-layer",
    "bitfit": "BitFit",
    "full":   "Full FT",
}

plt.rcParams.update({
    "font.family":    "sans-serif",
    "font.size":      11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":     150,
})


def savefig(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────
# Fig 1: Pareto — accuracy vs trainable params
# ─────────────────────────────────────────────

def fig_pareto_params(df, bench_df, out):
    """
    X: trainable params (from benchmark)
    Y: mean adapted accuracy across all conditions (from sweep)
    One point per method; LoRA gets one point per rank.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    # Mean acc per (method, rank, placement) averaged over corruption × severity
    grp = df.groupby(["method", "rank", "placement"])["best_adapted_acc"].mean().reset_index()

    for _, row in grp.iterrows():
        method = row["method"]
        if method == "lora":
            lbl = f"lora_r{int(row['rank'])}_{row['placement']}"
        else:
            lbl = method
        match = bench_df[bench_df["label"] == lbl]
        if match.empty:
            continue
        x = match["trainable_params"].values[0]
        y = row["best_adapted_acc"]
        color = COLORS[method]
        marker = "o" if method == "lora" else {"head": "s", "bitfit": "^", "full": "D"}[method]
        ax.scatter(x, y, color=color, marker=marker, s=80, zorder=3)
        if method == "lora":
            ax.annotate(f"r={int(row['rank'])}", (x, y),
                        textcoords="offset points", xytext=(5, 3), fontsize=8)

    # Legend
    patches = [mpatches.Patch(color=COLORS[m], label=METHOD_LABELS[m])
               for m in COLORS]
    ax.legend(handles=patches, loc="lower right", fontsize=9)

    ax.set_xscale("log")
    ax.set_xlabel("Trainable Parameters")
    ax.set_ylabel("Mean Adapted Accuracy")
    ax.set_title("Accuracy vs. Trainable Parameters (Pareto)")
    ax.grid(True, alpha=0.3)
    savefig(fig, out)


# ─────────────────────────────────────────────
# Fig 2: Pareto — accuracy vs latency
# ─────────────────────────────────────────────

def fig_pareto_latency(df, bench_df, out):
    fig, ax = plt.subplots(figsize=(7, 5))

    grp = df.groupby(["method", "rank", "placement"])["best_adapted_acc"].mean().reset_index()

    for _, row in grp.iterrows():
        method = row["method"]
        lbl = (f"lora_r{int(row['rank'])}_{row['placement']}"
               if method == "lora" else method)
        match = bench_df[bench_df["label"] == lbl]
        if match.empty:
            continue
        x = match["latency_mean_ms"].values[0]
        y = row["best_adapted_acc"]
        ax.scatter(x, y, color=COLORS[method], marker="o", s=80, zorder=3)
        if method == "lora":
            ax.annotate(f"r={int(row['rank'])}", (x, y),
                        textcoords="offset points", xytext=(4, 3), fontsize=8)

    patches = [mpatches.Patch(color=COLORS[m], label=METHOD_LABELS[m])
               for m in COLORS]
    ax.legend(handles=patches, fontsize=9)
    ax.set_xlabel("Inference Latency (ms, CPU batch=1)")
    ax.set_ylabel("Mean Adapted Accuracy")
    ax.set_title("Accuracy vs. Inference Latency (Pareto)")
    ax.grid(True, alpha=0.3)
    savefig(fig, out)


# ─────────────────────────────────────────────
# Fig 3: Accuracy vs severity per method
# ─────────────────────────────────────────────

def fig_severity_curves(df, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, metric, title in zip(
        axes,
        ["zero_shot_acc", "best_adapted_acc"],
        ["Zero-shot (no adaptation)", "After Adaptation"]
    ):
        grp = df.groupby(["method", "severity"])[metric].mean().reset_index()
        for method, color in COLORS.items():
            sub = grp[grp["method"] == method].sort_values("severity")
            if sub.empty:
                continue
            ax.plot(sub["severity"], sub[metric], marker="o",
                    color=color, label=METHOD_LABELS[method], linewidth=2)
        ax.set_xlabel("Corruption Severity")
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Accuracy vs. Corruption Severity", fontsize=13, y=1.02)
    savefig(fig, out)


# ─────────────────────────────────────────────
# Fig 4: LoRA rank sweep
# ─────────────────────────────────────────────

def fig_rank_sweep(df, bench_df, out):
    lora_df = df[(df["method"] == "lora") &
                 (df["corruption"] == "gaussian_noise") &
                 (df["severity"] == 3)].copy()
    if lora_df.empty:
        print("  [skip fig4] No rank sweep data yet")
        return

    ranks    = sorted(lora_df["rank"].dropna().unique())
    acc_vals = [lora_df[lora_df["rank"] == r]["best_adapted_acc"].mean() for r in ranks]

    # Trainable params from benchmark
    param_vals = []
    for r in ranks:
        lbl   = f"lora_r{int(r)}_late"
        match = bench_df[bench_df["label"] == lbl]
        param_vals.append(match["trainable_params"].values[0] if not match.empty else np.nan)

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax2 = ax1.twinx()

    ax1.plot(ranks, acc_vals,  "o-", color=COLORS["lora"],  linewidth=2, label="Accuracy")
    ax2.plot(ranks, [p/1e3 for p in param_vals], "s--",
             color="gray", linewidth=1.5, label="Trainable params (K)")

    ax1.set_xlabel("LoRA Rank (r)")
    ax1.set_ylabel("Adapted Accuracy", color=COLORS["lora"])
    ax2.set_ylabel("Trainable Parameters (K)", color="gray")
    ax1.set_xticks(ranks)
    ax1.set_title("LoRA Rank vs. Accuracy & Parameter Count")
    ax1.grid(True, alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    savefig(fig, out)


# ─────────────────────────────────────────────
# Fig 5: Placement heatmap
# ─────────────────────────────────────────────

def fig_placement_heatmap(df, out):
    lora_df = df[df["method"] == "lora"].copy()
    if lora_df.empty or "placement" not in lora_df.columns:
        print("  [skip fig5] No placement data yet")
        return

    pivot = lora_df.groupby(["placement", "corruption"])["best_adapted_acc"] \
                   .mean().unstack(fill_value=np.nan)
    if pivot.empty:
        print("  [skip fig5] Placement pivot is empty")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0.3, vmax=0.9)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    plt.colorbar(im, ax=ax, label="Adapted Accuracy")
    ax.set_title("LoRA Placement × Corruption (mean over severities)")

    # Annotate cells
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="black")
    savefig(fig, out)


# ─────────────────────────────────────────────
# Fig 6: Per-corruption bar chart
# ─────────────────────────────────────────────

def fig_corruption_breakdown(df, out):
    grp = df.groupby(["method", "corruption"])["best_adapted_acc"].mean().reset_index()
    corruptions = sorted(grp["corruption"].unique())
    methods     = list(COLORS.keys())
    x = np.arange(len(corruptions))
    width = 0.2

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, method in enumerate(methods):
        sub  = grp[grp["method"] == method].set_index("corruption")
        vals = [sub.loc[c, "best_adapted_acc"] if c in sub.index else np.nan
                for c in corruptions]
        ax.bar(x + i * width, vals, width, label=METHOD_LABELS[method],
               color=COLORS[method], alpha=0.85)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(corruptions, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean Adapted Accuracy")
    ax.set_title("Adapted Accuracy per Corruption Type")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")
    savefig(fig, out)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main(args):
    print("Loading data...")
    df       = load_sweep(args.sweep)
    bench_df = load_bench(args.bench)
    print(f"  Sweep rows: {len(df)}  |  Benchmark rows: {len(bench_df)}")

    out = args.out_dir
    print("\nGenerating figures...")
    fig_pareto_params(df, bench_df,  f"{out}/fig1_accuracy_vs_params.pdf")
    fig_pareto_latency(df, bench_df, f"{out}/fig2_accuracy_vs_latency.pdf")
    fig_severity_curves(df,          f"{out}/fig3_severity_curves.pdf")
    fig_rank_sweep(df, bench_df,     f"{out}/fig4_rank_sweep.pdf")
    fig_placement_heatmap(df,        f"{out}/fig5_placement_heatmap.pdf")
    fig_corruption_breakdown(df,     f"{out}/fig6_corruption_breakdown.pdf")
    print("\n✓ All figures saved.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sweep",   default="results/sweep.jsonl")
    p.add_argument("--bench",   default="results/benchmark.json")
    p.add_argument("--out_dir", default="figures")
    main(p.parse_args())