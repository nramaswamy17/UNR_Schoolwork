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
  figures/fig7_rank_x_severity.pdf          — H3a: rank × severity heatmap
  figures/fig8_rank_x_datasize.pdf          — H3b: rank × data size curves

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
    def label(row):
        if row["method"] == "lora":
            return f"LoRA-r{int(row['rank'])} ({row['placement']})"
        return {"head": "Last-layer", "bitfit": "BitFit", "full": "Full FT"}[row["method"]]
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
NON_LORA_MARKERS = {
    "head":   ("D", 100),
    "bitfit": ("^", 100),
    "full":   ("s", 100),
}

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
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
    fig, ax = plt.subplots(figsize=(7, 5))

    # LoRA — group by rank+placement (avoids NaN-drop from groupby with non-LoRA)
    lora_df  = df[df["method"] == "lora"]
    lora_grp = lora_df.groupby(["rank", "placement"])["best_adapted_acc"].mean().reset_index()
    for _, row in lora_grp.iterrows():
        lbl   = f"lora_r{int(row['rank'])}_{row['placement']}"
        match = bench_df[bench_df["label"] == lbl]
        if match.empty:
            continue
        x, y = match["trainable_params"].values[0], row["best_adapted_acc"]
        ax.scatter(x, y, color=COLORS["lora"], marker="o", s=80, zorder=3)
        ax.annotate(f"r={int(row['rank'])}", (x, y),
                    textcoords="offset points", xytext=(5, 3), fontsize=8)

    # Non-LoRA — group by method only (rank/placement are NaN, can't groupby)
    for method, (marker, size) in NON_LORA_MARKERS.items():
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        y     = sub["best_adapted_acc"].mean()
        match = bench_df[bench_df["label"] == method]
        if match.empty:
            continue
        x = match["trainable_params"].values[0]
        ax.scatter(x, y, color=COLORS[method], marker=marker, s=size, zorder=3)
        ax.annotate(METHOD_LABELS[method], (x, y),
                    textcoords="offset points", xytext=(5, 3), fontsize=8)

    patches = [mpatches.Patch(color=COLORS[m], label=METHOD_LABELS[m]) for m in COLORS]
    ax.legend(handles=patches, loc="lower right", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Trainable Parameters")
    ax.set_ylabel("Mean Adapted Accuracy")
    ax.set_title("Accuracy vs. Trainable Parameters (Pareto)\n"
                 "LoRA rank points show mean across all corruptions × severities\n"
                 "(not comparable to Table 2, which reports gaussian noise sev3 only)")
    ax.grid(True, alpha=0.3)
    savefig(fig, out)


# ─────────────────────────────────────────────
# Fig 2: Pareto — accuracy vs latency
# ─────────────────────────────────────────────

def fig_pareto_latency(df, bench_df, out):
    fig, ax = plt.subplots(figsize=(7, 5))

    # LoRA points
    lora_df  = df[df["method"] == "lora"]
    lora_grp = lora_df.groupby(["rank", "placement"])["best_adapted_acc"].mean().reset_index()
    lora_points = []
    for _, row in lora_grp.iterrows():
        lbl   = f"lora_r{int(row['rank'])}_{row['placement']}"
        match = bench_df[bench_df["label"] == lbl]
        if match.empty:
            continue
        x, y = match["latency_mean_ms"].values[0], row["best_adapted_acc"]
        ax.scatter(x, y, color=COLORS["lora"], marker="o", s=80, zorder=3)
        ax.annotate(f"r={int(row['rank'])}", (x, y),
                    textcoords="offset points", xytext=(4, 3), fontsize=8)
        lora_points.append((x, y))

    # Connect LoRA rank points with dashed line
    if lora_points:
        lora_points.sort(key=lambda p: p[0])
        xs, ys = zip(*lora_points)
        ax.plot(xs, ys, color=COLORS["lora"], linewidth=1.2,
                linestyle="--", zorder=2, alpha=0.6)

    # Non-LoRA points
    for method, (marker, size) in NON_LORA_MARKERS.items():
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        y     = sub["best_adapted_acc"].mean()
        match = bench_df[bench_df["label"] == method]
        if match.empty:
            continue
        x = match["latency_mean_ms"].values[0]
        ax.scatter(x, y, color=COLORS[method], marker=marker, s=size, zorder=3)
        ax.annotate(METHOD_LABELS[method], (x, y),
                    textcoords="offset points", xytext=(4, 3), fontsize=8)

    patches = [mpatches.Patch(color=COLORS[m], label=METHOD_LABELS[m]) for m in COLORS]
    ax.legend(handles=patches, fontsize=9)
    ax.set_xlabel("Inference Latency (ms, CPU batch=1)")
    ax.set_ylabel("Mean Adapted Accuracy")
    ax.set_title("Accuracy vs. Inference Latency (Pareto)\n"
                 "LoRA rank points show mean across all corruptions × severities")
    ax.grid(True, alpha=0.3)
    if not bench_df.empty:
        ax.set_xlim(0, bench_df["latency_mean_ms"].max() * 1.15)
    savefig(fig, out)


# ─────────────────────────────────────────────
# Fig 3: Accuracy vs severity per method
# ─────────────────────────────────────────────

def fig_severity_curves(df, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # Zero-shot is a property of the frozen baseline + corrupted data, NOT the
    # adaptation method. Use the method with the most complete corruption ×
    # severity coverage as the single reference curve.
    coverage = df.groupby("method").apply(
        lambda g: g.groupby(["corruption", "severity"]).ngroups
    )
    ref_method = coverage.idxmax()
    zero_grp = df[df["method"] == ref_method] \
                 .groupby("severity")["zero_shot_acc"].mean().reset_index()

    # Left panel — single shared zero-shot curve
    ax0 = axes[0]
    ax0.plot(zero_grp["severity"], zero_grp["zero_shot_acc"],
             marker="o", color="black", linewidth=2, label="Baseline (all methods)")
    ax0.set_xlabel("Corruption Severity")
    ax0.set_ylabel("Accuracy")
    ax0.set_title("Zero-shot (no adaptation)")
    ax0.set_xticks([1, 2, 3, 4, 5])
    ax0.legend(fontsize=9)
    ax0.grid(True, alpha=0.3)

    # Right panel — post-adaptation per method
    ax1 = axes[1]
    adap_grp = df.groupby(["method", "severity"])["best_adapted_acc"].mean().reset_index()
    for method, color in COLORS.items():
        sub = adap_grp[adap_grp["method"] == method].sort_values("severity")
        if sub.empty:
            continue
        ax1.plot(sub["severity"], sub["best_adapted_acc"], marker="o",
                 color=color, label=METHOD_LABELS[method], linewidth=2)
    ax1.set_xlabel("Corruption Severity")
    ax1.set_title("After Adaptation")
    ax1.set_xticks([1, 2, 3, 4, 5])
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

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

    param_vals = []
    for r in ranks:
        lbl   = f"lora_r{int(r)}_late"
        match = bench_df[bench_df["label"] == lbl]
        param_vals.append(match["trainable_params"].values[0] if not match.empty else np.nan)

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax2 = ax1.twinx()

    ax1.plot(ranks, acc_vals, "o-", color=COLORS["lora"], linewidth=2, label="Accuracy")
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
    display = np.where(np.isnan(pivot.values), 0, pivot.values)
    im = ax.imshow(display, aspect="auto", cmap="RdYlGn", vmin=0.3, vmax=0.9)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    plt.colorbar(im, ax=ax, label="Adapted Accuracy")
    ax.set_title("LoRA Placement × Corruption (mean over severities)\n"
                 "Gray hatch = not measured (placement ablation run on gaussian_noise only)")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if np.isnan(val):
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=True, facecolor="#cccccc", edgecolor="white",
                    hatch="////", linewidth=0
                ))
                ax.text(j, i, "N/M", ha="center", va="center",
                        fontsize=7, color="#555555")
            else:
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
    x     = np.arange(len(corruptions))
    width = 0.2

    METHOD_MARKERS_BAR = {"head": "/", "bitfit": "\\", "full": "x", "lora": ""}

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, method in enumerate(methods):
        sub  = grp[grp["method"] == method].set_index("corruption")
        vals = [sub.loc[c, "best_adapted_acc"] if c in sub.index else np.nan
                for c in corruptions]
        ax.bar(x + i * width, vals, width, label=METHOD_LABELS[method],
               color=COLORS[method], alpha=0.85,
               hatch=METHOD_MARKERS_BAR[method])

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(corruptions, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean Adapted Accuracy")
    ax.set_title("Adapted Accuracy per Corruption Type")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")
    savefig(fig, out)


# ─────────────────────────────────────────────
# Fig 7: H3a — rank × severity heatmap
# ─────────────────────────────────────────────

def fig_rank_x_severity(df, out):
    lora_df = df[
        (df["method"] == "lora") &
        (df["corruption"] == "gaussian_noise") &
        (df["placement"] == "late")
    ].copy()

    if lora_df.empty:
        print("  [skip fig7] No rank×severity data yet")
        return

    pivot = lora_df.groupby(["rank", "severity"])["best_adapted_acc"] \
                   .mean().unstack(fill_value=np.nan)
    if pivot.empty:
        print("  [skip fig7] Pivot is empty")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0.3, vmax=0.9)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"Sev {s}" for s in pivot.columns], fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"r={int(r)}" for r in pivot.index], fontsize=10)
    plt.colorbar(im, ax=ax, label="Adapted Accuracy")
    ax.set_title("H3a: LoRA Rank × Corruption Severity\n(gaussian noise, late placement)")
    ax.set_xlabel("Severity")
    ax.set_ylabel("LoRA Rank")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color="black")

    for j in range(pivot.shape[1]):
        col = pivot.values[:, j]
        if not np.all(np.isnan(col)):
            best_i = np.nanargmax(col)
            ax.add_patch(plt.Rectangle(
                (j - 0.5, best_i - 0.5), 1, 1,
                fill=False, edgecolor="black", linewidth=2
            ))
    savefig(fig, out)


# ─────────────────────────────────────────────
# Fig 8: H3b — rank × adapt data size
# ─────────────────────────────────────────────

def fig_rank_x_datasize(df, out):
    lora_df = df[
        (df["method"] == "lora") &
        (df["corruption"] == "gaussian_noise") &
        (df["severity"] == 3) &
        (df["placement"] == "late")
    ].copy()

    if "adapt_size" not in lora_df.columns or lora_df["adapt_size"].nunique() < 2:
        print("  [skip fig8] No rank×datasize data yet")
        return

    ranks = sorted(lora_df["rank"].dropna().unique())
    sizes = sorted(lora_df["adapt_size"].dropna().unique())

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap       = plt.cm.Blues
    color_vals = np.linspace(0.4, 1.0, len(ranks))

    for rank, cval in zip(ranks, color_vals):
        sub = lora_df[lora_df["rank"] == rank].groupby("adapt_size")["best_adapted_acc"] \
                     .mean().reset_index().sort_values("adapt_size")
        if sub.empty:
            continue
        ax.plot(sub["adapt_size"], sub["best_adapted_acc"],
                marker="o", label=f"r={int(rank)}",
                color=cmap(cval), linewidth=2)

    ax.set_xscale("log")
    ax.set_xlabel("Adaptation Data Size (samples)")
    ax.set_ylabel("Adapted Accuracy")
    ax.set_title("H3b: LoRA Rank × Adaptation Data Size\n(gaussian noise, severity 3, late)")
    ax.legend(title="LoRA Rank", fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
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
    fig_rank_x_severity(df,          f"{out}/fig7_rank_x_severity.pdf")
    fig_rank_x_datasize(df,          f"{out}/fig8_rank_x_datasize.pdf")
    print("\n✓ All figures saved.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sweep",   default="results/sweep.jsonl")
    p.add_argument("--bench",   default="results/benchmark.json")
    p.add_argument("--out_dir", default="figures")
    main(p.parse_args())