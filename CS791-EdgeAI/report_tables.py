"""
report_tables.py
----------------
Generates LaTeX tables for the final report from sweep + benchmark results.

Produces:
  tables/table1_method_comparison.tex   — main accuracy + resource table
  tables/table2_rank_ablation.tex       — LoRA rank sweep
  tables/table3_placement_ablation.tex  — layer placement results
  tables/table4_per_corruption.tex      — per-corruption accuracy breakdown

Usage:
    python report_tables.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────

def load_sweep(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def load_bench(path):
    with open(path) as f:
        return pd.DataFrame(json.load(f))


def save_tex(tex: str, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(tex)
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────

def bold(val, condition):
    s = f"{val:.4f}"
    return f"\\textbf{{{s}}}" if condition else s


def fmt_params(n):
    if n is None or np.isnan(n):
        return "---"
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    return f"{n/1e3:.1f}K"


def fmt_kb(kb):
    if kb is None or np.isnan(float(kb)):
        return "---"
    if kb >= 1000:
        return f"{kb/1000:.2f} MB"
    return f"{kb:.1f} KB"


# ─────────────────────────────────────────────
# Table 1: Method comparison
# ─────────────────────────────────────────────

def table_method_comparison(df, bench_df, out):
    """
    Rows: method (lora-r4-late, head, bitfit, full)
    Cols: zero-shot acc | adapted acc | Δ | trainable params | latency | adapt RAM | storage
    Averaged over all corruptions × severities.
    """
    METHOD_ORDER  = ["lora", "head", "bitfit", "full"]
    METHOD_LABELS = {
        "lora":   r"LoRA (r=4, late)",
        "head":   r"Last-layer",
        "bitfit": r"BitFit",
        "full":   r"Full Fine-tune",
    }

    # Sweep aggregation — use canonical rank/placement for lora
    lora_mask = (df["method"] == "lora") & (df["rank"] == 4) & (df["placement"] == "late")
    rows_list = []
    for method in METHOD_ORDER:
        if method == "lora":
            sub = df[lora_mask]
        else:
            sub = df[df["method"] == method]
        if sub.empty:
            continue

        zero  = sub["zero_shot_acc"].mean()
        adap  = sub["best_adapted_acc"].mean()
        delta = adap - zero

        # Benchmark
        lbl   = "lora_r4_late" if method == "lora" else method
        match = bench_df[bench_df["label"] == lbl]
        train_p  = match["trainable_params"].values[0]  if not match.empty else np.nan
        lat      = match["latency_mean_ms"].values[0]   if not match.empty else np.nan
        adapt_r  = match["adapt_ram_mb"].values[0]      if not match.empty else np.nan
        storage  = match["adapter_storage_kb"].values[0] if not match.empty else np.nan

        rows_list.append({
            "method": method,
            "label":  METHOD_LABELS[method],
            "zero":   zero,
            "adap":   adap,
            "delta":  delta,
            "params": train_p,
            "lat":    lat,
            "adapt_r": adapt_r,
            "storage": storage,
        })

    rows_list = pd.DataFrame(rows_list)
    best_adap = rows_list["adap"].max()

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Method comparison averaged over all corruptions and severities. "
                 r"Latency measured on CPU with batch size 1 and single thread.}")
    lines.append(r"\label{tab:method_comparison}")
    lines.append(r"\resizebox{\columnwidth}{!}{%")
    lines.append(r"\begin{tabular}{lcccccccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Zero-shot & Adapted & $\Delta$ & "
                 r"Train. Params & Latency (ms) & Adapt RAM & Storage \\")
    lines.append(r"\midrule")

    for _, r in rows_list.iterrows():
        is_best = abs(r["adap"] - best_adap) < 1e-6
        line = (
            f"{r['label']} & "
            f"{r['zero']:.4f} & "
            f"{bold(r['adap'], is_best)} & "
            f"{r['delta']:+.4f} & "
            f"{fmt_params(r['params'])} & "
            f"{r['lat']:.1f} & "
            f"{r['adapt_r']:.1f} MB & "
            f"{fmt_kb(r['storage'])} \\\\"
        )
        lines.append(line)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table}")
    save_tex("\n".join(lines), out)


# ─────────────────────────────────────────────
# Table 2: Rank ablation
# ─────────────────────────────────────────────

def table_rank_ablation(df, bench_df, out):
    lora_df = df[
        (df["method"] == "lora") &
        (df["corruption"] == "gaussian_noise") &
        (df["severity"] == 3)
    ].copy()

    if lora_df.empty:
        print("  [skip table2] No rank sweep data")
        return

    ranks = sorted(lora_df["rank"].dropna().unique())
    best_adap = max(lora_df.groupby("rank")["best_adapted_acc"].mean())

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{LoRA rank ablation on gaussian noise, severity 3.}")
    lines.append(r"\label{tab:rank_ablation}")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(r"Rank $r$ & Adapted Acc & Train. Params & "
                 r"Latency (ms) & Adapt RAM & Storage \\")
    lines.append(r"\midrule")

    for r in ranks:
        sub  = lora_df[lora_df["rank"] == r]
        adap = sub["best_adapted_acc"].mean()
        lbl  = f"lora_r{int(r)}_late"
        match = bench_df[bench_df["label"] == lbl]
        params  = match["trainable_params"].values[0]  if not match.empty else np.nan
        lat     = match["latency_mean_ms"].values[0]   if not match.empty else np.nan
        adapt_r = match["adapt_ram_mb"].values[0]      if not match.empty else np.nan
        storage = match["adapter_storage_kb"].values[0] if not match.empty else np.nan
        is_best = abs(adap - best_adap) < 1e-6
        line = (
            f"{int(r)} & "
            f"{bold(adap, is_best)} & "
            f"{fmt_params(params)} & "
            f"{lat:.1f} & "
            f"{adapt_r:.1f} MB & "
            f"{fmt_kb(storage)} \\\\"
        )
        lines.append(line)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    save_tex("\n".join(lines), out)


# ─────────────────────────────────────────────
# Table 3: Placement ablation
# ─────────────────────────────────────────────

def table_placement_ablation(df, bench_df, out):
    lora_df = df[
        (df["method"] == "lora") &
        (df["rank"] == 4) &
        (df["corruption"] == "gaussian_noise") &
        (df["severity"] == 3)
    ].copy()

    if lora_df.empty or "placement" not in lora_df.columns:
        print("  [skip table3] No placement data")
        return

    placements = lora_df["placement"].dropna().unique()
    best_adap  = max(lora_df.groupby("placement")["best_adapted_acc"].mean())

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{LoRA layer placement ablation (rank=4, gaussian noise sev 3).}")
    lines.append(r"\label{tab:placement_ablation}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Placement & Adapted Acc & Train. Params & Latency (ms) & Storage \\")
    lines.append(r"\midrule")

    for pl in ["early", "late", "all", "head"]:
        if pl not in placements:
            continue
        sub  = lora_df[lora_df["placement"] == pl]
        adap = sub["best_adapted_acc"].mean()
        lbl  = f"lora_r4_{pl}"
        match  = bench_df[bench_df["label"] == lbl]
        params  = match["trainable_params"].values[0]  if not match.empty else np.nan
        lat     = match["latency_mean_ms"].values[0]   if not match.empty else np.nan
        storage = match["adapter_storage_kb"].values[0] if not match.empty else np.nan
        is_best = abs(adap - best_adap) < 1e-6
        line = (
            f"{pl.capitalize()} & "
            f"{bold(adap, is_best)} & "
            f"{fmt_params(params)} & "
            f"{lat:.1f} & "
            f"{fmt_kb(storage)} \\\\"
        )
        lines.append(line)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    save_tex("\n".join(lines), out)


# ─────────────────────────────────────────────
# Table 4: Per-corruption breakdown
# ─────────────────────────────────────────────

def table_per_corruption(df, out):
    METHOD_ORDER  = ["lora", "head", "bitfit", "full"]
    METHOD_LABELS = {
        "lora": r"LoRA", "head": r"Last-layer",
        "bitfit": r"BitFit", "full": r"Full FT",
    }

    lora_mask = (df["method"] == "lora") & (df["rank"] == 4) & (df["placement"] == "late")
    corruptions = sorted(df["corruption"].unique())

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Mean adapted accuracy per corruption type (averaged over severities 1--5).}")
    lines.append(r"\label{tab:per_corruption}")
    col_spec = "l" + "c" * len(corruptions)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    header = "Method & " + " & ".join(
        c.replace("_", r"\_") for c in corruptions) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    # Per-corruption best for bolding
    per_corr_best = {}
    for corr in corruptions:
        vals = []
        for method in METHOD_ORDER:
            sub = df[lora_mask if method == "lora" else df["method"] == method]
            sub = sub[sub["corruption"] == corr]
            if not sub.empty:
                vals.append(sub["best_adapted_acc"].mean())
        per_corr_best[corr] = max(vals) if vals else 0

    for method in METHOD_ORDER:
        if method == "lora":
            sub = df[lora_mask]
        else:
            sub = df[df["method"] == method]
        if sub.empty:
            continue
        row_parts = [METHOD_LABELS[method]]
        for corr in corruptions:
            csub = sub[sub["corruption"] == corr]
            if csub.empty:
                row_parts.append("---")
            else:
                val     = csub["best_adapted_acc"].mean()
                is_best = abs(val - per_corr_best[corr]) < 1e-6
                row_parts.append(bold(val, is_best))
        lines.append(" & ".join(row_parts) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    save_tex("\n".join(lines), out)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main(args):
    print("Loading data...")
    df       = load_sweep(args.sweep)
    bench_df = load_bench(args.bench)
    print(f"  Sweep rows: {len(df)}  |  Benchmark rows: {len(bench_df)}")

    print("\nGenerating tables...")
    table_method_comparison(df, bench_df, f"{args.out_dir}/table1_method_comparison.tex")
    table_rank_ablation(df, bench_df,     f"{args.out_dir}/table2_rank_ablation.tex")
    table_placement_ablation(df, bench_df,f"{args.out_dir}/table3_placement_ablation.tex")
    table_per_corruption(df,              f"{args.out_dir}/table4_per_corruption.tex")
    print("\n✓ All tables saved.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sweep",   default="results/sweep.jsonl")
    p.add_argument("--bench",   default="results/benchmark.json")
    p.add_argument("--out_dir", default="tables")
    main(p.parse_args())