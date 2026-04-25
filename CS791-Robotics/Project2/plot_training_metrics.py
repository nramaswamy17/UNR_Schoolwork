#!/usr/bin/env python3
"""
Plot training metrics from training_metrics.csv for a CS791 report.

Creates:
  01_total_episode_reward.png
  02_episode_length.png
  03_termination_reasons.png
  04_reward_breakdown.png
  plot_summary.csv

Usage:
  python plot_training_metrics.py /path/to/training_metrics.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def find_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first column name that exists in the dataframe."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def sanitize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce every column to numeric where possible."""
    out = df.copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="ignore")
    return out


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python plot_training_metrics.py /path/to/training_metrics.csv")
        return 1

    csv_path = Path(sys.argv[1]).expanduser().resolve()
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        return 1

    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"ERROR: CSV is empty: {csv_path}")
        return 1

    df = sanitize_numeric(df)

    plots_dir = csv_path.parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    x_col = find_first_existing(df, ["iteration", "Iteration", "iter"])
    if x_col is None:
        df["iteration"] = range(len(df))
        x_col = "iteration"

    reward_col = find_first_existing(
        df,
        [
            "Mean reward",
            "mean_reward",
            "episode_reward",
            "Episode_Reward/total",
            "total_episode_reward",
        ],
    )

    episode_length_col = find_first_existing(
        df,
        [
            "Mean episode length",
            "mean_episode_length",
            "episode_length",
            "Episode_Length",
        ],
    )

    termination_cols = [
        c for c in df.columns
        if "Episode_Termination/" in c
        or c.startswith("termination/")
        or c in {"time_outs", "fallen", "bad_orientation"}
    ]

    reward_breakdown_cols = [
        c for c in df.columns
        if "Episode_Reward/" in c and c != "Episode_Reward/total"
    ]

    summary_rows: list[dict[str, object]] = []

    # 1) Total episode reward
    if reward_col is not None:
        plt.figure(figsize=(8, 5))
        plt.plot(df[x_col], df[reward_col])
        plt.xlabel("Iteration")
        plt.ylabel("Reward")
        plt.title("Total Episode Reward During Training")
        plt.tight_layout()
        out = plots_dir / "01_total_episode_reward.png"
        plt.savefig(out, dpi=200)
        plt.close()

        summary_rows.append({
            "plot": "total_episode_reward",
            "source_column": reward_col,
            "output_file": out.name,
        })
    else:
        print("WARNING: No total reward column found.")

    # 2) Episode length
    if episode_length_col is not None:
        plt.figure(figsize=(8, 5))
        plt.plot(df[x_col], df[episode_length_col])
        plt.xlabel("Iteration")
        plt.ylabel("Episode Length")
        plt.title("Episode Length During Training")
        plt.tight_layout()
        out = plots_dir / "02_episode_length.png"
        plt.savefig(out, dpi=200)
        plt.close()

        summary_rows.append({
            "plot": "episode_length",
            "source_column": episode_length_col,
            "output_file": out.name,
        })
    else:
        print("WARNING: No episode length column found.")

    # 3) Termination reasons / fall rate
    if termination_cols:
        plt.figure(figsize=(9, 5))
        for col in termination_cols:
            plt.plot(df[x_col], df[col], label=col.replace("Episode_Termination/", ""))
        plt.xlabel("Iteration")
        plt.ylabel("Termination Metric")
        plt.title("Termination Reasons During Training")
        plt.legend()
        plt.tight_layout()
        out = plots_dir / "03_termination_reasons.png"
        plt.savefig(out, dpi=200)
        plt.close()

        summary_rows.append({
            "plot": "termination_reasons",
            "source_column": "; ".join(termination_cols),
            "output_file": out.name,
        })
    else:
        print("WARNING: No termination columns found.")

    # 4) Reward breakdown
    if reward_breakdown_cols:
        plt.figure(figsize=(10, 6))
        for col in reward_breakdown_cols:
            plt.plot(df[x_col], df[col], label=col.replace("Episode_Reward/", ""))
        plt.xlabel("Iteration")
        plt.ylabel("Reward Term Value")
        plt.title("Reward Breakdown During Training")
        plt.legend()
        plt.tight_layout()
        out = plots_dir / "04_reward_breakdown.png"
        plt.savefig(out, dpi=200)
        plt.close()

        summary_rows.append({
            "plot": "reward_breakdown",
            "source_column": "; ".join(reward_breakdown_cols),
            "output_file": out.name,
        })
    else:
        print("WARNING: No reward breakdown columns found.")

    # Save summary
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(plots_dir / "plot_summary.csv", index=False)

    print(f"Done. Plots saved to: {plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
