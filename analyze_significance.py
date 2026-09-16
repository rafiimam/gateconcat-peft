"""Paired significance tests for GateConcat vs LoRA from per-seed CSV.

Usage:
  /home/eai/trtllm-env/bin/python analyze_significance.py \\
      --per-seed gatelora_v2_funsd07_per_seed.csv \\
      --method-a lora --method-b gate_concat \\
      --out gatelora_v2_funsd07_significance.json
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List

import pandas as pd
from scipy import stats


METRICS = [
    ("noisy_macro", "noisy_macro"),
    ("clean_macro", "clean_macro"),
    ("noisy_weighted", "noisy_weighted"),
    ("clean_weighted", "clean_weighted"),
]


def col(method: str, metric: str) -> str:
    return f"{method}_{metric}"


def run_tests(df: pd.DataFrame, method_a: str, method_b: str) -> Dict:
    results = {
        "method_a": method_a,
        "method_b": method_b,
        "n_seeds": len(df),
        "seeds": df["seed"].tolist() if "seed" in df.columns else list(range(len(df))),
        "metrics": {},
    }
    for label, metric in METRICS:
        a_col, b_col = col(method_a, metric), col(method_b, metric)
        if a_col not in df.columns or b_col not in df.columns:
            continue
        x = df[a_col].astype(float).values
        y = df[b_col].astype(float).values
        diff = y - x
        t_stat, t_p = stats.ttest_rel(y, x)
        try:
            w_stat, w_p = stats.wilcoxon(y, x)
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")
        results["metrics"][label] = {
            f"{method_a}_mean": float(x.mean()),
            f"{method_b}_mean": float(y.mean()),
            "mean_diff": float(diff.mean()),
            "per_seed_diff": diff.tolist(),
            "paired_t_stat": float(t_stat),
            "paired_t_p": float(t_p),
            "wilcoxon_stat": float(w_stat) if w_stat == w_stat else None,
            "wilcoxon_p": float(w_p) if w_p == w_p else None,
            "significant_005_t": bool(t_p < 0.05),
            "significant_005_wilcoxon": bool(w_p < 0.05) if w_p == w_p else False,
        }
    return results


def print_report(res: Dict) -> None:
    print(f"\n{res['method_b']} vs {res['method_a']} (paired, n={res['n_seeds']})")
    print("=" * 70)
    for label, m in res["metrics"].items():
        ma = f"{res['method_a']}_mean"
        mb = f"{res['method_b']}_mean"
        print(f"\n{label}:")
        print(f"  {res['method_a']} mean={m[ma]:.4f}  "
              f"{res['method_b']} mean={m[mb]:.4f}  "
              f"Δ={m['mean_diff']:+.4f}")
        print(f"  per-seed Δ: {[round(d, 4) for d in m['per_seed_diff']]}")
        print(f"  paired t-test: t={m['paired_t_stat']:.4f}, p={m['paired_t_p']:.4f} "
              f"{'*' if m['significant_005_t'] else ''}")
        print(f"  Wilcoxon: p={m['wilcoxon_p']:.4f} "
              f"{'*' if m['significant_005_wilcoxon'] else ''}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--per-seed", type=str, required=True)
    p.add_argument("--method-a", type=str, default="lora")
    p.add_argument("--method-b", type=str, default="gate_concat")
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    df = pd.read_csv(args.per_seed)
    res = run_tests(df, args.method_a, args.method_b)
    print_report(res)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nWrote {args.out}")
