"""Paired significance tests (from analyze_significance.py, unchanged logic)."""
from __future__ import annotations

from typing import Dict

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
