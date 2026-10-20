"""Paired significance tests for GateConcat vs LoRA from per-seed CSV.

Thin CLI wrapper around gateconcat.significance (logic unchanged).

Usage:
  python analyze_significance.py \
      --per-seed results/gatelora_v2_funsd07_per_seed.csv \
      --method-a lora --method-b gate_concat
"""
from __future__ import annotations

import argparse

from gateconcat.json_utils import dump_json
from gateconcat.significance import print_report, run_tests


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--per-seed", type=str, required=True)
    p.add_argument("--method-a", type=str, default="lora")
    p.add_argument("--method-b", type=str, default="gate_concat")
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    import pandas as pd

    args = parse_args()
    df = pd.read_csv(args.per_seed)
    res = run_tests(df, args.method_a, args.method_b)
    print_report(res)
    if args.out:
        with open(args.out, "w") as f:
            dump_json(res, f)
        print(f"\nWrote {args.out}")
