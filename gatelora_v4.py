"""GateLoRA v4 — corruption transfer + fair GateConcat upgrade (experiment script).

All shared code (locked Config, corruption, dataset, models, training loop)
lives in the `gateconcat` package — this file contains only what is specific
to the v4 experiment: the method registry, _build/_train_and_eval wiring,
run_compare, and the CLI. The v4 protocol is unchanged from the original
monolith; see docs/REFACTOR.md for the mapping.

Usage:
  python -u gatelora_v4.py \
      --dataset funsd --corruption-type ocr_confusion --corruption-rate 0.7 \
      --out results/run_ocr07.csv
"""
from __future__ import annotations

import os

os.environ["PEFT_DISABLE_TORCHAO"] = "1"

import argparse
from typing import Dict, List, Optional

import pandas as pd
from transformers import AutoTokenizer

from gateconcat.config import Config
from gateconcat.data import DocumentDatasetWithNoise, load_and_prepare_dataset
from gateconcat.json_utils import dump_json
from gateconcat.models import (
    GateConcatFrozenModel,
    GateConcatModel,
    GateConcatPlusConstModel,
    GateConcatPlusModel,
    StandardLoRAModel,
)
from gateconcat.losses import quality_to_clean_target  # noqa: F401  (re-export)
from gateconcat.seed import clear_cuda, set_seed
from gateconcat.training import (
    _stats,
    evaluate,
    evaluate_gate_corruption_auc,
    make_seeded_loader,
    train_task_only,
)

# Names re-exported for analyze_gate_v4.py and backwards compatibility.
__all__ = [
    "Config", "DocumentDatasetWithNoise", "GateConcatModel", "GateConcatPlusModel",
    "clear_cuda", "load_and_prepare_dataset", "make_seeded_loader",
    "set_seed", "train_task_only", "run_compare",
]


ALL_METHODS = (
    "lora",
    "gate_concat",
    "gate_concat_frozen",
    "gate_concat_plus",
    "gate_concat_plus_const",
)
METHOD_LABELS = {
    "lora": "StandardLoRA",
    "gate_concat": "GateConcat(v3)",
    "gate_concat_frozen": "GateConcatFrozen(g=0.5)",
    "gate_concat_plus": "GateConcat+(||Δ||)",
    "gate_concat_plus_const": "GateConcat+Const(0.5)",
}


def _build(name: str, num_labels: int):
    if name == "lora":
        return StandardLoRAModel(num_labels)
    if name == "gate_concat":
        return GateConcatModel(num_labels)
    if name == "gate_concat_frozen":
        return GateConcatFrozenModel(num_labels)
    if name == "gate_concat_plus":
        return GateConcatPlusModel(num_labels)
    if name == "gate_concat_plus_const":
        return GateConcatPlusConstModel(num_labels)
    raise ValueError(name)


def _train_and_eval(name: str, num_labels: int, train_loader, clean_loader, noisy_loader):
    m = _build(name, num_labels).to(Config.device)
    train_gate = name != "gate_concat_frozen"
    train_task_only(m, train_loader, train_gate=train_gate)
    cw, cm = evaluate(m, clean_loader)
    nw, nm = evaluate(m, noisy_loader)
    auc = evaluate_gate_corruption_auc(m, noisy_loader)
    del m
    clear_cuda()
    return cw, cm, nw, nm, auc


def run_compare(config_override: Optional[dict] = None, methods: Optional[List[str]] = None) -> dict:
    if config_override:
        for k, v in config_override.items():
            setattr(Config, k, v)

    active = list(methods) if methods else list(ALL_METHODS)
    for m in active:
        if m not in ALL_METHODS:
            raise ValueError(f"Unknown method '{m}'. Choose from {ALL_METHODS}")

    print(f"Device: {Config.device}")
    print("=" * 60)
    print(
        f"GateLoRA v4 | {Config.dataset_name} | {Config.corruption_type} @ "
        f"{Config.corruption_rate * 100:.0f}%"
    )
    print(f"epochs={Config.epochs} seeds={Config.seeds}")
    print(f"Methods: {' | '.join(METHOD_LABELS[m] for m in active)}")
    print("Config locked: same LR/WD/epochs/LoRA as v3")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(Config.backbone_name)
    train_data, test_data, num_labels = load_and_prepare_dataset(
        Config.dataset_name, tokenizer, Config.max_seq_length
    )
    Config.num_labels = num_labels

    base_seed = 42
    noisy_train = DocumentDatasetWithNoise(
        train_data, tokenizer, p=Config.corruption_rate, max_len=Config.max_seq_length,
        noise_type=Config.corruption_type, seed=base_seed, box_swap_quality=Config.box_swap_quality,
    )
    test_clean = DocumentDatasetWithNoise(
        test_data, tokenizer, p=0.0, max_len=Config.max_seq_length,
        noise_type=Config.corruption_type, seed=base_seed, box_swap_quality=Config.box_swap_quality,
    )
    test_noisy = DocumentDatasetWithNoise(
        test_data, tokenizer, p=Config.corruption_rate, max_len=Config.max_seq_length,
        noise_type=Config.corruption_type, seed=base_seed, box_swap_quality=Config.box_swap_quality,
    )

    probe = make_seeded_loader(noisy_train, Config.seeds[0], Config.batch_size, True)
    Config._steps_per_run = max(1, len(probe) * Config.epochs)

    per_seed = []
    buckets: Dict[str, list] = {f"{m}_{s}": [] for m in active for s in ("cw", "cm", "nw", "nm", "auc")}

    for seed in list(Config.seeds):
        set_seed(seed)
        print(f"\n--- Seed {seed} ---")
        train_loader = make_seeded_loader(noisy_train, seed, Config.batch_size, True)
        clean_loader = make_seeded_loader(test_clean, seed, Config.batch_size, False)
        noisy_loader = make_seeded_loader(test_noisy, seed, Config.batch_size, False)

        row = {
            "seed": seed,
            "corruption_rate": Config.corruption_rate,
            "corruption_type": Config.corruption_type,
        }
        for name in active:
            print(f"  Training {METHOD_LABELS[name]}...")
            cw, cm, nw, nm, auc = _train_and_eval(
                name, num_labels, train_loader, clean_loader, noisy_loader
            )
            print(f"    clean W/M={cw:.4f}/{cm:.4f}  noisy W/M={nw:.4f}/{nm:.4f}  gateAUC={auc}")
            row[f"{name}_clean_weighted"] = cw
            row[f"{name}_clean_macro"] = cm
            row[f"{name}_noisy_weighted"] = nw
            row[f"{name}_noisy_macro"] = nm
            row[f"{name}_gate_auc"] = auc
            buckets[f"{name}_cw"].append(cw)
            buckets[f"{name}_cm"].append(cm)
            buckets[f"{name}_nw"].append(nw)
            buckets[f"{name}_nm"].append(nm)
            if auc is not None:
                buckets[f"{name}_auc"].append(auc)
        per_seed.append(row)

    print("\n" + "=" * 60)
    print("SUMMARY (mean ± std)")
    print("=" * 60)
    summary = {
        "dataset": Config.dataset_name,
        "corruption_type": Config.corruption_type,
        "corruption_rate": Config.corruption_rate,
        "epochs": Config.epochs,
        "seeds": list(Config.seeds),
        "methods": active,
        "protocol": "gatelora_v4",
        "lr": Config.lr,
        "lora_r": Config.lora_r,
    }

    lora_nm = lora_cm = None
    for name in active:
        cw_m, cw_s = _stats(buckets[f"{name}_cw"])
        cm_m, cm_s = _stats(buckets[f"{name}_cm"])
        nw_m, nw_s = _stats(buckets[f"{name}_nw"])
        nm_m, nm_s = _stats(buckets[f"{name}_nm"])
        auc_m, auc_s = (
            _stats(buckets[f"{name}_auc"]) if buckets[f"{name}_auc"] else (float("nan"), float("nan"))
        )
        print(f"{name:18s} clean M={cm_m:.4f}±{cm_s:.4f}  noisy M={nm_m:.4f}±{nm_s:.4f}  AUC={auc_m:.4f}")
        summary[f"{name}_clean_weighted_mean"] = cw_m
        summary[f"{name}_clean_macro_mean"] = cm_m
        summary[f"{name}_clean_macro_std"] = cm_s
        summary[f"{name}_noisy_weighted_mean"] = nw_m
        summary[f"{name}_noisy_macro_mean"] = nm_m
        summary[f"{name}_noisy_macro_std"] = nm_s
        summary[f"{name}_gate_auc_mean"] = auc_m
        if name == "lora":
            lora_nm, lora_cm = nm_m, cm_m
        elif lora_nm is not None:
            summary[f"{name}_minus_lora_noisy_macro"] = nm_m - lora_nm
            summary[f"{name}_minus_lora_clean_macro"] = cm_m - lora_cm

    print("=" * 60)
    return {"summary": summary, "per_seed": per_seed}


def parse_args():
    p = argparse.ArgumentParser(description="GateLoRA v4 — transfer + GateConcat+")
    p.add_argument("--dataset", default="funsd", choices=["funsd"],
                   help="Only funsd until CORD/SROIE label mapping is fixed")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--corruption-rate", type=float, default=None)
    p.add_argument("--corruption-rates", type=float, nargs="+", default=None)
    p.add_argument(
        "--corruption-type",
        default="substitution",
        choices=["substitution", "ocr_confusion", "box_swap", "insertion", "deletion"],
    )
    p.add_argument("--methods", type=str, nargs="+", default=None)
    p.add_argument("--out", default="gatelora_v4_results.csv")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rates = list(args.corruption_rates) if args.corruption_rates else [
        0.7 if args.corruption_rate is None else args.corruption_rate
    ]
    all_summaries, all_per_seed = [], []
    for rate in rates:
        override = {
            "dataset_name": args.dataset,
            "corruption_type": args.corruption_type,
            "corruption_rate": rate,
            "epochs": 15 if args.epochs is None else args.epochs,
            "seeds": [42, 123, 2024, 456, 789] if args.seeds is None else args.seeds,
        }
        out = run_compare(override, methods=args.methods)
        all_summaries.append(out["summary"])
        all_per_seed.extend(out["per_seed"])

    per_path = args.out.replace(".csv", "_per_seed.csv")
    json_path = args.out.replace(".csv", "_summary.json")
    pd.DataFrame(all_per_seed).to_csv(per_path, index=False)
    pd.DataFrame(all_summaries).to_csv(args.out, index=False)
    with open(json_path, "w") as f:
        dump_json(all_summaries if len(all_summaries) > 1 else all_summaries[0], f)
    print(f"\nWrote {args.out}, {per_path}, {json_path}")
