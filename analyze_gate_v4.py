"""Gate analysis for GateConcat / GateConcat+ (v4).

Trains one model under the locked Config (same as gatelora_v4), then dumps
per-token gate diagnostics on the noisy test set — without changing training HPs.

Correlations checked (not vs corruption labels alone):
  - gate vs synthetic corruption flag
  - gate vs ||delta||
  - gate vs bbox area
  - gate vs NER label id
  - gate vs prediction correctness
  - mean gate on clean vs corrupted tokens

Usage:
  /home/eai/trtllm-env/bin/python -u analyze_gate_v4.py \\
    --method gate_concat --corruption-type substitution --corruption-rate 0.7 \\
    --seed 42 --out-dir gate_analysis_v4

  /home/eai/trtllm-env/bin/python -u analyze_gate_v4.py \\
    --method gate_concat_plus --corruption-type ocr_confusion --corruption-rate 0.7 \\
    --seed 42 --out-dir gate_analysis_v4_ocr
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Reuse v4 training stack (locked Config)
from gatelora_v4 import (
    Config,
    DocumentDatasetWithNoise,
    GateConcatModel,
    GateConcatPlusModel,
    clear_cuda,
    load_and_prepare_dataset,
    make_seeded_loader,
    quality_to_clean_target,
    set_seed,
    train_task_only,
)


def build_model(method: str, num_labels: int):
    if method == "gate_concat":
        return GateConcatModel(num_labels)
    if method == "gate_concat_plus":
        return GateConcatPlusModel(num_labels)
    raise ValueError(method)


def collect_token_rows(model, loader) -> pd.DataFrame:
    model.eval()
    rows: List[Dict] = []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(Config.device)
            attn = batch["attention_mask"].to(Config.device)
            bbox = batch["bbox"].to(Config.device)
            labels = batch["labels"].to(Config.device)
            tq = batch["token_quality_mask"].to(Config.device)

            logits, gates, delta_norm = model(ids, bbox, attn, return_debug=True)
            preds = logits.argmax(-1)
            clean_t, valid = quality_to_clean_target(tq)
            corrupted = 1.0 - clean_t

            # bbox area in normalized coords
            bb = bbox.float()
            area = ((bb[..., 2] - bb[..., 0]).clamp(min=0) * (bb[..., 3] - bb[..., 1]).clamp(min=0)) / (1000.0 ** 2)

            bsz, seq = labels.shape
            for bi in range(bsz):
                for ti in range(seq):
                    if valid[bi, ti].item() < 0.5:
                        continue
                    y = int(labels[bi, ti].item())
                    p = int(preds[bi, ti].item())
                    rows.append(
                        {
                            "gate": float(gates[bi, ti].item()),
                            "delta_norm": float(delta_norm[bi, ti].item()),
                            "corrupted": float(corrupted[bi, ti].item()),
                            "clean": float(clean_t[bi, ti].item()),
                            "label": y,
                            "pred": p,
                            "correct": int(y == p),
                            "bbox_area": float(area[bi, ti].item()),
                            "token_pos": ti,
                        }
                    )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> Dict:
    out: Dict = {
        "n_tokens": int(len(df)),
        "gate_mean": float(df["gate"].mean()),
        "gate_std": float(df["gate"].std()),
        "gate_mean_clean": float(df.loc[df["clean"] > 0.5, "gate"].mean()) if (df["clean"] > 0.5).any() else None,
        "gate_mean_corrupted": float(df.loc[df["corrupted"] > 0.5, "gate"].mean()) if (df["corrupted"] > 0.5).any() else None,
        "corr_gate_corrupted": float(df["gate"].corr(df["corrupted"])) if df["corrupted"].nunique() > 1 else None,
        "corr_gate_delta_norm": float(df["gate"].corr(df["delta_norm"])),
        "corr_gate_bbox_area": float(df["gate"].corr(df["bbox_area"])),
        "corr_gate_correct": float(df["gate"].corr(df["correct"])),
        "corr_gate_token_pos": float(df["gate"].corr(df["token_pos"])),
        "gate_by_label": df.groupby("label")["gate"].mean().to_dict(),
        "acc_high_gate": float(df.loc[df["gate"] >= df["gate"].median(), "correct"].mean()),
        "acc_low_gate": float(df.loc[df["gate"] < df["gate"].median(), "correct"].mean()),
    }
    # Convert label keys to str for JSON
    out["gate_by_label"] = {str(k): float(v) for k, v in out["gate_by_label"].items()}
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="gate_concat", choices=["gate_concat", "gate_concat_plus"])
    p.add_argument("--dataset", default="funsd")
    p.add_argument("--corruption-type", default="substitution",
                   choices=["substitution", "ocr_confusion", "box_swap"])
    p.add_argument("--corruption-rate", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=None, help="default: Config.epochs (15)")
    p.add_argument("--out-dir", default="gate_analysis_v4")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    Config.dataset_name = args.dataset
    Config.corruption_type = args.corruption_type
    Config.corruption_rate = args.corruption_rate
    if args.epochs is not None:
        Config.epochs = args.epochs

    set_seed(args.seed)
    print(f"Device: {Config.device}")
    print(f"Analyze {args.method} | {args.corruption_type} @ {args.corruption_rate} | seed={args.seed}")

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
    test_noisy = DocumentDatasetWithNoise(
        test_data, tokenizer, p=Config.corruption_rate, max_len=Config.max_seq_length,
        noise_type=Config.corruption_type, seed=base_seed, box_swap_quality=Config.box_swap_quality,
    )

    train_loader = make_seeded_loader(noisy_train, args.seed, Config.batch_size, True)
    Config._steps_per_run = max(1, len(train_loader) * Config.epochs)
    test_loader = make_seeded_loader(test_noisy, args.seed, Config.batch_size, False)

    print("Training...")
    model = build_model(args.method, num_labels).to(Config.device)
    train_task_only(model, train_loader)

    print("Collecting token-level gate stats on noisy test...")
    df = collect_token_rows(model, test_loader)
    summary = summarize(df)

    tag = f"{args.method}_{args.corruption_type}_p{args.corruption_rate}_seed{args.seed}"
    csv_path = os.path.join(args.out_dir, f"{tag}_tokens.csv")
    json_path = os.path.join(args.out_dir, f"{tag}_summary.json")
    # subsample tokens CSV if huge
    if len(df) > 200_000:
        df.sample(200_000, random_state=args.seed).to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")

    del model
    clear_cuda()
