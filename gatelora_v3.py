"""GateLoRA v3 — capacity controls + noise-aware GateConcat (experiment script).

Shared code lives in the `gateconcat` package. Kept in THIS file, verbatim,
because their v3 semantics differ from v4 in ways that affect results:
  - soft_gate_aux_loss and the GateConcatAux trainer (train_concat_aux)
  - ScalarConcatModel / GateConcatAuxModel (v3-only variants)
  - _trainable_setup / train_task_only (scalar-projection handling)
  - evaluate_gate_corruption_auc (v3 scores ScalarConcat's scalar as a gate;
    the v4 version returns None for models without a gate MLP)

GateConcat / GateConcatFrozen themselves are the package (v4) definitions:
their forward(..., return_gates=True) contract is identical for v3 callers.

Usage:
  python -u gatelora_v3.py \
      --dataset funsd --corruption-rate 0.7 \
      --out results/run_sub07.csv
"""
from __future__ import annotations

import os

os.environ["PEFT_DISABLE_TORCHAO"] = "1"

import argparse
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer

from gateconcat.config import Config
from gateconcat.data import DocumentDatasetWithNoise, load_and_prepare_dataset
from gateconcat.json_utils import dump_json
from gateconcat.losses import ce_mean, quality_to_clean_target  # noqa: F401
from gateconcat.models import (
    GateConcatFrozenModel,
    GateConcatModel,
    StandardLoRAModel,
    _SharedLoRAShell,
)
from gateconcat.seed import clear_cuda, set_seed
from gateconcat.training import _make_optim, _stats, evaluate, make_seeded_loader


# ---------------------------------------------------------------------------
# v3-only loss / models / trainers (verbatim from the original v3 monolith)
# ---------------------------------------------------------------------------
def soft_gate_aux_loss(gates: torch.Tensor, token_quality: torch.Tensor) -> torch.Tensor:
    valid = (token_quality > 0).float()
    clean = (token_quality >= Config.clean_token_weight - 1e-6).float()
    target = clean + (1.0 - clean) * Config.soft_corrupt_target
    if valid.sum() < 1:
        return gates.new_zeros(())
    return F.binary_cross_entropy(
        gates.clamp(1e-6, 1 - 1e-6), target, weight=valid, reduction="sum"
    ) / valid.sum()


class ScalarConcatModel(_SharedLoRAShell):
    """Capacity control: concat(h_lora, σ(w·h_lora)); no gate MLP, no aux."""

    def __init__(self, num_labels: int):
        super().__init__(num_labels, classifier_in=None)
        self.classifier = nn.Linear(self.hidden_size + 1, num_labels)
        self.scalar_proj = nn.Linear(self.hidden_size, 1)

    def forward(self, input_ids, bbox, attention_mask=None, return_gates=False):
        h_lora = self._encode(input_ids, bbox, attention_mask, True)
        g = torch.sigmoid(self.scalar_proj(h_lora))
        logits = self.classifier(torch.cat([h_lora, g], dim=-1))
        return (logits, g.squeeze(-1)) if return_gates else logits


class GateConcatAuxModel(GateConcatModel):
    """Proposed v3: concat(h_lora, g) + soft aux teaches g to track corruption."""

    pass  # same forward; different trainer


# -----------------------------------------------------------------------------
# TRAIN / EVAL


def _trainable_setup(model, train_gate=True, train_scalar=True):
    for name, p in model.backbone.named_parameters():
        p.requires_grad = "lora" in name
    for p in model.classifier.parameters():
        p.requires_grad = True
    if hasattr(model, "gate") and train_gate:
        for p in model.gate.parameters():
            p.requires_grad = True
    if hasattr(model, "scalar_proj") and train_scalar:
        for p in model.scalar_proj.parameters():
            p.requires_grad = True

def train_task_only(model, train_loader, train_gate=True, train_scalar=True):
    _trainable_setup(model, train_gate=train_gate, train_scalar=train_scalar)
    opt, sched = _make_optim(model)
    for _ in range(Config.epochs):
        model.train()
        for batch in train_loader:
            ids = batch["input_ids"].to(Config.device)
            attn = batch["attention_mask"].to(Config.device)
            bbox = batch["bbox"].to(Config.device)
            labels = batch["labels"].to(Config.device)
            logits = model(ids, bbox, attn)
            loss = ce_mean(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()

def train_concat_aux(model, train_loader):
    _trainable_setup(model, train_gate=True)
    opt, sched = _make_optim(model)
    for epoch in range(Config.epochs):
        lam = 0.0 if epoch < Config.aux_warmup_epochs else Config.lambda_gate
        model.train()
        for batch in train_loader:
            ids = batch["input_ids"].to(Config.device)
            attn = batch["attention_mask"].to(Config.device)
            bbox = batch["bbox"].to(Config.device)
            labels = batch["labels"].to(Config.device)
            tq = batch["token_quality_mask"].to(Config.device)
            logits, gates = model(ids, bbox, attn, return_gates=True)
            task = ce_mean(logits, labels)
            loss = task + lam * soft_gate_aux_loss(gates, tq) if lam > 0 else task
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()

def evaluate_gate_corruption_auc(model, loader) -> Optional[float]:
    # Frozen/constant gates cannot rank corruption; skip to avoid spurious AUC ≈ 0.5.
    if getattr(model, "gate_auc_applicable", True) is False:
        return None

    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(Config.device)
            attn = batch["attention_mask"].to(Config.device)
            bbox = batch["bbox"].to(Config.device)
            tq = batch["token_quality_mask"].to(Config.device)
            out = model(ids, bbox, attn, return_gates=True)
            if not isinstance(out, tuple):
                return None
            _, gates = out
            if gates is None:
                return None
            clean_t, valid = quality_to_clean_target(tq)
            m = valid.bool()
            if m.sum() == 0:
                continue
            corrupted = 1.0 - clean_t[m]
            scores.extend((1.0 - gates[m]).cpu().numpy().tolist())
            labels.extend(corrupted.cpu().numpy().tolist())
    if len(set(labels)) < 2:
        return None
    if len(scores) > 1 and float(np.std(scores)) < 1e-8:
        return None  # degenerate constant scores
    return float(roc_auc_score(labels, scores))

ALL_METHODS = (
    "lora",
    "gate_concat",
    "gate_concat_frozen",
    "scalar_concat",
    "gate_concat_aux",
)

METHOD_LABELS = {
    "lora": "StandardLoRA",
    "gate_concat": "GateConcat",
    "gate_concat_frozen": "GateConcatFrozen(g=0.5)",
    "scalar_concat": "ScalarConcat",
    "gate_concat_aux": "GateConcatAux",
}


def _train_and_eval(name: str, num_labels: int, train_loader, clean_loader, noisy_loader):
    if name == "lora":
        m = StandardLoRAModel(num_labels).to(Config.device)
        train_task_only(m, train_loader)
    elif name == "gate_concat":
        m = GateConcatModel(num_labels).to(Config.device)
        train_task_only(m, train_loader)
    elif name == "gate_concat_frozen":
        m = GateConcatFrozenModel(num_labels).to(Config.device)
        train_task_only(m, train_loader, train_gate=False)
    elif name == "scalar_concat":
        m = ScalarConcatModel(num_labels).to(Config.device)
        train_task_only(m, train_loader, train_gate=False)
    elif name == "gate_concat_aux":
        m = GateConcatAuxModel(num_labels).to(Config.device)
        train_concat_aux(m, train_loader)
    else:
        raise ValueError(f"Unknown method: {name}")

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
        f"GateLoRA v3 | {Config.dataset_name} | {Config.corruption_type} @ "
        f"{Config.corruption_rate * 100:.0f}%"
    )
    print(f"epochs={Config.epochs} seeds={Config.seeds}")
    print(f"Methods: {' | '.join(METHOD_LABELS[m] for m in active)}")
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
    buckets: Dict[str, list] = {
        f"{m}_{s}": [] for m in active for s in ("cw", "cm", "nw", "nm", "auc")
    }

    for seed in list(Config.seeds):
        set_seed(seed)
        print(f"\n--- Seed {seed} ---")
        train_loader = make_seeded_loader(noisy_train, seed, Config.batch_size, True)
        clean_loader = make_seeded_loader(test_clean, seed, Config.batch_size, False)
        noisy_loader = make_seeded_loader(test_noisy, seed, Config.batch_size, False)

        row = {"seed": seed, "corruption_rate": Config.corruption_rate, "corruption_type": Config.corruption_type}
        for name in active:
            print(f"  Training {METHOD_LABELS[name]}...")
            cw, cm, nw, nm, auc = _train_and_eval(name, num_labels, train_loader, clean_loader, noisy_loader)
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
        "protocol": "gatelora_v3",
        "lambda_gate": Config.lambda_gate,
        "soft_corrupt_target": Config.soft_corrupt_target,
        "aux_warmup_epochs": Config.aux_warmup_epochs,
        "frozen_gate_value": Config.frozen_gate_value,
        "lr": Config.lr,
        "lora_r": Config.lora_r,
    }

    lora_cm_m = None
    lora_nm_m = None
    for name in active:
        cw_m, cw_s = _stats(buckets[f"{name}_cw"])
        cm_m, cm_s = _stats(buckets[f"{name}_cm"])
        nw_m, nw_s = _stats(buckets[f"{name}_nw"])
        nm_m, nm_s = _stats(buckets[f"{name}_nm"])
        auc_m, auc_s = _stats(buckets[f"{name}_auc"]) if buckets[f"{name}_auc"] else (float("nan"), float("nan"))
        print(f"{name:20s} clean M={cm_m:.4f}±{cm_s:.4f}  noisy M={nm_m:.4f}±{nm_s:.4f}  AUC={auc_m:.4f}")
        summary[f"{name}_clean_weighted_mean"] = cw_m
        summary[f"{name}_clean_macro_mean"] = cm_m
        summary[f"{name}_clean_macro_std"] = cm_s
        summary[f"{name}_noisy_weighted_mean"] = nw_m
        summary[f"{name}_noisy_macro_mean"] = nm_m
        summary[f"{name}_noisy_macro_std"] = nm_s
        summary[f"{name}_gate_auc_mean"] = auc_m
        if name == "lora":
            lora_cm_m, lora_nm_m = cm_m, nm_m
        elif lora_cm_m is not None:
            summary[f"{name}_minus_lora_clean_macro"] = cm_m - lora_cm_m
            summary[f"{name}_minus_lora_noisy_macro"] = nm_m - lora_nm_m

    print("=" * 60)
    return {"summary": summary, "per_seed": per_seed}


def parse_args():
    p = argparse.ArgumentParser(description="GateLoRA v3 — capacity controls + GateConcatAux")
    p.add_argument("--dataset", default="funsd", choices=["funsd", "cord", "sroie"])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--corruption-rate", type=float, default=None)
    p.add_argument("--corruption-rates", type=float, nargs="+", default=None)
    p.add_argument("--corruption-type", default="substitution",
                   choices=["substitution", "ocr_confusion", "box_swap"])
    p.add_argument("--methods", type=str, nargs="+", default=None)
    p.add_argument("--out", default="gatelora_v3_results.csv")
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
