"""GateLoRA — architecture-matched comparison under the locked protocol.

Methods (identical backbone shell / LoRA / optimizer / data / seeds / epochs):
  1. StandardLoRA  — LayoutLMv3Model + LoRA + Linear head (no gate)
  2. RouterLoRA    — same shell + gate(h_lora, bbox); task loss only
  3. GateLoRA      — same shell + gate(h_base, delta, ||delta||, bbox) + aux BCE
  4. GateLoRA-NoAux (optional) — GateLoRA features, lambda_gate=0 (ablation)

Fairness rules (do not violate for venue submissions):
  - Same Config for every method (LR, WD, epochs, LoRA r/alpha, batch, seeds).
  - Same backbone class (LayoutLMv3Model) and same Linear classifier init pattern.
  - Same trainable set: LoRA adapters + classifier (+ gate if present); base frozen.
  - Same noisy train set and same clean/noisy test sets.
  - Task loss is unweighted CE for ALL methods (no quality reweighting).
  - GateLoRA aux uses train-time synthetic corruption labels only; never for test preds.
  - No per-method LR/epoch/warmup privileges. No early stopping on test.
  - lambda_gate is fixed (not swept per seed/dataset after seeing results).

Usage:
  python gatelora.py --dataset funsd --out gatelora_funsd_results.csv
  python gatelora.py --dataset funsd --corruption-rates 0.0 0.3 0.5 0.7
  python gatelora.py --dataset funsd --include-no-aux
"""
from __future__ import annotations

import os

os.environ["PEFT_DISABLE_TORCHAO"] = "1"

import argparse
import json
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import f1_score, roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    LayoutLMv3Model,
    get_linear_schedule_with_warmup,
)
import warnings

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# CONFIGURATION — locked; identical to sallora.py (+ one GateLoRA constant)
# -----------------------------------------------------------------------------
class Config:
    backbone_name = "microsoft/layoutlmv3-base"
    max_seq_length = 512
    num_labels = 7  # overridden per dataset

    batch_size = 4
    epochs = 15
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.1
    lr = 2e-4
    weight_decay = 0.01
    seeds = [42, 123, 2024, 456, 789]

    use_scalars = True
    soft_mask_factor = 0.15
    mask_keep_ratio = 0.2
    clean_ref_size = 32
    box_swap_quality = 0.5

    gate_hidden = 256
    gate_dropout = 0.1

    # Fixed method hyperparameter for GateLoRA (not tuned per seed/dataset).
    # L = L_task + lambda_gate * L_gate
    lambda_gate = 0.5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_name = "funsd"
    corruption_type = "substitution"
    corruption_rate = 0.7
    loss_type = "cross_entropy"
    label_smoothing_eps = 0.1
    focal_gamma = 2.0

    corrupted_token_weight = 0.1
    clean_token_weight = 1.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clear_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# -----------------------------------------------------------------------------
# DATA — identical protocol to sallora.py
# -----------------------------------------------------------------------------
def load_and_prepare_dataset(dataset_name: str, tokenizer, max_len: int):
    if dataset_name == "funsd":
        ds = load_dataset("nielsr/funsd-layoutlmv3")
        return ds["train"], ds["test"], 7

    if dataset_name == "cord":
        ds = load_dataset("naver-clova-ix/cord-v2")

        def process_cord(examples):
            tokens_list, bboxes_list, ner_tags_list = [], [], []
            for gt_str in examples["ground_truth"]:
                data = json.loads(gt_str)
                words, boxes, tags = [], [], []
                for line in data.get("valid_line", []):
                    for word_info in line.get("words", []):
                        words.append(word_info["text"])
                        boxes.append(word_info["bbox"])
                        tags.append(word_info["category"])
                tokens_list.append(words)
                bboxes_list.append(boxes)
                ner_tags_list.append(tags)
            return {"tokens": tokens_list, "bboxes": bboxes_list, "ner_tags": ner_tags_list}

        train = ds["train"].map(process_cord, batched=True, remove_columns=ds["train"].column_names)
        test = ds["test"].map(process_cord, batched=True, remove_columns=ds["test"].column_names)
        return train, test, 30

    if dataset_name == "sroie":
        ds = load_dataset("darentang/sroie")

        def process_sroie(examples):
            tokens = examples.get("tokens", examples.get("words", []))
            bboxes = examples.get("bboxes", [])
            ner_tags = examples.get("ner_tags", [])
            if isinstance(tokens, str):
                tokens = [tokens]
            if isinstance(bboxes, str):
                bboxes = [bboxes]
            if isinstance(ner_tags, str):
                ner_tags = [ner_tags]
            return {"tokens": tokens, "bboxes": bboxes, "ner_tags": ner_tags}

        train = ds["train"].map(process_sroie)
        test = ds["test"].map(process_sroie)
        return train, test, 4

    raise ValueError(f"Unknown dataset: {dataset_name}")


def ocr_confusion_sub(word: str) -> str:
    table = {"0": "O", "O": "0", "1": "l", "l": "1", "5": "S", "S": "5", "c": "e", "e": "c"}
    for k, v in table.items():
        if k in word:
            i = word.find(k)
            return word[:i] + v + word[i + len(k) :]
    return word


def swap_two_boxes(boxes, rng):
    if len(boxes) < 2:
        return boxes
    boxes = [list(b) for b in boxes]
    i, j = rng.choice(len(boxes), size=2, replace=False)
    boxes[i], boxes[j] = boxes[j], boxes[i]
    return boxes


class OCRNoiseTransform:
    def __init__(self, p=0.5, noise_type="substitution", seed=42, box_swap_quality=0.5):
        self.p = p
        self.noise_type = noise_type
        self.seed = seed
        self.box_swap_quality = box_swap_quality

    def __call__(self, words, boxes, sample_idx=0):
        rng = np.random.RandomState(self.seed + sample_idx)
        words = list(words)
        boxes = [list(b) for b in boxes]

        if self.p == 0.0 or self.noise_type == "none":
            return words, [Config.clean_token_weight] * len(words), boxes

        if self.noise_type == "box_swap":
            if rng.random() < self.p:
                boxes = swap_two_boxes(boxes, rng)
                q = [self.box_swap_quality] * len(words)
            else:
                q = [Config.clean_token_weight] * len(words)
            return words, q, boxes

        out, q = [], []
        for w in words:
            if len(w) and rng.random() < self.p:
                if self.noise_type == "substitution":
                    pos = rng.randint(0, len(w))
                    w = w[:pos] + chr(ord("a") + rng.randint(0, 26)) + w[pos + 1 :]
                elif self.noise_type == "ocr_confusion":
                    w = ocr_confusion_sub(w)
                q.append(Config.corrupted_token_weight)
            else:
                q.append(Config.clean_token_weight)
            out.append(w)
        return out, q, boxes


class DocumentDatasetWithNoise(Dataset):
    def __init__(
        self,
        hf_dataset,
        tokenizer,
        p=0.0,
        max_len=512,
        noise_type="substitution",
        seed=42,
        box_swap_quality=0.5,
    ):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.noise = OCRNoiseTransform(p, noise_type, seed, box_swap_quality)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        words, boxes, labels = ex["tokens"], ex["bboxes"], ex["ner_tags"]
        labels = [l if isinstance(l, int) and 0 <= l < Config.num_labels else -100 for l in labels]
        words, word_q, boxes = self.noise(words, boxes, sample_idx=idx)

        # Tokenize with offset mapping to obtain reliable word_ids for quality mask.
        enc_with_ids = self.tokenizer(
            words,
            boxes=boxes,
            word_labels=labels,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_offsets_mapping=True,
            return_tensors=None,
        )
        word_ids = enc_with_ids.word_ids()
        if word_ids is None:
            raise RuntimeError(
                "tokenizer returned no word_ids; token_quality_mask cannot be built"
            )

        token_q = torch.ones(self.max_len, dtype=torch.float)
        for i, wid in enumerate(word_ids):
            if wid is None:
                token_q[i] = 0.0
            elif wid < len(word_q):
                token_q[i] = word_q[wid]
            else:
                token_q[i] = 0.0

        enc = self.tokenizer(
            words,
            boxes=boxes,
            word_labels=labels,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "bbox": enc["bbox"].squeeze(0),
            "labels": enc["labels"].squeeze(0),
            "token_quality_mask": token_q,
        }


def ce_with_weights(logits, labels, weights=None, ignore_index=-100):
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=ignore_index,
        reduction="none",
    )
    mask = (labels.view(-1) != ignore_index).float()
    if weights is not None:
        mask = mask * weights.view(-1)
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


def get_loss_fn(loss_type: str):
    if loss_type == "cross_entropy":
        return ce_with_weights
    if loss_type == "label_smoothing":

        def _ls(logits, labels, weights=None, ignore_index=-100):
            n, c = logits.size(0) * logits.size(1), logits.size(-1)
            lf = logits.view(n, c)
            y = labels.view(n)
            mask = (y != ignore_index).float()
            logp = F.log_softmax(lf, dim=-1)
            with torch.no_grad():
                t = torch.full_like(logp, Config.label_smoothing_eps / c)
                valid = y != ignore_index
                t[valid] = t[valid].scatter(1, y[valid].unsqueeze(1), 1.0 - Config.label_smoothing_eps)
            loss = -(t * logp).sum(-1)
            if weights is not None:
                mask = mask * weights.view(-1)
            return (loss * mask).sum() / mask.sum().clamp(min=1.0)

        return _ls
    if loss_type == "focal":

        def _focal(logits, labels, weights=None, ignore_index=-100):
            ce = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=ignore_index,
                reduction="none",
            )
            loss = ((1 - torch.exp(-ce)) ** Config.focal_gamma) * ce
            mask = (labels.view(-1) != ignore_index).float()
            if weights is not None:
                mask = mask * weights.view(-1)
            return (loss * mask).sum() / mask.sum().clamp(min=1.0)

        return _focal
    raise ValueError(loss_type)


def quality_to_clean_target(token_quality: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map quality weights to gate targets.

    Gate semantics: g≈1 → use LoRA residual; g≈0 → rely on frozen base.
    Clean tokens → target 1; corrupted → target 0; padding (q=0) ignored.
    """
    valid = token_quality > 0
    # Strict: only exact clean weight is "clean"; anything else positive is corrupted.
    clean = (token_quality >= Config.clean_token_weight - 1e-6).float()
    return clean, valid.float()


def gate_aux_loss(gates: torch.Tensor, token_quality: torch.Tensor) -> torch.Tensor:
    """BCE encouraging low gate on corrupted tokens, high gate on clean tokens."""
    target, valid = quality_to_clean_target(token_quality)
    if valid.sum() < 1:
        return gates.new_zeros(())
    return F.binary_cross_entropy(gates.clamp(1e-6, 1 - 1e-6), target, weight=valid, reduction="sum") / valid.sum()


# -----------------------------------------------------------------------------
# MODELS — shared shell so head/backbone cannot explain gaps
# -----------------------------------------------------------------------------
def _lora_cfg():
    return LoraConfig(
        r=Config.lora_r,
        lora_alpha=Config.lora_alpha,
        target_modules=["query", "value"],
        lora_dropout=Config.lora_dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )


class _SharedLoRAShell(nn.Module):
    """LayoutLMv3Model + LoRA + Linear classifier. Used by all methods."""

    def __init__(self, num_labels: int):
        super().__init__()
        self.backbone = LayoutLMv3Model.from_pretrained(Config.backbone_name)
        self.hidden_size = self.backbone.config.hidden_size
        self.backbone = get_peft_model(self.backbone, _lora_cfg())
        self.classifier = nn.Linear(self.hidden_size, num_labels)

    def _encode(self, input_ids, bbox, attention_mask, adapters_on: bool):
        if adapters_on:
            out = self.backbone(
                input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, return_dict=True
            )
        else:
            with self.backbone.disable_adapter():
                out = self.backbone(
                    input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, return_dict=True
                )
        return out.last_hidden_state


class StandardLoRAModel(_SharedLoRAShell):
    """Matched baseline: same backbone+head as Router/Gate; adapters always on; no gate."""

    def forward(self, input_ids, bbox, attention_mask=None, return_gates=False):
        h = self._encode(input_ids, bbox, attention_mask, True)
        logits = self.classifier(h)
        if return_gates:
            return logits, None
        return logits


class RouterLoRAModel(_SharedLoRAShell):
    """Prior router: gate(h_lora, bbox). No aux loss."""

    def __init__(self, num_labels: int):
        super().__init__(num_labels)
        self.gate = nn.Sequential(
            nn.Linear(self.hidden_size + 4, Config.gate_hidden),
            nn.ReLU(),
            nn.Dropout(Config.gate_dropout),
            nn.Linear(Config.gate_hidden, 1),
        )

    def forward(self, input_ids, bbox, attention_mask=None, return_gates=False):
        h_lora = self._encode(input_ids, bbox, attention_mask, True)
        h_base = self._encode(input_ids, bbox, attention_mask, False)
        bbox_feat = bbox.float() / 1000.0
        g = torch.sigmoid(self.gate(torch.cat([h_lora, bbox_feat], dim=-1)))
        h = h_base + g * (h_lora - h_base)
        logits = self.classifier(h)
        if return_gates:
            return logits, g.squeeze(-1)
        return logits


class GateLoRAModel(_SharedLoRAShell):
    """GateLoRA: h = h_base + g * delta, with richer gate features.

    Gate inputs: [h_base, delta, ||delta||, bbox/1000].
    Aux BCE (when lambda_gate > 0): clean→g=1, corrupted→g=0.
    """

    def __init__(self, num_labels: int):
        super().__init__(num_labels)
        gate_in = 2 * self.hidden_size + 1 + 4
        self.gate = nn.Sequential(
            nn.Linear(gate_in, Config.gate_hidden),
            nn.ReLU(),
            nn.Dropout(Config.gate_dropout),
            nn.Linear(Config.gate_hidden, 1),
        )

    def forward(self, input_ids, bbox, attention_mask=None, return_gates=False):
        h_lora = self._encode(input_ids, bbox, attention_mask, True)
        h_base = self._encode(input_ids, bbox, attention_mask, False)
        delta = h_lora - h_base
        delta_norm = delta.norm(dim=-1, keepdim=True) / (float(self.hidden_size) ** 0.5)
        bbox_feat = bbox.float() / 1000.0
        gate_in = torch.cat([h_base, delta, delta_norm, bbox_feat], dim=-1)
        g = torch.sigmoid(self.gate(gate_in))
        h = h_base + g * delta
        logits = self.classifier(h)
        if return_gates:
            return logits, g.squeeze(-1)
        return logits


# -----------------------------------------------------------------------------
# TRAIN / EVAL — matched optimizers and freeze pattern for ALL methods
# -----------------------------------------------------------------------------
def _trainable_setup(model):
    for name, p in model.backbone.named_parameters():
        p.requires_grad = "lora" in name
    for p in model.classifier.parameters():
        p.requires_grad = True
    if hasattr(model, "gate"):
        for p in model.gate.parameters():
            p.requires_grad = True


def _make_optim(model):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = AdamW(params, lr=Config.lr, weight_decay=Config.weight_decay)
    steps = max(1, Config._steps_per_run)
    sched = get_linear_schedule_with_warmup(opt, int(steps * 0.1), steps)
    return opt, sched


def train_task_only(model, train_loader, loss_type="cross_entropy"):
    """Shared trainer for StandardLoRA and RouterLoRA (unweighted task CE)."""
    _trainable_setup(model)
    opt, sched = _make_optim(model)
    loss_fn = get_loss_fn(loss_type)
    for _ in range(Config.epochs):
        model.train()
        for batch in train_loader:
            ids = batch["input_ids"].to(Config.device)
            attn = batch["attention_mask"].to(Config.device)
            bbox = batch["bbox"].to(Config.device)
            labels = batch["labels"].to(Config.device)
            logits = model(ids, bbox, attn)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()


def train_gatelora(model, train_loader, loss_type="cross_entropy", lambda_gate: Optional[float] = None):
    """Same optimizer as baselines; optional fixed-weight aux gate BCE."""
    _trainable_setup(model)
    opt, sched = _make_optim(model)
    loss_fn = get_loss_fn(loss_type)
    lam = Config.lambda_gate if lambda_gate is None else lambda_gate
    for _ in range(Config.epochs):
        model.train()
        for batch in train_loader:
            ids = batch["input_ids"].to(Config.device)
            attn = batch["attention_mask"].to(Config.device)
            bbox = batch["bbox"].to(Config.device)
            labels = batch["labels"].to(Config.device)
            tq = batch["token_quality_mask"].to(Config.device)
            logits, gates = model(ids, bbox, attn, return_gates=True)
            task = loss_fn(logits, labels)
            if lam > 0:
                loss = task + lam * gate_aux_loss(gates, tq)
            else:
                loss = task
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()


def evaluate(model, test_loader) -> Tuple[float, float]:
    model.eval()
    preds, gold = [], []
    with torch.no_grad():
        for batch in test_loader:
            ids = batch["input_ids"].to(Config.device)
            attn = batch["attention_mask"].to(Config.device)
            bbox = batch["bbox"].to(Config.device)
            labels = batch["labels"].to(Config.device)
            logits = model(ids, bbox, attn)
            valid = labels != -100
            preds.extend(logits.argmax(-1)[valid].cpu().numpy())
            gold.extend(labels[valid].cpu().numpy())
    return (
        float(f1_score(gold, preds, average="weighted", zero_division=0)),
        float(f1_score(gold, preds, average="macro", zero_division=0)),
    )


def evaluate_gate_corruption_auc(model, loader) -> Optional[float]:
    """AUC of (1-g) vs synthetic corruption on a held-out noisy loader (not train)."""
    if not hasattr(model, "gate"):
        return None
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(Config.device)
            attn = batch["attention_mask"].to(Config.device)
            bbox = batch["bbox"].to(Config.device)
            tq = batch["token_quality_mask"].to(Config.device)
            _, gates = model(ids, bbox, attn, return_gates=True)
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
    return float(roc_auc_score(labels, scores))


def make_seeded_loader(dataset, seed, batch_size, shuffle=True):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=g)


def _stats(vals):
    return float(np.mean(vals)), float(np.std(vals))


def run_compare(config_override: Optional[dict] = None, include_no_aux: bool = False) -> dict:
    if config_override:
        for k, v in config_override.items():
            setattr(Config, k, v)

    methods = "StandardLoRA | RouterLoRA | GateLoRA"
    if include_no_aux:
        methods += " | GateLoRA-NoAux"

    print(f"Device: {Config.device}")
    print("=" * 60)
    print(
        f"GateLoRA COMPARE | {Config.dataset_name} | {Config.corruption_type} @ "
        f"{Config.corruption_rate * 100:.0f}% | {Config.loss_type}"
    )
    print(f"epochs={Config.epochs} seeds={Config.seeds} lambda_gate={Config.lambda_gate}")
    print(f"Methods: {methods}")
    print("Shell: LayoutLMv3Model + LoRA + Linear head (matched for all)")
    print("Eval: clean test + noisy test; gate AUC on noisy test (held-out)")
    print("Fairness: identical LR/WD/epochs/LoRA/freeze/data; unweighted task CE")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(Config.backbone_name)
    train_data, test_data, num_labels = load_and_prepare_dataset(
        Config.dataset_name, tokenizer, Config.max_seq_length
    )
    Config.num_labels = num_labels

    base_seed = 42
    noisy_train = DocumentDatasetWithNoise(
        train_data,
        tokenizer,
        p=Config.corruption_rate,
        max_len=Config.max_seq_length,
        noise_type=Config.corruption_type,
        seed=base_seed,
        box_swap_quality=Config.box_swap_quality,
    )
    test_clean = DocumentDatasetWithNoise(
        test_data,
        tokenizer,
        p=0.0,
        max_len=Config.max_seq_length,
        noise_type=Config.corruption_type,
        seed=base_seed,
        box_swap_quality=Config.box_swap_quality,
    )
    test_noisy = DocumentDatasetWithNoise(
        test_data,
        tokenizer,
        p=Config.corruption_rate,
        max_len=Config.max_seq_length,
        noise_type=Config.corruption_type,
        seed=base_seed,
        box_swap_quality=Config.box_swap_quality,
    )

    probe_loader = make_seeded_loader(noisy_train, Config.seeds[0], Config.batch_size, True)
    Config._steps_per_run = max(1, len(probe_loader) * Config.epochs)

    per_seed = []
    bucket_keys = [
        "lora_cw", "lora_cm", "lora_nw", "lora_nm",
        "router_cw", "router_cm", "router_nw", "router_nm",
        "gate_cw", "gate_cm", "gate_nw", "gate_nm",
        "router_auc", "gate_auc",
    ]
    if include_no_aux:
        bucket_keys += ["noaux_cw", "noaux_cm", "noaux_nw", "noaux_nm", "noaux_auc"]
    buckets: Dict[str, list] = {k: [] for k in bucket_keys}

    for seed in list(Config.seeds):
        set_seed(seed)
        print(f"\n--- Seed {seed} ---")
        train_loader = make_seeded_loader(noisy_train, seed, Config.batch_size, True)
        clean_loader = make_seeded_loader(test_clean, seed, Config.batch_size, False)
        noisy_loader = make_seeded_loader(test_noisy, seed, Config.batch_size, False)

        print("  Training StandardLoRA (matched shell)...")
        m = StandardLoRAModel(num_labels).to(Config.device)
        train_task_only(m, train_loader, Config.loss_type)
        lcw, lcm = evaluate(m, clean_loader)
        lnw, lnm = evaluate(m, noisy_loader)
        del m
        clear_cuda()

        print("  Training RouterLoRA...")
        m = RouterLoRAModel(num_labels).to(Config.device)
        train_task_only(m, train_loader, Config.loss_type)
        rcw, rcm = evaluate(m, clean_loader)
        rnw, rnm = evaluate(m, noisy_loader)
        r_auc = evaluate_gate_corruption_auc(m, noisy_loader)
        del m
        clear_cuda()

        print("  Training GateLoRA...")
        m = GateLoRAModel(num_labels).to(Config.device)
        train_gatelora(m, train_loader, Config.loss_type, lambda_gate=Config.lambda_gate)
        gcw, gcm = evaluate(m, clean_loader)
        gnw, gnm = evaluate(m, noisy_loader)
        g_auc = evaluate_gate_corruption_auc(m, noisy_loader)
        del m
        clear_cuda()

        nacw = nacm = nanw = nanm = None
        na_auc = None
        if include_no_aux:
            print("  Training GateLoRA-NoAux (ablation, lambda=0)...")
            m = GateLoRAModel(num_labels).to(Config.device)
            train_gatelora(m, train_loader, Config.loss_type, lambda_gate=0.0)
            nacw, nacm = evaluate(m, clean_loader)
            nanw, nanm = evaluate(m, noisy_loader)
            na_auc = evaluate_gate_corruption_auc(m, noisy_loader)
            del m
            clear_cuda()

        print(f"  LoRA      clean W/M={lcw:.4f}/{lcm:.4f}  noisy W/M={lnw:.4f}/{lnm:.4f}")
        print(f"  Router    clean W/M={rcw:.4f}/{rcm:.4f}  noisy W/M={rnw:.4f}/{rnm:.4f}  gateAUC={r_auc}")
        print(f"  GateLoRA  clean W/M={gcw:.4f}/{gcm:.4f}  noisy W/M={gnw:.4f}/{gnm:.4f}  gateAUC={g_auc}")
        if include_no_aux:
            print(
                f"  NoAux     clean W/M={nacw:.4f}/{nacm:.4f}  "
                f"noisy W/M={nanw:.4f}/{nanm:.4f}  gateAUC={na_auc}"
            )

        row = {
            "seed": seed,
            "corruption_rate": Config.corruption_rate,
            "lora_clean_weighted": lcw,
            "lora_clean_macro": lcm,
            "lora_noisy_weighted": lnw,
            "lora_noisy_macro": lnm,
            "router_clean_weighted": rcw,
            "router_clean_macro": rcm,
            "router_noisy_weighted": rnw,
            "router_noisy_macro": rnm,
            "router_gate_auc": r_auc,
            "gatelora_clean_weighted": gcw,
            "gatelora_clean_macro": gcm,
            "gatelora_noisy_weighted": gnw,
            "gatelora_noisy_macro": gnm,
            "gatelora_gate_auc": g_auc,
        }
        if include_no_aux:
            row.update(
                {
                    "gatelora_noaux_clean_weighted": nacw,
                    "gatelora_noaux_clean_macro": nacm,
                    "gatelora_noaux_noisy_weighted": nanw,
                    "gatelora_noaux_noisy_macro": nanm,
                    "gatelora_noaux_gate_auc": na_auc,
                }
            )
        per_seed.append(row)
        buckets["lora_cw"].append(lcw)
        buckets["lora_cm"].append(lcm)
        buckets["lora_nw"].append(lnw)
        buckets["lora_nm"].append(lnm)
        buckets["router_cw"].append(rcw)
        buckets["router_cm"].append(rcm)
        buckets["router_nw"].append(rnw)
        buckets["router_nm"].append(rnm)
        buckets["gate_cw"].append(gcw)
        buckets["gate_cm"].append(gcm)
        buckets["gate_nw"].append(gnw)
        buckets["gate_nm"].append(gnm)
        if r_auc is not None:
            buckets["router_auc"].append(r_auc)
        if g_auc is not None:
            buckets["gate_auc"].append(g_auc)
        if include_no_aux:
            buckets["noaux_cw"].append(nacw)
            buckets["noaux_cm"].append(nacm)
            buckets["noaux_nw"].append(nanw)
            buckets["noaux_nm"].append(nanm)
            if na_auc is not None:
                buckets["noaux_auc"].append(na_auc)

    def pack(key):
        return _stats(buckets[key])

    lcw_m, lcw_s = pack("lora_cw")
    lcm_m, lcm_s = pack("lora_cm")
    lnw_m, lnw_s = pack("lora_nw")
    lnm_m, lnm_s = pack("lora_nm")
    rcw_m, rcw_s = pack("router_cw")
    rcm_m, rcm_s = pack("router_cm")
    rnw_m, rnw_s = pack("router_nw")
    rnm_m, rnm_s = pack("router_nm")
    gcw_m, gcw_s = pack("gate_cw")
    gcm_m, gcm_s = pack("gate_cm")
    gnw_m, gnw_s = pack("gate_nw")
    gnm_m, gnm_s = pack("gate_nm")
    ra_m, ra_s = _stats(buckets["router_auc"]) if buckets["router_auc"] else (float("nan"), float("nan"))
    ga_m, ga_s = _stats(buckets["gate_auc"]) if buckets["gate_auc"] else (float("nan"), float("nan"))

    print("\n" + "=" * 60)
    print("SUMMARY (mean ± std)")
    print("=" * 60)
    print(f"LoRA     clean W={lcw_m:.4f}±{lcw_s:.4f}  M={lcm_m:.4f}±{lcm_s:.4f}")
    print(f"         noisy W={lnw_m:.4f}±{lnw_s:.4f}  M={lnm_m:.4f}±{lnm_s:.4f}")
    print(f"Router   clean W={rcw_m:.4f}±{rcw_s:.4f}  M={rcm_m:.4f}±{rcm_s:.4f}")
    print(f"         noisy W={rnw_m:.4f}±{rnw_s:.4f}  M={rnm_m:.4f}±{rnm_s:.4f}  gateAUC={ra_m:.4f}±{ra_s:.4f}")
    print(f"GateLoRA clean W={gcw_m:.4f}±{gcw_s:.4f}  M={gcm_m:.4f}±{gcm_s:.4f}")
    print(f"         noisy W={gnw_m:.4f}±{gnw_s:.4f}  M={gnm_m:.4f}±{gnm_s:.4f}  gateAUC={ga_m:.4f}±{ga_s:.4f}")
    print(f"Gate - LoRA  clean W/M: {gcw_m - lcw_m:+.4f} / {gcm_m - lcm_m:+.4f}")
    print(f"Gate - LoRA  noisy W/M: {gnw_m - lnw_m:+.4f} / {gnm_m - lnm_m:+.4f}")
    print(f"Gate - Router clean W/M: {gcw_m - rcw_m:+.4f} / {gcm_m - rcm_m:+.4f}")
    if include_no_aux:
        nacw_m, nacw_s = pack("noaux_cw")
        nacm_m, nacm_s = pack("noaux_cm")
        print(f"NoAux    clean W={nacw_m:.4f}±{nacw_s:.4f}  M={nacm_m:.4f}±{nacm_s:.4f}")
        print(f"Gate - NoAux clean W/M: {gcw_m - nacw_m:+.4f} / {gcm_m - nacm_m:+.4f}")
    print("=" * 60)

    summary = {
        "dataset": Config.dataset_name,
        "corruption_type": Config.corruption_type,
        "corruption_rate": Config.corruption_rate,
        "loss_type": Config.loss_type,
        "epochs": Config.epochs,
        "seeds": list(Config.seeds),
        "lambda_gate": Config.lambda_gate,
        "architecture_matched": True,
        "backbone_shell": "LayoutLMv3Model+LoRA+Linear",
        "lr": Config.lr,
        "lora_r": Config.lora_r,
        "lora_alpha": Config.lora_alpha,
        "lora_clean_weighted_mean": lcw_m,
        "lora_clean_weighted_std": lcw_s,
        "lora_clean_macro_mean": lcm_m,
        "lora_clean_macro_std": lcm_s,
        "lora_noisy_weighted_mean": lnw_m,
        "lora_noisy_weighted_std": lnw_s,
        "lora_noisy_macro_mean": lnm_m,
        "lora_noisy_macro_std": lnm_s,
        "router_clean_weighted_mean": rcw_m,
        "router_clean_weighted_std": rcw_s,
        "router_clean_macro_mean": rcm_m,
        "router_clean_macro_std": rcm_s,
        "router_noisy_weighted_mean": rnw_m,
        "router_noisy_weighted_std": rnw_s,
        "router_noisy_macro_mean": rnm_m,
        "router_noisy_macro_std": rnm_s,
        "router_gate_auc_mean": ra_m,
        "router_gate_auc_std": ra_s,
        "gatelora_clean_weighted_mean": gcw_m,
        "gatelora_clean_weighted_std": gcw_s,
        "gatelora_clean_macro_mean": gcm_m,
        "gatelora_clean_macro_std": gcm_s,
        "gatelora_noisy_weighted_mean": gnw_m,
        "gatelora_noisy_weighted_std": gnw_s,
        "gatelora_noisy_macro_mean": gnm_m,
        "gatelora_noisy_macro_std": gnm_s,
        "gatelora_gate_auc_mean": ga_m,
        "gatelora_gate_auc_std": ga_s,
        "gatelora_minus_lora_clean_weighted": gcw_m - lcw_m,
        "gatelora_minus_lora_clean_macro": gcm_m - lcm_m,
        "gatelora_minus_lora_noisy_weighted": gnw_m - lnw_m,
        "gatelora_minus_lora_noisy_macro": gnm_m - lnm_m,
        "gatelora_minus_router_clean_weighted": gcw_m - rcw_m,
        "gatelora_minus_router_clean_macro": gcm_m - rcm_m,
        "include_no_aux": include_no_aux,
    }
    if include_no_aux:
        nacw_m, nacw_s = pack("noaux_cw")
        nacm_m, nacm_s = pack("noaux_cm")
        nanw_m, nanw_s = pack("noaux_nw")
        nanm_m, nanm_s = pack("noaux_nm")
        naa_m, naa_s = (
            _stats(buckets["noaux_auc"]) if buckets["noaux_auc"] else (float("nan"), float("nan"))
        )
        summary.update(
            {
                "gatelora_noaux_clean_weighted_mean": nacw_m,
                "gatelora_noaux_clean_weighted_std": nacw_s,
                "gatelora_noaux_clean_macro_mean": nacm_m,
                "gatelora_noaux_clean_macro_std": nacm_s,
                "gatelora_noaux_noisy_weighted_mean": nanw_m,
                "gatelora_noaux_noisy_weighted_std": nanw_s,
                "gatelora_noaux_noisy_macro_mean": nanm_m,
                "gatelora_noaux_noisy_macro_std": nanm_s,
                "gatelora_noaux_gate_auc_mean": naa_m,
                "gatelora_noaux_gate_auc_std": naa_s,
                "gatelora_minus_noaux_clean_weighted": gcw_m - nacw_m,
                "gatelora_minus_noaux_clean_macro": gcm_m - nacm_m,
            }
        )
    return {"summary": summary, "per_seed": per_seed}


def parse_args():
    p = argparse.ArgumentParser(description="Architecture-matched GateLoRA comparison")
    p.add_argument("--dataset", type=str, default="funsd", choices=["funsd", "cord", "sroie"])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--corruption-rate", type=float, default=None, help="Single rate (default 0.7)")
    p.add_argument(
        "--corruption-rates",
        type=float,
        nargs="+",
        default=None,
        help="Sweep rates, e.g. 0.0 0.1 0.3 0.5 0.7",
    )
    p.add_argument("--lambda-gate", type=float, default=None, help="Fixed aux weight (default 0.5)")
    p.add_argument(
        "--include-no-aux",
        action="store_true",
        help="Also run GateLoRA with lambda_gate=0 (feature ablation)",
    )
    p.add_argument("--out", type=str, default="gatelora_results.csv")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.corruption_rates is not None:
        rates: List[float] = list(args.corruption_rates)
    else:
        rates = [0.7 if args.corruption_rate is None else args.corruption_rate]

    all_summaries = []
    all_per_seed = []
    for rate in rates:
        override = {
            "dataset_name": args.dataset,
            "corruption_type": "substitution",
            "corruption_rate": rate,
            "loss_type": "cross_entropy",
            "epochs": 15 if args.epochs is None else args.epochs,
            "seeds": [42, 123, 2024, 456, 789] if args.seeds is None else args.seeds,
        }
        if args.lambda_gate is not None:
            override["lambda_gate"] = args.lambda_gate

        out = run_compare(override, include_no_aux=args.include_no_aux)
        all_summaries.append(out["summary"])
        all_per_seed.extend(out["per_seed"])

    per_path = args.out.replace(".csv", "_per_seed.csv")
    json_path = args.out.replace(".csv", "_summary.json")
    pd.DataFrame(all_per_seed).to_csv(per_path, index=False)
    pd.DataFrame(all_summaries).to_csv(args.out, index=False)
    with open(json_path, "w") as f:
        json.dump(all_summaries if len(all_summaries) > 1 else all_summaries[0], f, indent=2)
    print(f"\nWrote {args.out}, {per_path}, {json_path}")
