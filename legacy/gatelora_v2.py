"""GateLoRA v2 — follow-up variants after hard-aux residual gating failed.

Round-1 finding (FUNSD, matched protocol):
  - Hard aux BCE (corrupt→0, lambda=0.5) raises gate-AUC but hurts clean macro F1.
  - RouterLoRA (no aux) matches/beats LoRA; GateLoRA-NoAux still lags.

v2 hypothesis tests (same locked Config; only architecture/objective changes):
  1. StandardLoRA   — matched shell baseline
  2. RouterRich     — rich gate features [h_base, Δ, ||Δ||, bbox], residual mix, NO aux
  3. SoftResidual   — residual mix + soft aux (corrupt→0.3, clean→1) + λ=0.1 with warmup
  4. GateReweight   — forward uses h_lora only; gate reweights token CE (no residual mix)
  5. GateConcat     — classify concat(h_lora, g); no residual mix, no aux

Fairness (unchanged from v1):
  - Same LR / WD / epochs / LoRA r,α / batch / seeds / data / freeze pattern.
  - No per-method hyperparameter search after seeing results.
  - Soft-aux targets and λ are fixed method constants (pre-registered).
  - Synthetic corruption labels used only for SoftResidual aux; never at test for preds.

Usage (0.7 only, 5 seeds — not a full rate sweep):
  python -u gatelora_v2.py --dataset funsd \\
      --out gatelora_v2_funsd07.csv
  # quick smoke (do not report):
  python -u gatelora_v2.py --epochs 2 --seeds 42 \\
      --out gatelora_v2_smoke.csv
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
# CONFIG — locked protocol (+ fixed v2 method constants; not tuned per seed)
# -----------------------------------------------------------------------------
class Config:
    backbone_name = "microsoft/layoutlmv3-base"
    max_seq_length = 512
    num_labels = 7

    batch_size = 4
    epochs = 15
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.1
    lr = 2e-4
    weight_decay = 0.01
    seeds = [42, 123, 2024, 456, 789]

    gate_hidden = 256
    gate_dropout = 0.1
    box_swap_quality = 0.5

    # SoftResidual: soft targets + weak scheduled aux (pre-registered, not swept).
    lambda_gate = 0.1
    soft_corrupt_target = 0.3  # corrupted → 0.3 (not 0); clean → 1.0
    aux_warmup_epochs = 3      # λ=0 for first N epochs, then lambda_gate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_name = "funsd"
    corruption_type = "substitution"
    corruption_rate = 0.7
    loss_type = "cross_entropy"

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
# DATA — identical to gatelora.py / sallora.py
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

        return ds["train"].map(process_sroie), ds["test"].map(process_sroie), 4
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
        self, hf_dataset, tokenizer, p=0.0, max_len=512, noise_type="substitution", seed=42, box_swap_quality=0.5
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
            raise RuntimeError("tokenizer returned no word_ids; token_quality_mask cannot be built")

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


# -----------------------------------------------------------------------------
# LOSSES
# -----------------------------------------------------------------------------
def ce_tokenwise(logits, labels, ignore_index=-100):
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).view_as(labels)


def ce_mean(logits, labels, ignore_index=-100):
    loss = ce_tokenwise(logits, labels, ignore_index)
    mask = (labels != ignore_index).float()
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


def ce_gate_reweight(logits, labels, gates, ignore_index=-100):
    """L = sum_t g_t * CE_t / sum_t g_t  (valid tokens only)."""
    loss = ce_tokenwise(logits, labels, ignore_index)
    mask = (labels != ignore_index).float()
    w = gates * mask
    return (loss * w).sum() / w.sum().clamp(min=1.0)


def soft_gate_aux_loss(gates: torch.Tensor, token_quality: torch.Tensor) -> torch.Tensor:
    """Soft BCE: clean→1.0, corrupted→soft_corrupt_target (default 0.3), pad ignored."""
    valid = (token_quality > 0).float()
    clean = (token_quality >= Config.clean_token_weight - 1e-6).float()
    target = clean + (1.0 - clean) * Config.soft_corrupt_target
    if valid.sum() < 1:
        return gates.new_zeros(())
    return F.binary_cross_entropy(
        gates.clamp(1e-6, 1 - 1e-6), target, weight=valid, reduction="sum"
    ) / valid.sum()


def quality_to_clean_target(token_quality: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    valid = token_quality > 0
    clean = (token_quality >= Config.clean_token_weight - 1e-6).float()
    return clean, valid.float()


# -----------------------------------------------------------------------------
# MODELS — shared shell
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
    def __init__(self, num_labels: int, classifier_in: Optional[int] = None):
        super().__init__()
        self.backbone = LayoutLMv3Model.from_pretrained(Config.backbone_name)
        self.hidden_size = self.backbone.config.hidden_size
        self.backbone = get_peft_model(self.backbone, _lora_cfg())
        cin = self.hidden_size if classifier_in is None else classifier_in
        self.classifier = nn.Linear(cin, num_labels)

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

    def _rich_gate_features(self, h_lora, h_base, bbox):
        delta = h_lora - h_base
        delta_norm = delta.norm(dim=-1, keepdim=True) / (float(self.hidden_size) ** 0.5)
        bbox_feat = bbox.float() / 1000.0
        return torch.cat([h_base, delta, delta_norm, bbox_feat], dim=-1), delta


class StandardLoRAModel(_SharedLoRAShell):
    def forward(self, input_ids, bbox, attention_mask=None, return_gates=False):
        h = self._encode(input_ids, bbox, attention_mask, True)
        logits = self.classifier(h)
        if return_gates:
            return logits, None
        return logits


class RouterRichModel(_SharedLoRAShell):
    """Rich gate features + residual mix; NO aux (v2 primary constructive bet)."""

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
        feats, delta = self._rich_gate_features(h_lora, h_base, bbox)
        g = torch.sigmoid(self.gate(feats))
        h = h_base + g * delta
        logits = self.classifier(h)
        if return_gates:
            return logits, g.squeeze(-1)
        return logits


class SoftResidualModel(_SharedLoRAShell):
    """Residual mix + soft aux (corrupt→0.3) with scheduled λ."""

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
        feats, delta = self._rich_gate_features(h_lora, h_base, bbox)
        g = torch.sigmoid(self.gate(feats))
        h = h_base + g * delta
        logits = self.classifier(h)
        if return_gates:
            return logits, g.squeeze(-1)
        return logits


class GateReweightModel(_SharedLoRAShell):
    """Forward: h = h_lora. Gate only reweights token CE. Soft aux optional via trainer."""

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
        feats, _ = self._rich_gate_features(h_lora, h_base, bbox)
        g = torch.sigmoid(self.gate(feats))
        logits = self.classifier(h_lora)  # full LoRA representation
        if return_gates:
            return logits, g.squeeze(-1)
        return logits


class GateConcatModel(_SharedLoRAShell):
    """Classify concat(h_lora, g); no residual mix, no aux."""

    def __init__(self, num_labels: int):
        super().__init__(num_labels)
        self.classifier = nn.Linear(self.hidden_size + 1, num_labels)
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
        feats, _ = self._rich_gate_features(h_lora, h_base, bbox)
        g = torch.sigmoid(self.gate(feats))
        h = torch.cat([h_lora, g], dim=-1)
        logits = self.classifier(h)
        if return_gates:
            return logits, g.squeeze(-1)
        return logits


# -----------------------------------------------------------------------------
# TRAIN / EVAL
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


def train_task_only(model, train_loader):
    """Unweighted CE — StandardLoRA / RouterRich / GateConcat."""
    _trainable_setup(model)
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


def train_soft_residual(model, train_loader):
    """Residual mix + soft aux with epoch warmup schedule."""
    _trainable_setup(model)
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


def train_gate_reweight(model, train_loader):
    """h=h_lora; L = sum(g*CE)/sum(g). Soft aux with same warmup (stabilizes gate)."""
    _trainable_setup(model)
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
            task = ce_gate_reweight(logits, labels, gates)
            loss = task + lam * soft_gate_aux_loss(gates, tq) if lam > 0 else task
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


# -----------------------------------------------------------------------------
# COMPARE
# -----------------------------------------------------------------------------
ALL_METHODS = ("lora", "router_rich", "soft_residual", "gate_reweight", "gate_concat")


def _train_and_eval(name: str, num_labels: int, train_loader, clean_loader, noisy_loader):
    """Build model, train, evaluate; return (cw, cm, nw, nm, auc)."""
    if name == "lora":
        m = StandardLoRAModel(num_labels).to(Config.device)
        train_task_only(m, train_loader)
        auc = None
    elif name == "router_rich":
        m = RouterRichModel(num_labels).to(Config.device)
        train_task_only(m, train_loader)
        auc = evaluate_gate_corruption_auc(m, noisy_loader)
    elif name == "soft_residual":
        m = SoftResidualModel(num_labels).to(Config.device)
        train_soft_residual(m, train_loader)
        auc = evaluate_gate_corruption_auc(m, noisy_loader)
    elif name == "gate_reweight":
        m = GateReweightModel(num_labels).to(Config.device)
        train_gate_reweight(m, train_loader)
        auc = evaluate_gate_corruption_auc(m, noisy_loader)
    elif name == "gate_concat":
        m = GateConcatModel(num_labels).to(Config.device)
        train_task_only(m, train_loader)
        auc = evaluate_gate_corruption_auc(m, noisy_loader)
    else:
        raise ValueError(f"Unknown method: {name}")

    cw, cm = evaluate(m, clean_loader)
    nw, nm = evaluate(m, noisy_loader)
    del m
    clear_cuda()
    return cw, cm, nw, nm, auc


METHOD_LABELS = {
    "lora": "StandardLoRA",
    "router_rich": "RouterRich",
    "soft_residual": "SoftResidual",
    "gate_reweight": "GateReweight",
    "gate_concat": "GateConcat",
}


def run_compare(
    config_override: Optional[dict] = None,
    methods: Optional[List[str]] = None,
) -> dict:
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
        f"GateLoRA v2 | {Config.dataset_name} | {Config.corruption_type} @ "
        f"{Config.corruption_rate * 100:.0f}% | {Config.loss_type}"
    )
    print(f"epochs={Config.epochs} seeds={Config.seeds}")
    print(f"Methods: {' | '.join(METHOD_LABELS[m] for m in active)}")
    print("Shell: LayoutLMv3Model + LoRA + Linear (matched)")
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

        results = {}
        for name in active:
            print(f"  Training {METHOD_LABELS[name]}...")
            results[name] = _train_and_eval(name, num_labels, train_loader, clean_loader, noisy_loader)

        row = {"seed": seed, "corruption_rate": Config.corruption_rate, "corruption_type": Config.corruption_type}
        for name in active:
            cw, cm, nw, nm, auc = results[name]
            print(
                f"  {name:14s} clean W/M={cw:.4f}/{cm:.4f}  "
                f"noisy W/M={nw:.4f}/{nm:.4f}  gateAUC={auc}"
            )
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
        "loss_type": Config.loss_type,
        "epochs": Config.epochs,
        "seeds": list(Config.seeds),
        "methods": active,
        "architecture_matched": True,
        "protocol": "gatelora_v2",
        "lr": Config.lr,
        "lora_r": Config.lora_r,
        "lora_alpha": Config.lora_alpha,
    }

    lora_cm_m = lora_cw_m = None
    for name in active:
        cw_m, cw_s = _stats(buckets[f"{name}_cw"])
        cm_m, cm_s = _stats(buckets[f"{name}_cm"])
        nw_m, nw_s = _stats(buckets[f"{name}_nw"])
        nm_m, nm_s = _stats(buckets[f"{name}_nm"])
        if buckets[f"{name}_auc"]:
            auc_m, auc_s = _stats(buckets[f"{name}_auc"])
        else:
            auc_m, auc_s = float("nan"), float("nan")
        print(f"{name:14s} clean W={cw_m:.4f}±{cw_s:.4f}  M={cm_m:.4f}±{cm_s:.4f}")
        print(f"{'':14s} noisy W={nw_m:.4f}±{nw_s:.4f}  M={nm_m:.4f}±{nm_s:.4f}  "
              f"gateAUC={auc_m:.4f}±{auc_s:.4f}")
        summary[f"{name}_clean_weighted_mean"] = cw_m
        summary[f"{name}_clean_weighted_std"] = cw_s
        summary[f"{name}_clean_macro_mean"] = cm_m
        summary[f"{name}_clean_macro_std"] = cm_s
        summary[f"{name}_noisy_weighted_mean"] = nw_m
        summary[f"{name}_noisy_weighted_std"] = nw_s
        summary[f"{name}_noisy_macro_mean"] = nm_m
        summary[f"{name}_noisy_macro_std"] = nm_s
        summary[f"{name}_gate_auc_mean"] = auc_m
        summary[f"{name}_gate_auc_std"] = auc_s
        if name == "lora":
            lora_cm_m, lora_cw_m = cm_m, cw_m
        elif lora_cm_m is not None:
            summary[f"{name}_minus_lora_clean_weighted"] = cw_m - lora_cw_m
            summary[f"{name}_minus_lora_clean_macro"] = cm_m - lora_cm_m

    print("=" * 60)
    return {"summary": summary, "per_seed": per_seed}


def parse_args():
    p = argparse.ArgumentParser(description="GateLoRA v2 — matched comparison")
    p.add_argument("--dataset", type=str, default="funsd", choices=["funsd", "cord", "sroie"])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--corruption-rate", type=float, default=None, help="Single rate (default 0.7)")
    p.add_argument(
        "--corruption-rates",
        type=float,
        nargs="+",
        default=None,
        help="Sweep rates, e.g. 0.0 0.3 0.5 0.7",
    )
    p.add_argument(
        "--corruption-type",
        type=str,
        default="substitution",
        choices=["substitution", "ocr_confusion", "box_swap"],
    )
    p.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=None,
        help="Subset of: lora router_rich soft_residual gate_reweight gate_concat",
    )
    p.add_argument("--out", type=str, default="gatelora_v2_results.csv")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.corruption_rates is not None:
        rates: List[float] = list(args.corruption_rates)
    else:
        rates = [0.7 if args.corruption_rate is None else args.corruption_rate]

    methods = args.methods
    all_summaries = []
    all_per_seed = []
    for rate in rates:
        override = {
            "dataset_name": args.dataset,
            "corruption_type": args.corruption_type,
            "corruption_rate": rate,
            "loss_type": "cross_entropy",
            "epochs": 15 if args.epochs is None else args.epochs,
            "seeds": [42, 123, 2024, 456, 789] if args.seeds is None else args.seeds,
        }
        out = run_compare(override, methods=methods)
        all_summaries.append(out["summary"])
        all_per_seed.extend(out["per_seed"])

    per_path = args.out.replace(".csv", "_per_seed.csv")
    json_path = args.out.replace(".csv", "_summary.json")
    pd.DataFrame(all_per_seed).to_csv(per_path, index=False)
    pd.DataFrame(all_summaries).to_csv(args.out, index=False)
    with open(json_path, "w") as f:
        json.dump(all_summaries if len(all_summaries) > 1 else all_summaries[0], f, indent=2)
    print(f"\nWrote {args.out}, {per_path}, {json_path}")
