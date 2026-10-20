"""Training / evaluation loop — v4 versions, verbatim from gatelora_v4.py.

gatelora_v3.py keeps its own _trainable_setup / train_task_only /
evaluate_gate_corruption_auc because the v3 semantics differ in ways
that matter (e.g. ScalarConcat gate-AUC); see that file."""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F  # noqa: F401  (parity with original module)
from sklearn.metrics import f1_score, roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from .config import Config
from .losses import ce_mean, quality_to_clean_target
from .seed import clear_cuda  # noqa: F401  (re-exported; original lived beside these)


def _trainable_setup(model, train_gate=True):
    for name, p in model.backbone.named_parameters():
        p.requires_grad = "lora" in name
    for p in model.classifier.parameters():
        p.requires_grad = True
    if hasattr(model, "gate"):
        for p in model.gate.parameters():
            p.requires_grad = bool(train_gate)


def _make_optim(model):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = AdamW(params, lr=Config.lr, weight_decay=Config.weight_decay)
    steps = max(1, Config._steps_per_run)
    sched = get_linear_schedule_with_warmup(opt, int(steps * 0.1), steps)
    return opt, sched


def train_task_only(model, train_loader, train_gate=True):
    _trainable_setup(model, train_gate=train_gate)
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
    if getattr(model, "gate_auc_applicable", True) is False:
        return None
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
            out = model(ids, bbox, attn, return_gates=True)
            if not isinstance(out, tuple) or out[1] is None:
                return None
            gates = out[1]
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
        return None
    return float(roc_auc_score(labels, scores))


def make_seeded_loader(dataset, seed, batch_size, shuffle=True):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=g)


def _stats(vals):
    return float(np.mean(vals)), float(np.std(vals))
