"""Matched comparison: Standard LoRA vs SALLoRA vs RouterLoRA on noisy document NER.

Supports FUNSD, CORD, and SROIE datasets.

Usage:
  python sallora.py --mode compare --dataset funsd
  python sallora.py --mode compare --dataset cord
  python sallora.py --mode compare --dataset sroie
  python sallora.py --mode compare --dataset funsd --epochs 2 --seeds 42   # smoke test
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
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import (
    AutoTokenizer,
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Model,
    get_linear_schedule_with_warmup,
)
import warnings
warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
class Config:
    backbone_name = "microsoft/layoutlmv3-base"
    max_seq_length = 512
    num_labels = 7   # will be overridden per dataset

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_name = "funsd"          # can be overridden by CLI
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
# DATASET LOADERS (Unified)
# -----------------------------------------------------------------------------
def load_and_prepare_dataset(dataset_name: str, tokenizer, max_len: int):
    """
    Load dataset and return train/test splits with tokens, bboxes, ner_tags.
    Returns: train_split, test_split, num_labels
    """
    if dataset_name == "funsd":
        ds = load_dataset("nielsr/funsd-layoutlmv3")
        return ds["train"], ds["test"], 7

    elif dataset_name == "cord":
        # CORD v2 from naver-clova-ix/cord-v2
        ds = load_dataset("naver-clova-ix/cord-v2")
        # The dataset has 'ground_truth' as JSON strings
        def process_cord(examples):
            tokens_list = []
            bboxes_list = []
            ner_tags_list = []
            for gt_str in examples['ground_truth']:
                data = json.loads(gt_str)
                words = []
                boxes = []
                tags = []
                for line in data.get('valid_line', []):
                    for word_info in line.get('words', []):
                        words.append(word_info['text'])
                        # bbox is [x1, y1, x2, y2] but LayoutLMv3 expects [x1, y1, x2, y2] normalized? We'll keep as is.
                        boxes.append(word_info['bbox'])
                        tags.append(word_info['category'])
                tokens_list.append(words)
                bboxes_list.append(boxes)
                ner_tags_list.append(tags)
            return {'tokens': tokens_list, 'bboxes': bboxes_list, 'ner_tags': ner_tags_list}
        train = ds['train'].map(process_cord, batched=True, remove_columns=ds['train'].column_names)
        test = ds['test'].map(process_cord, batched=True, remove_columns=ds['test'].column_names)
        return train, test, 30   # CORD has 30 entity categories

    elif dataset_name == "sroie":
        # Use darentang/sroie (has tokens, bboxes, ner_tags)
        ds = load_dataset("darentang/sroie")
        def process_sroie(examples):
            tokens = examples.get('tokens', examples.get('words', []))
            bboxes = examples.get('bboxes', [])
            ner_tags = examples.get('ner_tags', [])
            # Ensure lists
            if isinstance(tokens, str):
                tokens = [tokens]
            if isinstance(bboxes, str):
                bboxes = [bboxes]
            if isinstance(ner_tags, str):
                ner_tags = [ner_tags]
            return {'tokens': tokens, 'bboxes': bboxes, 'ner_tags': ner_tags}
        train = ds['train'].map(process_sroie)
        test = ds['test'].map(process_sroie)
        return train, test, 4   # SROIE has 4 entity types

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# -----------------------------------------------------------------------------
# CORRUPTION, DATASET, LOSS, MODELS, TRAINING – identical to your code
# (We keep everything unchanged; only the dataset loader differs.)
# -----------------------------------------------------------------------------

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
    def __init__(self, hf_dataset, tokenizer, p=0.0, max_len=512, noise_type="substitution", seed=42, box_swap_quality=0.5):
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

        enc = self.tokenizer(
            words,
            boxes=boxes,
            word_labels=labels,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        word_ids = self.tokenizer(
            words,
            boxes=boxes,
            word_labels=labels,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
        ).word_ids()

        token_q = torch.ones(self.max_len, dtype=torch.float)
        for i, wid in enumerate(word_ids):
            if wid is None:
                token_q[i] = 0.0
            elif wid < len(word_q):
                token_q[i] = word_q[wid]
            else:
                token_q[i] = 0.0

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


def build_standard_lora(num_labels: int):
    base = LayoutLMv3ForTokenClassification.from_pretrained(
        Config.backbone_name, num_labels=num_labels, ignore_mismatched_sizes=True
    )
    cfg = LoraConfig(
        r=Config.lora_r,
        lora_alpha=Config.lora_alpha,
        target_modules=["query", "value"],
        lora_dropout=Config.lora_dropout,
        bias="none",
        task_type=TaskType.TOKEN_CLS,
    )
    return get_peft_model(base, cfg).to(Config.device)


def train_standard_lora(model, train_loader, test_loader, loss_type="cross_entropy"):
    opt = AdamW(model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay)
    steps = max(1, len(train_loader) * Config.epochs)
    sched = get_linear_schedule_with_warmup(opt, int(steps * 0.1), steps)
    loss_fn = get_loss_fn(loss_type)
    for _ in range(Config.epochs):
        model.train()
        for batch in train_loader:
            ids = batch["input_ids"].to(Config.device)
            attn = batch["attention_mask"].to(Config.device)
            bbox = batch["bbox"].to(Config.device)
            labels = batch["labels"].to(Config.device)
            logits = model(input_ids=ids, attention_mask=attn, bbox=bbox).logits
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
    return evaluate(model, test_loader)


class SALLoRAModel(nn.Module):
    def __init__(self, num_labels: int, use_scalars: bool = True):
        super().__init__()
        self.backbone = LayoutLMv3Model.from_pretrained(Config.backbone_name)
        self.hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        self.backbone = get_peft_model(
            self.backbone,
            LoraConfig(
                r=Config.lora_r,
                lora_alpha=Config.lora_alpha,
                target_modules=["query", "value"],
                lora_dropout=Config.lora_dropout,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            ),
        )
        base_cfg = getattr(self.backbone, "config", None) or self.backbone.base_model.model.config
        self.num_layers = base_cfg.num_hidden_layers
        self.layer_scalars = nn.Parameter(torch.ones(self.num_layers) * 0.5) if use_scalars else None
        self.grad_masks: Dict[str, torch.Tensor] = {}

    def forward(self, input_ids, bbox, attention_mask=None):
        out = self.backbone(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hs = out.hidden_states
        if self.layer_scalars is not None:
            wh, sr = None, 0.0
            for i, h in enumerate(hs[1:]):
                r = torch.sigmoid(self.layer_scalars[i])
                wh = r * h if wh is None else wh + r * h
                sr = sr + r
            h = wh / (sr + 1e-8)
        else:
            h = hs[-1]
        return self.classifier(h)

    def set_grad_masks(self, masks):
        self.grad_masks = masks

    def apply_grad_masks(self, soft_factor=0.15):
        for name, param in self.backbone.named_parameters():
            if "lora" in name and param.grad is not None and name in self.grad_masks:
                m = self.grad_masks[name]
                param.grad = param.grad * m + param.grad * (1.0 - m) * soft_factor


def compute_mask(model: SALLoRAModel, clean_loader) -> Dict[str, torch.Tensor]:
    model.eval()
    model.zero_grad()
    for batch in clean_loader:
        ids = batch["input_ids"].to(Config.device)
        bbox = batch["bbox"].to(Config.device)
        attn = batch["attention_mask"].to(Config.device)
        labels = batch["labels"].to(Config.device)
        logits = model(ids, bbox, attn)
        loss = F.cross_entropy(logits.view(-1, Config.num_labels), labels.view(-1), ignore_index=-100)
        loss.backward()

    grads, names = [], []
    for name, param in model.backbone.named_parameters():
        if "lora" in name and param.grad is not None:
            grads.append(param.grad.flatten())
            names.append(name)
    if not grads:
        return {}
    flat = torch.cat(grads)
    k = max(1, int(flat.numel() * Config.mask_keep_ratio))
    if k < flat.numel():
        thr = torch.sort(flat.abs(), descending=True).values[k - 1]
        gmask = (flat.abs() >= thr).float()
    else:
        gmask = torch.ones_like(flat)
    masks, idx = {}, 0
    for name in names:
        g = model.backbone.get_parameter(name).grad
        n = g.numel()
        masks[name] = gmask[idx : idx + n].view_as(g).to(Config.device)
        idx += n
    print(f"  SALLoRA mask active: {gmask.sum().item():.0f}/{flat.numel()}")
    model.zero_grad()
    return masks


def train_sallora(model, train_loader, clean_ref_loader, test_loader, loss_type="cross_entropy"):
    for name, p in model.backbone.named_parameters():
        p.requires_grad = "lora" in name
    for p in model.classifier.parameters():
        p.requires_grad = True
    if model.layer_scalars is not None:
        model.layer_scalars.requires_grad = True

    print("  Computing SALLoRA masks...")
    model.set_grad_masks(compute_mask(model, clean_ref_loader))

    params = [p for p in model.parameters() if p.requires_grad]
    opt = AdamW(params, lr=Config.lr, weight_decay=Config.weight_decay)
    steps = max(1, len(train_loader) * Config.epochs)
    sched = get_linear_schedule_with_warmup(opt, int(steps * 0.1), steps)
    loss_fn = get_loss_fn(loss_type)

    for _ in range(Config.epochs):
        model.train()
        for batch in train_loader:
            ids = batch["input_ids"].to(Config.device)
            attn = batch["attention_mask"].to(Config.device)
            bbox = batch["bbox"].to(Config.device)
            labels = batch["labels"].to(Config.device)
            tq = batch["token_quality_mask"].to(Config.device)
            logits = model(ids, bbox, attn)
            loss = loss_fn(logits, labels, weights=tq)
            loss.backward()
            model.apply_grad_masks(Config.soft_mask_factor)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
    return evaluate(model, test_loader)


class RouterLoRAModel(nn.Module):
    """Token gate modulates LoRA residual: h = h_base + g * (h_lora - h_base)."""

    def __init__(self, num_labels: int):
        super().__init__()
        self.backbone = LayoutLMv3Model.from_pretrained(Config.backbone_name)
        self.hidden_size = self.backbone.config.hidden_size
        self.backbone = get_peft_model(
            self.backbone,
            LoraConfig(
                r=Config.lora_r,
                lora_alpha=Config.lora_alpha,
                target_modules=["query", "value"],
                lora_dropout=Config.lora_dropout,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            ),
        )
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        self.gate = nn.Sequential(
            nn.Linear(self.hidden_size + 4, Config.gate_hidden),
            nn.ReLU(),
            nn.Dropout(Config.gate_dropout),
            nn.Linear(Config.gate_hidden, 1),
        )

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


def train_router_lora(model, train_loader, test_loader, loss_type="cross_entropy"):
    for name, p in model.backbone.named_parameters():
        p.requires_grad = "lora" in name
    for p in model.classifier.parameters():
        p.requires_grad = True
    for p in model.gate.parameters():
        p.requires_grad = True

    params = [p for p in model.parameters() if p.requires_grad]
    opt = AdamW(params, lr=Config.lr, weight_decay=Config.weight_decay)
    steps = max(1, len(train_loader) * Config.epochs)
    sched = get_linear_schedule_with_warmup(opt, int(steps * 0.1), steps)
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
    return evaluate(model, test_loader)


def evaluate(model, test_loader) -> Tuple[float, float]:
    model.eval()
    preds, gold = [], []
    with torch.no_grad():
        for batch in test_loader:
            ids = batch["input_ids"].to(Config.device)
            attn = batch["attention_mask"].to(Config.device)
            bbox = batch["bbox"].to(Config.device)
            labels = batch["labels"].to(Config.device)
            if hasattr(model, "backbone"):
                logits = model(ids, bbox, attn)
            else:
                logits = model(input_ids=ids, attention_mask=attn, bbox=bbox).logits
            valid = labels != -100
            preds.extend(logits.argmax(-1)[valid].cpu().numpy())
            gold.extend(labels[valid].cpu().numpy())
    return (
        float(f1_score(gold, preds, average="weighted", zero_division=0)),
        float(f1_score(gold, preds, average="macro", zero_division=0)),
    )


def make_seeded_loader(dataset, seed, batch_size, shuffle=True):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=g)


def run_compare(config_override: Optional[dict] = None) -> dict:
    if config_override:
        for k, v in config_override.items():
            setattr(Config, k, v)

    print(f"Device: {Config.device}")
    print("=" * 60)
    print(
        f"COMPARE | {Config.dataset_name} | {Config.corruption_type} @ "
        f"{Config.corruption_rate * 100:.0f}% | {Config.loss_type}"
    )
    print(f"epochs={Config.epochs} seeds={Config.seeds}")
    print("Methods: Standard LoRA | SALLoRA | RouterLoRA")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(Config.backbone_name)

    # Use unified loader
    train_data, test_data, num_labels = load_and_prepare_dataset(
        Config.dataset_name, tokenizer, Config.max_seq_length
    )
    Config.num_labels = num_labels

    base_seed = 42
    clean_train = DocumentDatasetWithNoise(
        train_data, tokenizer, p=0.0, max_len=Config.max_seq_length,
        noise_type=Config.corruption_type, seed=base_seed, box_swap_quality=Config.box_swap_quality,
    )
    noisy_train = DocumentDatasetWithNoise(
        train_data, tokenizer, p=Config.corruption_rate, max_len=Config.max_seq_length,
        noise_type=Config.corruption_type, seed=base_seed, box_swap_quality=Config.box_swap_quality,
    )
    test_ds = DocumentDatasetWithNoise(
        test_data, tokenizer, p=0.0, max_len=Config.max_seq_length,
        noise_type=Config.corruption_type, seed=base_seed, box_swap_quality=Config.box_swap_quality,
    )

    rng = np.random.RandomState(12345)
    n = len(clean_train)
    ref_idx = rng.choice(n, size=min(Config.clean_ref_size, n), replace=False).tolist()
    clean_ref_loader = DataLoader(
        Subset(clean_train, ref_idx), batch_size=min(4, max(1, len(ref_idx))), shuffle=False
    )

    per_seed = []
    buckets = {"lora": [], "sallora": [], "router": []}

    for seed in list(Config.seeds):
        set_seed(seed)
        print(f"\n--- Seed {seed} ---")
        train_loader = make_seeded_loader(noisy_train, seed, Config.batch_size, True)
        test_loader = make_seeded_loader(test_ds, seed, Config.batch_size, False)

        print("  Training Standard LoRA...")
        m = build_standard_lora(num_labels)
        lw, lm = train_standard_lora(m, train_loader, test_loader, Config.loss_type)
        del m
        clear_cuda()

        print("  Training SALLoRA...")
        m = SALLoRAModel(num_labels, use_scalars=Config.use_scalars).to(Config.device)
        sw, sm = train_sallora(m, train_loader, clean_ref_loader, test_loader, Config.loss_type)
        del m
        clear_cuda()

        print("  Training RouterLoRA...")
        m = RouterLoRAModel(num_labels).to(Config.device)
        rw, rm = train_router_lora(m, train_loader, test_loader, Config.loss_type)
        del m
        clear_cuda()

        print(f"  LoRA        W={lw:.4f} M={lm:.4f}")
        print(f"  SALLoRA     W={sw:.4f} M={sm:.4f}")
        print(f"  RouterLoRA  W={rw:.4f} M={rm:.4f}")

        buckets["lora"].append((lw, lm))
        buckets["sallora"].append((sw, sm))
        buckets["router"].append((rw, rm))
        per_seed.append(
            {
                "seed": seed,
                "lora_weighted": lw,
                "lora_macro": lm,
                "sallora_weighted": sw,
                "sallora_macro": sm,
                "router_weighted": rw,
                "router_macro": rm,
            }
        )

    def stats(pairs):
        w = [p[0] for p in pairs]
        m = [p[1] for p in pairs]
        return float(np.mean(w)), float(np.std(w)), float(np.mean(m)), float(np.std(m))

    lw_m, lw_s, lm_m, lm_s = stats(buckets["lora"])
    sw_m, sw_s, sm_m, sm_s = stats(buckets["sallora"])
    rw_m, rw_s, rm_m, rm_s = stats(buckets["router"])

    print("\n" + "=" * 60)
    print("SUMMARY (mean ± std)")
    print("=" * 60)
    print(f"Standard LoRA: W={lw_m:.4f}±{lw_s:.4f}  M={lm_m:.4f}±{lm_s:.4f}")
    print(f"SALLoRA:       W={sw_m:.4f}±{sw_s:.4f}  M={sm_m:.4f}±{sm_s:.4f}")
    print(f"RouterLoRA:    W={rw_m:.4f}±{rw_s:.4f}  M={rm_m:.4f}±{rm_s:.4f}")
    print(f"Router - LoRA (W): {rw_m - lw_m:+.4f}")
    print(f"Router - LoRA (M): {rm_m - lm_m:+.4f}")
    print("=" * 60)

    summary = {
        "dataset": Config.dataset_name,
        "corruption_type": Config.corruption_type,
        "corruption_rate": Config.corruption_rate,
        "loss_type": Config.loss_type,
        "epochs": Config.epochs,
        "seeds": list(Config.seeds),
        "lora_weighted_mean": lw_m,
        "lora_weighted_std": lw_s,
        "lora_macro_mean": lm_m,
        "lora_macro_std": lm_s,
        "sallora_weighted_mean": sw_m,
        "sallora_weighted_std": sw_s,
        "sallora_macro_mean": sm_m,
        "sallora_macro_std": sm_s,
        "router_weighted_mean": rw_m,
        "router_weighted_std": rw_s,
        "router_macro_mean": rm_m,
        "router_macro_std": rm_s,
        "router_minus_lora_weighted": rw_m - lw_m,
        "router_minus_lora_macro": rm_m - lm_m,
    }
    return {"summary": summary, "per_seed": per_seed}


def run_decision_notice():
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "emails",
        "01_decision_email_to_prof_zhang.txt",
    )
    print("Advisor decision email drafted at:")
    print(f"  {path}")
    print("Send that email. Default path in plan: Option 2 (RouterLoRA) unless he prefers Option 1.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["decision", "compare"], default="compare")
    p.add_argument("--dataset", type=str, default="funsd", choices=["funsd", "cord", "sroie"],
                   help="Dataset to use")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--corruption-rate", type=float, default=None)
    p.add_argument("--out", type=str, default="router_compare_results.csv")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "decision":
        run_decision_notice()
    else:
        override = {
            "dataset_name": args.dataset,
            "corruption_type": "substitution",
            "corruption_rate": 0.7 if args.corruption_rate is None else args.corruption_rate,
            "loss_type": "cross_entropy",
            "epochs": 15 if args.epochs is None else args.epochs,
            "seeds": [42, 123, 2024, 456, 789] if args.seeds is None else args.seeds,
        }
        out = run_compare(override)
        per_path = args.out.replace(".csv", "_per_seed.csv")
        json_path = args.out.replace(".csv", "_summary.json")
        pd.DataFrame(out["per_seed"]).to_csv(per_path, index=False)
        pd.DataFrame([out["summary"]]).to_csv(args.out, index=False)
        with open(json_path, "w") as f:
            json.dump(out["summary"], f, indent=2)
        print(f"\nWrote {args.out}, {per_path}, {json_path}")