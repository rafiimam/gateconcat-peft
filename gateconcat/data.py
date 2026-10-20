"""Dataset loading and the noisy-token dataset (from gatelora_v4.py)."""
from __future__ import annotations

import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from .config import Config
from .corruption import OCRNoiseTransform


def load_and_prepare_dataset(dataset_name: str, tokenizer, max_len: int):
    if dataset_name == "funsd":
        ds = load_dataset("nielsr/funsd-layoutlmv3")
        return ds["train"], ds["test"], 7
    if dataset_name in ("cord", "sroie"):
        raise NotImplementedError(
            f"{dataset_name}: label mapping not fixed yet. "
            "String/category tags would be mapped to -100 and training would be invalid. "
            "Use --dataset funsd until a proper label mapper is added."
        )
    raise ValueError(f"Unknown dataset: {dataset_name}")


class DocumentDatasetWithNoise(Dataset):
    def __init__(
        self, hf_dataset, tokenizer, p=0.0, max_len=512, noise_type="substitution", seed=42, box_swap_quality=0.5
    ):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.noise = OCRNoiseTransform(
            p, noise_type, seed, box_swap_quality,
            clean_weight=Config.clean_token_weight,
            corrupted_weight=Config.corrupted_token_weight,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        words, boxes, labels = ex["tokens"], ex["bboxes"], ex["ner_tags"]
        labels = [l if isinstance(l, int) and 0 <= l < Config.num_labels else -100 for l in labels]
        words, word_q, boxes = self.noise(words, boxes, sample_idx=idx)

        # Single tokenization (the original called the tokenizer twice per
        # sample — once for word_ids, once for tensors — with identical inputs).
        enc = self.tokenizer(
            words, boxes=boxes, word_labels=labels,
            truncation=True, padding="max_length", max_length=self.max_len,
            return_offsets_mapping=True, return_tensors="pt",
        )
        word_ids = enc.word_ids()
        if word_ids is None:
            raise RuntimeError("tokenizer returned no word_ids")

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
