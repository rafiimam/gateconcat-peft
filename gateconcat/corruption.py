"""Synthetic document corruption — pure numpy, no torch (unit-testable).

Logic verbatim from gatelora_v4.py. Note (preserved on purpose): the RNG
is seeded per (seed, sample_idx), so a sample's corruption is identical
across epochs. That was the published protocol; changing it would change
the results, so it is documented here rather than silently 'fixed'."""
from __future__ import annotations

import numpy as np


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
    def __init__(self, p=0.5, noise_type="substitution", seed=42, box_swap_quality=0.5,
                 clean_weight=1.0, corrupted_weight=0.1):
        self.p = p
        self.noise_type = noise_type
        self.seed = seed
        self.box_swap_quality = box_swap_quality
        # Token-quality weights. The original code read these from the global
        # Config at call time; they are now explicit parameters whose defaults
        # equal the Config clean/corrupted token weights, and the dataset
        # class passes the Config values through — so an experiment that
        # overrides the Config weights behaves exactly as before.
        self.clean_weight = clean_weight
        self.corrupted_weight = corrupted_weight

    def __call__(self, words, boxes, sample_idx=0):
        rng = np.random.RandomState(self.seed + sample_idx)
        words = list(words)
        boxes = [list(b) for b in boxes]
        if self.p == 0.0 or self.noise_type == "none":
            return words, [self.clean_weight] * len(words), boxes
        if self.noise_type == "box_swap":
            if rng.random() < self.p:
                boxes = swap_two_boxes(boxes, rng)
                q = [self.box_swap_quality] * len(words)
            else:
                q = [self.clean_weight] * len(words)
            return words, q, boxes
        out, q = [], []
        for w in words:
            if len(w) and rng.random() < self.p:
                if self.noise_type == "substitution":
                    pos = rng.randint(0, len(w))
                    w = w[:pos] + chr(ord("a") + rng.randint(0, 26)) + w[pos + 1 :]
                elif self.noise_type == "ocr_confusion":
                    w = ocr_confusion_sub(w)
                elif self.noise_type == "insertion":
                    pos = rng.randint(0, len(w) + 1)
                    w = w[:pos] + chr(ord("a") + rng.randint(0, 26)) + w[pos:]
                elif self.noise_type == "deletion":
                    if len(w) == 1:
                        w = ""
                    else:
                        pos = rng.randint(0, len(w))
                        w = w[:pos] + w[pos + 1 :]
                else:
                    raise ValueError(f"Unknown text noise_type: {self.noise_type}")
                q.append(self.corrupted_weight)
            else:
                q.append(self.clean_weight)
            out.append(w)
        return out, q, boxes
