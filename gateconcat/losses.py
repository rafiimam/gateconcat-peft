"""Loss helpers (verbatim from gatelora_v4.py)."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from .config import Config


def ce_mean(logits, labels, ignore_index=-100):
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)), labels.view(-1),
        ignore_index=ignore_index, reduction="none",
    ).view_as(labels)
    mask = (labels != ignore_index).float()
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


def quality_to_clean_target(token_quality: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    valid = token_quality > 0
    clean = (token_quality >= Config.clean_token_weight - 1e-6).float()
    return clean, valid.float()
