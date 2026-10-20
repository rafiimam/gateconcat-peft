"""Locked experiment configuration — single source of truth.

These values ARE the protocol: every experiment script imports Config
from here, so no method can quietly drift to different hyper-parameters.
Values are verbatim from the original gatelora_v4.py / gatelora_v3.py."""
from __future__ import annotations

from typing import List


def _default_device():
    """torch.device if torch is installed, else the string 'cpu' (light imports)."""
    try:
        import torch

        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    except ImportError:
        return 'cpu'


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

    # v3 method constants (capacity / aux controls in gatelora_v3.py)
    lambda_gate = 0.1
    soft_corrupt_target = 0.3
    aux_warmup_epochs = 3
    frozen_gate_value = 0.5

    device = _default_device()

    dataset_name = "funsd"
    corruption_type = "substitution"
    corruption_rate = 0.7
    loss_type = "cross_entropy"

    corrupted_token_weight = 0.1
    clean_token_weight = 1.0
