"""Models — canonical v4 definitions, verbatim from gatelora_v4.py.

GateConcat (v3 method) is defined here in its v4 form: forward() adds an
optional return_debug flag and is otherwise call-compatible with the v3
script (return_gates behaviour is identical). v3-only variants
(ScalarConcat, GateConcatAux) stay in gatelora_v3.py."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import LayoutLMv3Model

from .config import Config


def _lora_cfg():
    return LoraConfig(
        r=Config.lora_r, lora_alpha=Config.lora_alpha,
        target_modules=["query", "value"], lora_dropout=Config.lora_dropout,
        bias="none", task_type=TaskType.FEATURE_EXTRACTION,
    )


class _SharedLoRAShell(nn.Module):
    def __init__(self, num_labels: int, classifier_in: Optional[int] = None):
        super().__init__()
        self.backbone = LayoutLMv3Model.from_pretrained(Config.backbone_name)
        self.hidden_size = self.backbone.config.hidden_size
        self.backbone = get_peft_model(self.backbone, _lora_cfg())
        cin = classifier_in if classifier_in is not None else self.hidden_size
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
        return torch.cat([h_base, delta, delta_norm, bbox_feat], dim=-1), delta, delta_norm


class StandardLoRAModel(_SharedLoRAShell):
    def forward(self, input_ids, bbox, attention_mask=None, return_gates=False, return_debug=False):
        h = self._encode(input_ids, bbox, attention_mask, True)
        logits = self.classifier(h)
        if return_debug:
            return logits, None, None
        if return_gates:
            return logits, None
        return logits


class GateConcatModel(_SharedLoRAShell):
    """v3 GateConcat: concat(h_lora, g). Unchanged."""

    def __init__(self, num_labels: int):
        super().__init__(num_labels)
        self.classifier = nn.Linear(self.hidden_size + 1, num_labels)
        gate_in = 2 * self.hidden_size + 1 + 4
        self.gate = nn.Sequential(
            nn.Linear(gate_in, Config.gate_hidden), nn.ReLU(),
            nn.Dropout(Config.gate_dropout), nn.Linear(Config.gate_hidden, 1),
        )

    def _gate_value(self, h_lora, h_base, bbox):
        feats, delta, delta_norm = self._rich_gate_features(h_lora, h_base, bbox)
        return torch.sigmoid(self.gate(feats)), delta_norm

    def forward(self, input_ids, bbox, attention_mask=None, return_gates=False, return_debug=False):
        h_lora = self._encode(input_ids, bbox, attention_mask, True)
        h_base = self._encode(input_ids, bbox, attention_mask, False)
        g, delta_norm = self._gate_value(h_lora, h_base, bbox)
        logits = self.classifier(torch.cat([h_lora, g], dim=-1))
        if return_debug:
            return logits, g.squeeze(-1), delta_norm.squeeze(-1)
        if return_gates:
            return logits, g.squeeze(-1)
        return logits


class GateConcatFrozenModel(GateConcatModel):
    """H+1 capacity control: g ≡ 0.5; gate MLP unused (same as v3)."""

    gate_auc_applicable = False
    frozen_gate_value = 0.5

    def __init__(self, num_labels: int):
        super().__init__(num_labels)
        for p in self.gate.parameters():
            p.requires_grad = False

    def _gate_value(self, h_lora, h_base, bbox):
        _, _, delta_norm = self._rich_gate_features(h_lora, h_base, bbox)
        g = torch.full(
            (h_lora.size(0), h_lora.size(1), 1),
            self.frozen_gate_value,
            device=h_lora.device,
            dtype=h_lora.dtype,
        )
        return g, delta_norm


class GateConcatPlusModel(_SharedLoRAShell):
    """Fair upgrade: concat(h_lora, g, ||delta||). Same Config; no aux; no new HP.

    Motivation: under higher corruption, LoRA–base disagreement magnitude may be a
    useful token feature for NER; expose it explicitly to the classifier (already
    used inside the gate MLP in v3).
    """

    def __init__(self, num_labels: int):
        super().__init__(num_labels)
        self.classifier = nn.Linear(self.hidden_size + 2, num_labels)
        gate_in = 2 * self.hidden_size + 1 + 4
        self.gate = nn.Sequential(
            nn.Linear(gate_in, Config.gate_hidden), nn.ReLU(),
            nn.Dropout(Config.gate_dropout), nn.Linear(Config.gate_hidden, 1),
        )

    def forward(self, input_ids, bbox, attention_mask=None, return_gates=False, return_debug=False):
        h_lora = self._encode(input_ids, bbox, attention_mask, True)
        h_base = self._encode(input_ids, bbox, attention_mask, False)
        feats, delta, delta_norm = self._rich_gate_features(h_lora, h_base, bbox)
        g = torch.sigmoid(self.gate(feats))
        logits = self.classifier(torch.cat([h_lora, g, delta_norm], dim=-1))
        if return_debug:
            return logits, g.squeeze(-1), delta_norm.squeeze(-1)
        if return_gates:
            return logits, g.squeeze(-1)
        return logits


class GateConcatPlusConstModel(GateConcatPlusModel):
    """Capacity control for Plus: same H+2 classifier, third feature ≡ 0.5 (not ||Δ||).

    If Plus ≈ Const, the gain is width, not disagreement magnitude.
    """

    const_feature_value = 0.5

    def forward(self, input_ids, bbox, attention_mask=None, return_gates=False, return_debug=False):
        h_lora = self._encode(input_ids, bbox, attention_mask, True)
        h_base = self._encode(input_ids, bbox, attention_mask, False)
        feats, delta, delta_norm = self._rich_gate_features(h_lora, h_base, bbox)
        g = torch.sigmoid(self.gate(feats))
        const = torch.full_like(delta_norm, self.const_feature_value)
        logits = self.classifier(torch.cat([h_lora, g, const], dim=-1))
        if return_debug:
            return logits, g.squeeze(-1), delta_norm.squeeze(-1)
        if return_gates:
            return logits, g.squeeze(-1)
        return logits
