"""GateConcat: diagnostic study of noise-aware PEFT for document NER.

Light imports only — importing this package does not require torch, so the
pure pieces (config, corruption, significance, JSON utils) stay testable
anywhere. Heavy pieces live in submodules:

    gateconcat.data      — FUNSD loading + noisy dataset (torch, datasets)
    gateconcat.models    — LoRA shell + GateConcat family (torch, peft)
    gateconcat.training  — train / evaluate loop (torch, sklearn)
"""
from __future__ import annotations

from .config import Config
from .corruption import OCRNoiseTransform, ocr_confusion_sub, swap_two_boxes
from .json_utils import dump_json, sanitize_for_json
from .seed import set_seed
from .significance import run_tests

__version__ = "0.2.0"

__all__ = [
    "Config",
    "OCRNoiseTransform",
    "ocr_confusion_sub",
    "swap_two_boxes",
    "dump_json",
    "sanitize_for_json",
    "set_seed",
    "run_tests",
    "__version__",
]
