"""Strict-JSON writing: NaN / Infinity become null instead of bare NaN tokens.

The original experiment scripts used json.dump directly, which emits `NaN` —
accepted by Python but invalid strict JSON, so 9 of the published summary
files in results/ fail strict parsers. dump_json sanitizes first.
"""
from __future__ import annotations

import json
import math
from typing import Any


def sanitize_for_json(obj: Any) -> Any:
    """Recursively replace non-finite floats with None (-> JSON null)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    return obj


def dump_json(obj: Any, fp, **kwargs) -> None:
    """json.dump with sanitized data and allow_nan=False as a hard guard."""
    kwargs.setdefault("indent", 2)
    json.dump(sanitize_for_json(obj), fp, allow_nan=False, **kwargs)
