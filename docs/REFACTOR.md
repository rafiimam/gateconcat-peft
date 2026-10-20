# Refactor notes — 2026-10-20

The original release was five monoliths (`sallora.py`, `gatelora.py`,
`gatelora_v2.py`, `gatelora_v3.py`, `gatelora_v4.py`, 684–938 lines each)
that duplicated the locked Config, corruption transforms, dataset, model
shell, and training loop with 30–59% line overlap. This refactor extracts
the shared code into the `gateconcat` package **without changing the
protocol or any published result**.

## What moved where

| Original (in every monolith) | Now |
|---|---|
| `Config` (locked protocol) | `gateconcat/config.py` — single source of truth, incl. v3 constants |
| `set_seed` / `clear_cuda` | `gateconcat/seed.py` |
| `OCRNoiseTransform`, `ocr_confusion_sub`, `swap_two_boxes` | `gateconcat/corruption.py` (torch-free, unit-tested) |
| `load_and_prepare_dataset`, `DocumentDatasetWithNoise` | `gateconcat/data.py` |
| `ce_mean`, `quality_to_clean_target` | `gateconcat/losses.py` |
| LoRA shell + GateConcat family (v4 forms) | `gateconcat/models.py` |
| Train / evaluate loop (v4) | `gateconcat/training.py` |
| Paired tests from `analyze_significance.py` | `gateconcat/significance.py` |
| — | `gateconcat/json_utils.py` — strict-JSON writer (NaN → null) |

Experiment scripts shrank: `gatelora_v4.py` 684 → ~260 lines,
`gatelora_v3.py` 708 → ~370 lines. They now contain only their method
registry, wiring, `run_compare`, and CLI — plus, for v3, the code whose
semantics genuinely differ from v4 (see below).

## Deliberate exceptions (behaviour preservation)

- **v3 keeps its own** `soft_gate_aux_loss`, `ScalarConcatModel`,
  `GateConcatAuxModel`, `_trainable_setup` / `train_task_only`,
  `train_concat_aux`, and `evaluate_gate_corruption_auc`. The v4 AUC
  helper returns `None` for models without a gate MLP; v3 scores
  ScalarConcat's scalar as a gate. Reusing v4's version would have
  changed v3 outputs, so v3's is kept verbatim.
- **Legacy snapshots are frozen.** `sallora.py`, `gatelora.py` (v1) and
  `gatelora_v2.py` moved to `legacy/` verbatim (local interpreter paths
  in docstrings aside). They produced the Phase A/B results; editing
  them would break provenance. See `legacy/README.md`.
- **Corruption RNG is unchanged**: seeded per `(seed, sample_idx)`, so a
  sample's corruption is identical across epochs — that was the published
  protocol. It is documented in `gateconcat/corruption.py`, not silently
  altered.

## Fixes included

- `DocumentDatasetWithNoise` tokenized twice per sample; it now tokenizes
  once (`return_tensors="pt"` + offsets in a single call). A regression
  test (`tests/test_data.py`) counts tokenizer calls.
- Result summaries were written with bare `NaN` (invalid strict JSON).
  Writers now use `gateconcat.json_utils.dump_json` (NaN/±Inf → null),
  and the 9 affected published files under `results/` were sanitized in
  place — all measured values unchanged, only `NaN` became `null`.
- Hardcoded local interpreter paths (`/home/eai/...`) removed from
  docstrings.
- Token-quality weights are explicit `OCRNoiseTransform` parameters;
  `DocumentDatasetWithNoise` passes the `Config` values through, so
  overriding `Config` behaves exactly as before.

## Verification

- `tests/` (unittest, pytest-compatible): corruption semantics,
  significance helpers, JSON utils, locked-config guard, dataset with a
  fake tokenizer. The light tests need only numpy/pandas/scipy.
- Corruption parity: the package transform was checked against the
  original v4 code on 216 (noise type × rate × seed × sample) cases —
  outputs identical.
- Heavy modules (torch/peft/transformers) are byte-compiled and
  import-checked; full training runs were **not** re-executed for this
  refactor. Re-running the README reproduction commands is the final
  end-to-end check.

## Packaging

`pyproject.toml` (setuptools, `pip install -e .`), GitHub Actions CI in
`.github/workflows/ci.yml` (CPU torch, byte-compile, pytest).
`requirements.txt` is unchanged.
