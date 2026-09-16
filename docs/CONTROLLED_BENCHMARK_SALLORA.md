# SALLoRA controlled benchmark — closed negative result

**Status:** Do not spend more GPU on SALLoRA variants. Use only as failed baseline / motivation.

## Protocol (matched)

- Model: LayoutLMv3-base + LoRA (r=16, alpha=32, query/value)
- Dataset: FUNSD
- Seeds: 42, 123, 2024, 456, 789
- Unified LR 2e-4, identical epochs, no special warmup privilege
- Losses: CE, label smoothing, focal
- Corruptions: substitution, OCR confusion, box swap

## Summary (controlled matched table)

| Setting | LoRA W | SALLoRA W | Gain W | LoRA M | SALLoRA M | Gap M |
|--------|--------|-----------|--------|--------|-----------|-------|
| sub + CE | 0.7799 | 0.7772 | -0.0027 | 0.7233 | 0.7016 | -0.0217 |
| sub + LS | 0.7826 | 0.7814 | -0.0012 | 0.7283 | 0.7035 | -0.0248 |
| sub + Focal | 0.7719 | 0.7782 | +0.0063 | 0.7180 | 0.7256 | +0.0076 |
| OCR + CE | 0.8009 | 0.7904 | -0.0104 | 0.7414 | 0.7105 | -0.0309 |
| box + CE | 0.8135 | 0.8138 | +0.0003 | 0.7505 | 0.7473 | -0.0031 |

**Conclusion:** No reliable weighted-F1 improvement; macro-F1 often worse. SALLoRA is dead as a method.

## Not for reporting

`archive/DO_NOT_REPORT_*` — undertrained 1-seed / 3-epoch smoke test (F1≈0.40). Ignore completely.
