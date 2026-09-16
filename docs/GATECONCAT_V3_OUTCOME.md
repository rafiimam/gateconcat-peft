# GateLoRA v3 — controlled outcome (FUNSD, substitution @ 70%)

**Status:** Closed for this protocol. Do not retune LR / epochs / LoRA rank / λ for this table.  
**Code:** `gatelora_v3.py`  
**Artifacts:** `gatelora_v3_funsd07.csv`, `_per_seed.csv`, `_summary.json`

## Protocol (matched)

| Setting | Value |
|--------|--------|
| Backbone | LayoutLMv3-base + shared shell (`LayoutLMv3Model` + LoRA + Linear) |
| LoRA | r=16, α=32, query/value, dropout=0.1 |
| Optim | AdamW lr=2e-4, WD=0.01, 15 epochs, batch=4, warmup 10% |
| Seeds | 42, 123, 2024, 456, 789 |
| Train noise | substitution @ 70% |
| Eval | clean test + noisy test @ same rate |
| Task loss | unweighted CE for all methods |

Methods: StandardLoRA | GateConcat | GateConcatFrozen (g≡0.5) | ScalarConcat | GateConcatAux (soft aux λ=0.1, warmup 3).

## Summary (mean ± std)

| Method | Clean macro | Noisy macro | Noisy weighted | Gate AUC |
|--------|-------------|-------------|----------------|----------|
| StandardLoRA | 0.7203 ± 0.0089 | 0.7312 ± 0.0104 | 0.7887 | — |
| **GateConcat** | **0.7226 ± 0.0149** | **0.7463 ± 0.0130** | **0.8016** | **0.521** |
| GateConcatFrozen | 0.7221 ± 0.0099 | 0.7336 ± 0.0203 | 0.7872 | — (N/A) |
| ScalarConcat | 0.7030 ± 0.0224 | 0.7246 ± 0.0220 | 0.7919 | 0.509 |
| GateConcatAux | 0.7106 ± 0.0124 | 0.7318 ± 0.0072 | 0.7869 | **0.909** |

Deltas (noisy macro): GateConcat − LoRA = **+0.0151**; GateConcat − Frozen = +0.0127 (mean only).

## Paired significance (n=5 seeds)

| Comparison | Metric | Δ | paired *t* p | Wilcoxon p |
|------------|--------|---|--------------|------------|
| GateConcat vs LoRA | noisy macro | +0.0151 | **0.003*** | 0.063† |
| GateConcat vs LoRA | noisy weighted | +0.0129 | **0.028*** | 0.063† |
| GateConcat vs LoRA | clean macro | +0.0023 | 0.817 | 1.000 |
| GateConcat vs Frozen | noisy macro | +0.0127 | 0.336 | 0.625 |
| Frozen vs LoRA | noisy macro | +0.0024 | 0.843 | 0.625 |
| GateConcatAux vs LoRA | noisy macro | +0.0006 | 0.925 | 0.813 |

† With n=5 and all diffs positive, two-sided Wilcoxon cannot go below 0.0625.

**Fair claims:**
1. GateConcat improves **noisy** F1 over LoRA under this protocol (*t*-test significant for macro + weighted).
2. Clean F1 is essentially matched (not a clean-data sacrifice, also not a clean win).
3. Constant extra dim (Frozen) ≈ LoRA — does **not** reproduce the LoRA gap.
4. Tiny scalar head (ScalarConcat) fails — not “any extra scalar.”
5. Soft aux (GateConcatAux) achieves high corruption AUC (~0.91) but **does not** improve NER vs LoRA.

**Overclaims to avoid:**
- “GateConcat significantly beats Frozen” — mean gap exists; **not** significant at n=5.
- “Noise-aware / OCR-aware gating” — GateConcat AUC ≈ **0.52** (chance). The gate is **not** tracking synthetic corruption labels.

## Prior context (v2 rate sweep, GateConcat vs LoRA only)

Noisy macro Δ (GateConcat − LoRA) rose with corruption rate: ~0 @ 0%, ~0 @ 30%, +0.6pp @ 50%, **+1.5pp @ 70%**. That trend motivated keeping GateConcat; v3 capacity controls were required before claiming a method.

## Scientific conclusion

GateConcat is a **modest positive** under high synthetic substitution noise: richer token feature `concat(h_lora, g)` helps vs matched LoRA. Mechanism is **not** synthetic-corruption detection. Supervised corruption detection (Aux) is the wrong objective for this task under clean NER labels.

**Venue realism (honest):** not enough alone for NeurIPS/ICML/ICLR/ACL main. Possible Findings/workshop **if** gain transfers to other corruption types and ≥1 more dataset, plus a non-hand-wavy mechanism story.

## Next (v4)

See `gatelora_v4.py`: OCR-confusion + box-swap (LoRA vs GateConcat / GateConcat+), same locked Config; gate analysis without changing training hyperparameters.
