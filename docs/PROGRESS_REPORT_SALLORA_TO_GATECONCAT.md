# Progress Report: From SALLoRA Failure to GateConcat

**Project:** Noise-aware / robust PEFT for document NER (LayoutLMv3 + LoRA)  
**Dataset (primary):** FUNSD  
**Protocol:** Matched comparisons — same backbone, LoRA (r=16, α=32), LR 2e-4, WD 0.01, 15 epochs, batch 4, seeds `[42, 123, 2024, 456, 789]`, unweighted CE unless noted. No per-method hyperparameter privileges.  
**Status date:** August 30, 2026 (weekend experiments complete)  
**Audience:** Advisor progress update / internal research log

---

## 1. One-paragraph executive summary

We started from a **hand-crafted noise-aware PEFT recipe (SALLoRA)** that **did not beat standard LoRA** under a fair protocol. Early routing (RouterLoRA) and residual gating with supervised corruption losses also failed or hurt task F1. After architecture-matched redesigns and capacity controls, a simple method — **GateConcat**: `classify(concat(h_LoRA, g))` with a rich token gate and **no** corruption auxiliary loss — shows a **modest but statistically significant gain over LoRA** under **high-severity synthetic text corruption** (substitution @70%: **+1.5 pp** noisy macro, *p*=0.003; OCR @70%: **+2.3 pp**, *p*=0.002; deletion @70%: **+2.0 pp**, *p*=0.073). The gate does **not** detect synthetic corruption (AUC ≈ 0.52). Gains **do not transfer** to layout noise (box-swap @50%: **−0.2 pp**). **Weekend capacity controls:** a constant-gate Frozen baseline also beats LoRA on OCR (+0.7 pp), and GateConcat beats Frozen directionally (+1.5 pp) but **not significantly** (*p*=0.21 OCR, *p*=0.34 sub@70%). **Verdict:** the effect is **real but narrow** — text-only, not noise-aware, partly explainable by H+1 capacity; learned gating may add a small extra benefit at high severity but is **not established** with n=5. **Recommendation:** write a **diagnostic / Findings / workshop** paper; stop architecture variants.

---

## 2. Locked experimental protocol (what “fair” means)

| Setting | Value |
|--------|--------|
| Backbone | `microsoft/layoutlmv3-base` |
| PEFT | LoRA on `query`/`value`, r=16, α=32, dropout=0.1 |
| Shell (from GateLoRA onward) | Shared `LayoutLMv3Model` + Linear head for all methods |
| Optimizer | AdamW, lr=`2e-4`, weight decay=`0.01`, linear warmup 10% |
| Epochs / batch | 15 / 4 |
| Seeds | 42, 123, 2024, 456, 789 |
| Train | Noisy documents at stated corruption rate |
| Eval | **Clean** test **and** **noisy** test (same rate), weighted + macro F1 |
| Task loss | Unweighted CE (no quality reweighting on the task loss for GateConcat line) |

Corruptions studied: **substitution**, **OCR confusion**, **box swap** (synthetic).

Artifacts live in this repository (`legacy/sallora.py`, `legacy/gatelora.py`, `legacy/gatelora_v2.py`, `gatelora_v3/v4.py`, `results/`). Phase A/B scripts moved under `legacy/` in the 2026-10-20 package refactor (see `docs/REFACTOR.md`); v3/v4 now import the shared `gateconcat/` package.

---

## 3. Timeline of what we did

### Phase A — SALLoRA (closed **negative**)

**Idea:** Hand-crafted quality weighting + gradient masking + layer scalars for noisy document PEFT.

**Controlled FUNSD table (matched LoRA vs SALLoRA):**

| Setting | LoRA macro | SALLoRA macro | Gap |
|--------|------------|---------------|-----|
| substitution + CE | 0.723 | 0.702 | **−0.022** |
| substitution + LS | 0.728 | 0.704 | **−0.025** |
| substitution + Focal | 0.718 | 0.726 | +0.008 |
| OCR + CE | 0.741 | 0.711 | **−0.031** |
| box + CE | 0.751 | 0.747 | −0.003 |

**Finding:** No reliable gain; macro often worse. **SALLoRA is dead as a method** — keep only as failed baseline / motivation.  
**Doc:** `CONTROLLED_BENCHMARK_SALLORA.md`

Also: SROIE was near ceiling (all methods ~0.99 weighted F1) — useless for ranking. CORD blocked earlier by disk / label mapping (still not safe to run).

---

### Phase B — RouterLoRA / residual gating (mostly negative)

**Idea:** Learned gate modulates LoRA residual: `h = h_base + g · (h_LoRA − h_base)`.

| Variant | Result (high level) |
|--------|----------------------|
| RouterLoRA (gate on `h_LoRA`+bbox, no aux) | FUNSD sub@70%: **below or ≈ LoRA** (e.g. router−LoRA macro ≈ −0.013 in early compare) |
| GateLoRA v1 (rich features + hard aux, corrupt→0, λ=0.5) | High gate–corruption AUC, **worse** clean/noisy macro than LoRA |
| Soft residual / reweight variants (v2) | Soft aux still hurt; **GateReweight collapsed**; **GateConcat emerged** as only positive |

**Key scientific lesson (repeated across runs):**  
**Detecting synthetic corruption ≠ better NER** when entity labels remain clean. Supervised gates learn noise (AUC ~0.9) but task F1 does not improve.

---

### Phase C — GateConcat discovery (v2 → v3)

**Method (current champion):**

```text
h_LoRA, h_base = encode(adapters on / off)
Δ = h_LoRA − h_base
g = σ(MLP([h_base, Δ, ‖Δ‖, bbox]))
logits = Classifier( concat(h_LoRA, g) )     # H+1 features
# Train: unweighted CE only — NO corruption aux
```

#### C.1 Substitution rate trend (GateConcat vs LoRA)

| Corruption rate | LoRA noisy macro | GateConcat noisy macro | Δ |
|-----------------|------------------|------------------------|---|
| 0% | 0.751 | 0.751 | ~0 |
| 30% | 0.746 | 0.745 | ~0 |
| 50% | 0.730 | 0.736 | **+0.6 pp** |
| **70%** | **0.731** | **0.746** | **+1.5 pp** |

Relative gain appears mainly under **severe text noise**.

#### C.2 Capacity & aux controls @ substitution 70% (v3)

| Method | Noisy macro | Clean macro | Gate AUC | vs LoRA (noisy macro) |
|--------|-------------|-------------|----------|------------------------|
| StandardLoRA | 0.731 | 0.720 | — | — |
| **GateConcat** | **0.746** | 0.723 | **0.52** | **+0.015** (*t* p≈0.003) |
| GateConcatFrozen (g≡0.5) | 0.734 | 0.722 | N/A | +0.002 (n.s.) |
| ScalarConcat | 0.725 | 0.703 | 0.51 | −0.007 |
| GateConcatAux (soft aux) | 0.732 | 0.711 | **0.91** | ~0 |

**How to read this:**
- GateConcat beats LoRA on **noisy** F1 with all 5 seeds positive (*t*-test significant).
- Frozen ≈ LoRA → a **constant** extra dim does **not** explain the LoRA gap (mean Gate vs Frozen +0.013 is **not** significant at n=5 — do not overclaim).
- ScalarConcat fails → not “any extra scalar.”
- Aux learns corruption but **does not** help NER.

**Doc:** `GATECONCAT_V3_OUTCOME.md`

---

### Phase D — Transfer tests (v4)

| Corruption | GateConcat − LoRA (noisy macro) | Clean | Gate AUC | Verdict |
|------------|----------------------------------|-------|----------|---------|
| **OCR confusion @ 70%** | **+0.023** (all seeds +; *t* p≈0.002) | +0.014 | ~0.53 | **Transfers; larger than sub** |
| **Box-swap @ 50%** | **−0.002** | ~0 | ~0.50 | **Does not transfer** |

**GateConcatPlus** (`concat(h_LoRA, g, ‖Δ‖)`): did **not** beat GateConcat on OCR noisy macro; PlusConst almost as good as Plus → **stop Plus**; keep H+1 GateConcat.

---

### Phase E — Weekend experiments (August 27–28, 2026) **COMPLETE**

Queued via `weekend_queue.sh` after OCR capacity job. All jobs finished Fri Aug 28 ~01:08.

#### E.1 OCR capacity control @ 70% (v3: LoRA / GateConcat / Frozen)

| Method | Noisy macro | Clean macro | vs LoRA (noisy) | vs Frozen (noisy) |
|--------|-------------|-------------|-----------------|-------------------|
| LoRA | 0.731 | 0.733 | — | — |
| GateConcatFrozen (g≡0.5) | 0.739 | 0.743 | +0.007 (n.s.) | — |
| **GateConcat** | **0.754** | **0.747** | **+0.023** (*p*=**0.002**) | +0.015 (*p*=0.21) |

Per-seed Concat−Frozen noisy macro: `[+0.001, +0.026, +0.050, −0.010, +0.010]`. High Frozen variance (seed 2024: 0.704) inflates the capacity comparison.

#### E.2 Substitution rate sweep with Frozen (v4)

| Rate | LoRA noisy | Concat noisy | Frozen noisy | Concat−LoRA | Concat−Frozen |
|------|------------|--------------|--------------|-------------|---------------|
| 0% | 0.751 | 0.751 | 0.749 | ~0 | +0.000 |
| 30% | 0.746 | 0.745 | 0.742 | ~0 | +0.003 |
| 50% | 0.730 | 0.736 | 0.735 | **+0.006** | +0.001 |
| **70%** | **0.731** | **0.746** | **0.734** | **+0.015** (*p*=**0.003**) | +0.013 (*p*=0.34) |

**Pattern:** Concat advantage over Frozen appears only at **high** corruption; at 50% Concat and Frozen both beat LoRA similarly.

#### E.3 Other text corruptions @ 70% (v4: LoRA vs GateConcat only)

| Corruption | LoRA noisy | Concat noisy | Concat−LoRA | Paired *t* *p* |
|------------|------------|--------------|-------------|----------------|
| Insertion | 0.737 | 0.745 | +0.007 | 0.13 (n.s.) |
| Deletion | 0.730 | 0.750 | +0.020 | 0.07 (borderline) |

Insertion shows the same directional text-noise benefit but weaker significance; clean macro dips slightly on insertion (−0.6 pp).

#### E.4 Gate analysis — OCR @ 70%, seed 42

| Signal | Correlation with gate *g* |
|--------|---------------------------|
| Token corrupted (yes/no) | **−0.025** (≈ none) |
| ‖Δ‖ (LoRA–base disagreement) | −0.19 |
| Bbox area | −0.27 |
| Token position | −0.26 |
| Prediction correct | −0.04 |

Gate mean: clean 0.37 vs corrupted 0.35 (negligible). Gate varies strongly **by entity label** (e.g. label 1: 0.82, label 5: 0.07) — salience/label routing, not corruption detection. Acc@high-gate (0.33) ≈ Acc@low-gate (0.35).

**Artifacts:** `gatelora_v3_ocr07_capacity_*`, `gatelora_v4_sub_sweep_with_frozen_*`, `gatelora_v4_insert07_*`, `gatelora_v4_delete07_*`, `gate_analysis_v4_ocr/`, `weekend_queue.log`

---

## 4. Results and discussion (weekend verdict)

### 4.1 Main empirical findings

1. **GateConcat reliably beats LoRA under severe text noise.** Paired *t*-tests (n=5 seeds) are significant for substitution @70% (*p*=0.003) and OCR @70% (*p*=0.002). All five seeds favor Concat on OCR noisy macro. Deletion @70% is directionally consistent (+2.0 pp, *p*=0.07); insertion @70% is weaker (+0.7 pp, *p*=0.13).

2. **The gain is text-specific, not multimodal.** Box-swap @50% shows no benefit (Concat−LoRA = −0.2 pp). The method does not address layout perturbations.

3. **The gate is not a corruption detector.** Gate–corruption AUC ≈ 0.52 across settings. Supervised aux (v3) achieves AUC ≈ 0.91 but does not improve F1. Token-level gate analysis confirms near-zero correlation with corruption (−0.025) while correlating with label identity and spatial features.

4. **Capacity partially explains the gain — learned gating is not proven.** Frozen (constant g≡0.5, same H+1 architecture) also beats LoRA on OCR (+0.7 pp) and substitution @50% (+0.5 pp). GateConcat beats Frozen directionally at high severity (+1.5 pp OCR, +1.3 pp sub@70%) but **neither comparison is significant** at n=5 (*p*=0.21 and *p*=0.34). We cannot claim the **learned** gate is responsible for the improvement over a capacity-matched constant feature.

5. **Severity matters.** Substitution sweep shows ~zero gain at 0–30% corruption, emerging at 50% and strongest at 70%. This supports a “helps when text is heavily degraded” interpretation rather than a general clean-data boost.

### 4.2 What we can and cannot claim

**Can claim (with evidence):**
- Under matched LoRA on FUNSD, concatenating a token scalar to classifier input improves noisy F1 under high-severity synthetic **text** corruptions.
- The improvement is **not** from detecting corruption labels.
- Supervised noise routing (SALLoRA, aux gates, residual routers) fails or hurts.
- Extra capacity (Frozen) captures part of the benefit; ScalarConcat (v3) fails — not “any extra dim,” but H+1 with a meaningful scalar pathway.

**Cannot claim:**
- “Noise-aware” or “uncertainty-aware” PEFT.
- Learned gating significantly outperforms a frozen capacity control.
- General document robustness (layout noise negative).
- Main-track method contribution without a second dataset and stronger capacity ablation.

### 4.3 Interpretation

GateConcat’s benefit is best understood as a **small classifier-head augmentation** under text degradation: an extra token feature — whether learned or constant — gives the linear head slightly more expressivity when LoRA-adapted representations are noisy. The learned gate may specialize that feature (label/bbox/Δ patterns in analysis), but with current sample size we cannot separate that from H+1 capacity. The negative results (SALLoRA, aux, routing, layout) are as important as the positive ones: they show that **synthetic noise detection and NER robustness decouple**, and that PEFT robustness claims need capacity-matched controls.

### 4.4 Venue and framing recommendation

**Target:** workshop or Findings-style **diagnostic** paper.

**Working title direction:** *“When does PEFT help under synthetic document noise? A controlled negative result for noise-aware gating and a narrow positive for classifier capacity.”*

**One-sentence takeaway for the community:**
> Modest PEFT robustness under synthetic text noise can arise from classifier capacity rather than learned corruption routing — noise-aware gating claims should be validated with capacity-matched controls.

**Stop:** new architecture variants (Plus, aux, v5), CORD/SROIE until labels fixed, extra box-swap runs.

**Optional:** 10 seeds for OCR Concat vs Frozen if a significant capacity separation is required — likely marginal given Frozen variance.

---

## 5. Why GateConcat “overperforms” (what we know vs don’t)

### What we can say (evidence-based)

1. **Empirical win under text noise:** Concatenating token scalar `g` to `h_LoRA` improves noisy macro F1 for substitution, OCR, insertion, and deletion at high severity (significant vs LoRA on sub and OCR).
2. **Not noise detection:** Gate–corruption AUC ≈ **0.52**; token analysis corr(g, corrupted) ≈ **−0.025**.
3. **Supervision on corruption is the wrong objective:** Aux drives AUC → ~0.9 but F1 stays at LoRA level.
4. **Scope is text, not layout:** Box-swap shows no gain.
5. **Severity trend:** Gain ~0 at low noise, grows to ~+1.5 pp at 70% substitution.
6. **Capacity matters:** Frozen beats LoRA on OCR (+0.7 pp); Concat vs Frozen not significant.

### Plausible mechanisms (hypotheses — not proven)

| Hypothesis | Status |
|------------|--------|
| `g` is an implicit **token salience / label-routing** feature | **Supported directionally** — gate varies by entity label; weak corr with corruption |
| Extra classifier capacity (H+1) alone explains part of OCR gain | **Partially supported** — Frozen > LoRA; Concat > Frozen n.s. |
| Gate uses LoRA–base disagreement without naming “corruption” | Possible (corr with ‖Δ‖ = −0.19); not the primary story |

### What we must **not** claim

- “Uncertainty-aware / noise-aware / OCR-robust PEFT” as the main story  
- “Learned gate significantly beats frozen capacity control” (*p* > 0.2 on OCR and sub@70%)  
- Main-conference-ready general method without 2nd dataset  

---

## 6. Failed ideas (keep in the report — they are results)

| Idea | Outcome |
|------|---------|
| SALLoRA quality + mask + layer scalars | Negative vs LoRA |
| Residual routing without aux | No reliable win |
| Hard/soft aux on residual or concat | Detects noise, hurts or fails to help NER |
| Gate as CE reweighting | Collapsed / unusable |
| GateConcatPlus + ‖Δ‖ to classifier | No noisy-macro win over Concat |
| SROIE ranking | Saturated; ignore for method claims |

---

## 7. Current decision state

| Item | State |
|------|--------|
| Champion method | **GateConcat (v3)** — empirical only, not proven over Frozen |
| Architecture upgrades (Plus, aux, residual) | **Stopped** |
| OCR capacity control | **Done** — Frozen > LoRA; Concat > Frozen n.s. |
| Sub sweep + Frozen | **Done** — severity trend confirmed |
| Insert / delete @70% | **Done** — directional text transfer |
| Gate analysis (OCR seed 42) | **Done** — not corruption-aware |
| CORD / SROIE | **Blocked** until label mapping fixed |
| Paper path | **Diagnostic / Findings / workshop** — start drafting |

**Decision (Aug 30):** Capacity is **not fully ruled out**. Write up honestly; do not claim learned noise-aware gating.

---

## 8. Fair claim paragraph (abstract draft)

> Under a matched LayoutLMv3-LoRA protocol on FUNSD, hand-crafted SALLoRA and residual/auxiliary noise-aware gates do not improve NER. A simple alternative, GateConcat — concatenating a learned token gate feature to LoRA hidden states before classification, trained with standard CE only — improves noisy macro F1 by about **1.5 pp** under 70% character substitution (*p*=0.003) and **2.3 pp** under 70% OCR confusion (*p*=0.002), with similar directional gains on insertion and deletion. The gate does not track synthetic corruption (AUC ≈ 0.52; analysis corr ≈ 0). A constant-gate capacity control (Frozen) also beats LoRA on OCR (+0.7 pp); GateConcat exceeds Frozen directionally (+1.5 pp) but not significantly (*p*=0.21). Gains do not transfer to bbox swap. We conclude that modest text-corruption robustness can arise from classifier-head capacity rather than learned corruption routing, and recommend capacity-matched controls for future PEFT robustness claims.

---

## 9. Artifact map

| Stage | Code | Key results |
|-------|------|-------------|
| SALLoRA | `legacy/sallora.py` | `CONTROLLED_BENCHMARK_SALLORA.md` |
| Early Gate/Router | `legacy/gatelora.py` | `gatelora_funsd_results_*`, `router_compare_*` |
| Variant search | `legacy/gatelora_v2.py` | `gatelora_v2_funsd07_*`, concat sweep |
| Capacity @ sub70% | `gatelora_v3.py` | `gatelora_v3_funsd07_*`, `GATECONCAT_V3_OUTCOME.md` |
| Transfer + weekend | `gatelora_v4.py`, `scripts/weekend_queue.sh` | `gatelora_v4_ocr07_*`, `gatelora_v4_box05_*`, `gatelora_v3_ocr07_capacity_*`, `gatelora_v4_sub_sweep_with_frozen_*`, `gatelora_v4_insert07_*`, `gatelora_v4_delete07_*` |
| Analysis | `analyze_gate_v4.py`, `analyze_significance.py` | `gate_analysis_v4_ocr/` |

---

## 10. Next actions (short)

1. **Draft paper** — diagnostic framing per §4.4; use §8 as abstract seed.  
2. **Do not** run new architecture variants or CORD until labels fixed.  
3. **Optional:** 10 seeds OCR Concat vs Frozen only if a significant capacity separation is required for submission.  
4. Use this report + `GATECONCAT_V3_OUTCOME.md` for go/no-go on submission target.

---

*This document is for progress reporting. Numbers are from controlled runs in this repo; discard smoke / undertrained pilots in `archive/`.*
