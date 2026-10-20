# GateConcat: Diagnostic Study of Noise-Aware PEFT for Document NER

**Author:** MD. Rafi Imam  
**Status:** Research code + controlled experiment artifacts (diagnostic / Findings-style study)  
**Primary dataset:** [FUNSD](https://guillaumejaume.github.io/FUNSD/) via `nielsr/funsd-layoutlmv3`  
**Backbone:** LayoutLMv3-base + LoRA

This repository contains a **controlled diagnostic study** of parameter-efficient fine-tuning (PEFT) under synthetic document noise. It starts from a failed hand-crafted robust adapter (**SALLoRA**), explores routing / supervised gating variants, and arrives at a simple method (**GateConcat**) that yields a **modest, text-specific** gain over matched LoRA — with capacity controls and honest limits.

> **Fair takeaway:** modest PEFT robustness under synthetic *text* noise can arise from classifier-head capacity rather than learned corruption routing. Noise-aware gating claims should be validated with capacity-matched controls.

---

## Key results (FUNSD, 5 seeds, matched protocol)

| Setting | GateConcat − LoRA (noisy macro F1) | Notes |
|--------|-------------------------------------|--------|
| Substitution @ 70% | **+1.5 pp** (*p* ≈ 0.003) | Significant vs LoRA |
| OCR confusion @ 70% | **+2.3 pp** (*p* ≈ 0.002) | Transfers across text noise |
| Insertion / deletion @ 70% | +0.7 / +2.0 pp | Directional; weaker significance |
| Box-swap @ 50% | **−0.2 pp** | No layout transfer |
| Gate vs corruption labels | AUC ≈ **0.52** | Not noise detection |
| Soft aux (corruption supervised) | AUC ≈ **0.91**, F1 ≈ LoRA | Detection ≠ task gain |
| Frozen gate (g ≡ 0.5) on OCR @ 70% | Concat − Frozen ≈ +1.5 pp (*p* ≈ 0.21) | Capacity not fully ruled out |

Full narrative: [`docs/PROGRESS_REPORT_SALLORA_TO_GATECONCAT.md`](docs/PROGRESS_REPORT_SALLORA_TO_GATECONCAT.md)

---

## Method: GateConcat

```text
h_LoRA, h_base = encode(adapters on / off)
Δ = h_LoRA − h_base
g = σ(MLP([h_base, Δ, ‖Δ‖, bbox]))
logits = Classifier( concat(h_LoRA, g) )   # H+1 features
# Train: unweighted CE only — NO corruption auxiliary loss
```

**Controls used:**
- `gate_concat_frozen` — same H+1 head, constant `g = 0.5`
- `scalar_concat` — tiny learned scalar (fails)
- `gate_concat_aux` — soft corruption aux (high AUC, no F1 gain)

---

## Locked protocol (fair comparison)

| Setting | Value |
|--------|--------|
| Backbone | `microsoft/layoutlmv3-base` |
| LoRA | r=16, α=32, query/value, dropout=0.1 |
| Optimizer | AdamW, lr=`2e-4`, WD=`0.01`, linear warmup 10% |
| Epochs / batch | 15 / 4 |
| Seeds | `42, 123, 2024, 456, 789` |
| Eval | Clean test **and** noisy test; weighted + macro F1 |
| Task loss | Unweighted CE (GateConcat line) |

No per-method hyperparameter privileges.

---

## Repository layout

```text
.
├── gateconcat/                # Shared package (single source of truth)
│   ├── config.py              #   Locked protocol Config
│   ├── corruption.py          #   Synthetic noise transforms (torch-free)
│   ├── data.py                #   FUNSD loading + noisy dataset
│   ├── models.py              #   LoRA shell + GateConcat family
│   ├── training.py            #   Train / evaluate loop
│   ├── losses.py              #   CE + gate-target helpers
│   ├── significance.py        #   Paired t-test / Wilcoxon
│   ├── seed.py                #   Deterministic seeding
│   └── json_utils.py          #   Strict-JSON writing (NaN -> null)
├── gatelora_v3.py             # Experiment: capacity + aux controls @ sub 70%
├── gatelora_v4.py             # Experiment: transfer (OCR / box / insert / delete)
├── legacy/                    # Frozen Phase A/B monoliths (sallora, v1, v2)
├── analyze_significance.py    # CLI wrapper over gateconcat.significance
├── analyze_gate_v4.py         # Token-level gate correlation dump
├── tests/                     # Unit tests (pytest / unittest)
├── scripts/weekend_queue.sh   # Sequential experiment queue
├── docs/                      # Progress report + outcome notes + REFACTOR.md
└── results/                   # CSV / JSON summaries + per-seed tables
```

Refactored 2026-10-20 from five duplicated monoliths into the package
above — protocol and published results unchanged; see
[`docs/REFACTOR.md`](docs/REFACTOR.md) for the mapping and verification.

**Not included (intentionally):** personal emails, training logs, smoke pilots, `__pycache__`, large token dumps.

---

## Setup

```bash
# Python 3.10+ recommended; CUDA GPU strongly preferred
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# or, for the package + test deps:
# pip install -e ".[test]"
```

Hugging Face will download LayoutLMv3 and FUNSD on first run.

---

## Reproduce main experiments

```bash
# Substitution @ 70% with capacity controls (v3)
python -u gatelora_v3.py \
  --dataset funsd \
  --corruption-type substitution \
  --corruption-rate 0.7 \
  --methods lora gate_concat gate_concat_frozen \
  --out results/run_sub07.csv

# OCR @ 70% transfer (v4)
python -u gatelora_v4.py \
  --dataset funsd \
  --corruption-type ocr_confusion \
  --corruption-rate 0.7 \
  --methods lora gate_concat gate_concat_frozen \
  --out results/run_ocr07.csv

# Rate sweep + frozen
python -u gatelora_v4.py \
  --dataset funsd --corruption-type substitution \
  --corruption-rates 0.0 0.3 0.5 0.7 \
  --methods lora gate_concat gate_concat_frozen \
  --out results/run_sub_sweep.csv

# Significance
python analyze_significance.py \
  --per-seed results/gatelora_v3_ocr07_capacity_per_seed.csv \
  --method-a lora --method-b gate_concat

# Gate analysis (OCR, seed 42)
python -u analyze_gate_v4.py \
  --method gate_concat \
  --corruption-type ocr_confusion --corruption-rate 0.7 \
  --seed 42 --out-dir results/gate_analysis
```

Each full 5-seed run is multi-hour on a single consumer GPU; do not parallelize heavy jobs on one device.

Run the unit tests (light tests need only numpy/pandas/scipy):

```bash
python -m pytest tests/ -q
```

---

## What this is / is not

| Is | Is not |
|----|--------|
| Controlled negative results for hand-crafted / aux “noise-aware” PEFT | A claim of general OCR-robust PEFT |
| Modest text-corruption gain for GateConcat vs matched LoRA | Proof that the *learned* gate beats capacity |
| Capacity + mechanism diagnostics | Main-track NeurIPS/ICML method paper alone |

Recommended framing: **workshop / Findings diagnostic paper**.

---

## Citation / sharing

If you use this code or numbers, please cite the repository and credit **MD. Rafi Imam**.  
For PhD application materials: this repo is intended as a transparent research artifact accompanying the progress report under `docs/`.

---

## License

MIT — see [`LICENSE`](LICENSE).
