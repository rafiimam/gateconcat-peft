#!/usr/bin/env bash
# Sequential experiment queue — ONE GPU at a time.
# Runs: substitution sweep + frozen → insertion@0.7 → deletion@0.7 → gate analysis.
#
# Usage (from repo root):
#   chmod +x scripts/weekend_queue.sh
#   nohup ./scripts/weekend_queue.sh > weekend_queue.log 2>&1 &

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"

mkdir -p results

echo "===== weekend_queue start $(date) ====="

echo "[$(date)] JOB 1/4: substitution sweep + frozen"
$PY -u gatelora_v4.py \
  --dataset funsd --corruption-type substitution \
  --corruption-rates 0.0 0.3 0.5 0.7 \
  --methods lora gate_concat gate_concat_frozen \
  --out results/gatelora_v4_sub_sweep_with_frozen.csv
echo "[$(date)] JOB 1 done"

echo "[$(date)] JOB 2/4: insertion @ 0.7"
$PY -u gatelora_v4.py \
  --dataset funsd --corruption-type insertion --corruption-rate 0.7 \
  --methods lora gate_concat \
  --out results/gatelora_v4_insert07.csv
echo "[$(date)] JOB 2 done"

echo "[$(date)] JOB 3/4: deletion @ 0.7"
$PY -u gatelora_v4.py \
  --dataset funsd --corruption-type deletion --corruption-rate 0.7 \
  --methods lora gate_concat \
  --out results/gatelora_v4_delete07.csv
echo "[$(date)] JOB 3 done"

echo "[$(date)] JOB 4/4: gate analysis OCR seed 42"
$PY -u analyze_gate_v4.py \
  --method gate_concat \
  --corruption-type ocr_confusion --corruption-rate 0.7 \
  --seed 42 --out-dir results/gate_analysis_v4_ocr
echo "[$(date)] JOB 4 done"

echo "===== weekend_queue finished $(date) ====="
