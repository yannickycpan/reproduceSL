#!/usr/bin/env bash
# Run on a CPU compute node after the candidate campaign; cached evaluation resumes.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
FULLTD_DATA_DIR="${FULLTD_DATA_DIR:-$HOME/scratch/fulltd/data}"
FULLTD_RESULTS_DIR="${FULLTD_RESULTS_DIR:-$HOME/scratch/fulltd/results}"
FULLTD_OUTPUT_DIR="${FULLTD_OUTPUT_DIR:-$HOME/scratch/fulltd/outputs}"
VENV="${FULLTD_ENV:-$HOME/scratch/fulltd/.venv-fulltd}"
DATASET="${FULLTD_DATASET:-all}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
"$VENV/bin/python" -m fulltd.analysis select --manifest configs/fulltd/manifest.json \
  --results-dir "$FULLTD_RESULTS_DIR" --output-dir "$FULLTD_OUTPUT_DIR" --dataset "$DATASET"
"$VENV/bin/python" -m fulltd.evaluate --manifest configs/fulltd/manifest.json \
  --results-dir "$FULLTD_RESULTS_DIR" --selection-dir "$FULLTD_OUTPUT_DIR" \
  --data-dir "$FULLTD_DATA_DIR" --device cpu --threads 1 --batch-size "${FULLTD_EVAL_BATCH:-128}"
"$VENV/bin/python" -m fulltd.analysis summarize --metrics "$FULLTD_OUTPUT_DIR/metrics.csv" \
  --table-dir "$FULLTD_OUTPUT_DIR" --figures-dir "$FULLTD_OUTPUT_DIR/figures"
