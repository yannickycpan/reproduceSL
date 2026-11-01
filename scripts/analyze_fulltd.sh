#!/usr/bin/env bash
# Run on a CPU compute node after the candidate campaign; cached evaluation resumes.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/fulltd_environment.sh"
cd "$ROOT"
FULLTD_OUTPUT_DIR="${FULLTD_OUTPUT_DIR:-$HOME/scratch/fulltd/outputs}"
VENV="$FULLTD_ENV"
DATASET="${FULLTD_DATASET:-all}"
"$VENV/bin/python" -m fulltd.analysis select --manifest configs/fulltd/manifest.json \
  --results-dir "$FULLTD_RESULTS_DIR" --output-dir "$FULLTD_OUTPUT_DIR" --dataset "$DATASET"
"$VENV/bin/python" -m fulltd.evaluate --manifest configs/fulltd/manifest.json \
  --results-dir "$FULLTD_RESULTS_DIR" --selection-dir "$FULLTD_OUTPUT_DIR" \
  --data-dir "$FULLTD_DATA_DIR" --device cpu --threads 1 --batch-size "${FULLTD_EVAL_BATCH:-128}"
"$VENV/bin/python" -m fulltd.analysis summarize --metrics "$FULLTD_OUTPUT_DIR/metrics.csv" \
  --table-dir "$FULLTD_OUTPUT_DIR" --figures-dir "$FULLTD_OUTPUT_DIR/figures"
