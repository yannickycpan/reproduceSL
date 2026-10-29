#!/usr/bin/env bash
# Run on a compute node after candidate arrays finish; can resume cached evaluation.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${FULLTD_DATA_DIR:?Set FULLTD_DATA_DIR}"
: "${FULLTD_RESULTS_DIR:?Set FULLTD_RESULTS_DIR}"
: "${FULLTD_OUTPUT_DIR:?Set FULLTD_OUTPUT_DIR}"
VENV="${FULLTD_ENV:-$ROOT/.venv-fulltd}"
DATASET="${FULLTD_DATASET:-all}"
"$VENV/bin/python" -m fulltd.analysis select --manifest configs/fulltd/manifest.json \
  --results-dir "$FULLTD_RESULTS_DIR" --output-dir "$FULLTD_OUTPUT_DIR" --dataset "$DATASET"
"$VENV/bin/python" -m fulltd.evaluate --manifest configs/fulltd/manifest.json \
  --results-dir "$FULLTD_RESULTS_DIR" --selection-dir "$FULLTD_OUTPUT_DIR" \
  --data-dir "$FULLTD_DATA_DIR" --device "${FULLTD_DEVICE:-cuda}" --batch-size "${FULLTD_EVAL_BATCH:-128}"
"$VENV/bin/python" -m fulltd.analysis summarize --metrics "$FULLTD_OUTPUT_DIR/metrics.csv" \
  --table-dir "$FULLTD_OUTPUT_DIR" --figures-dir "$FULLTD_OUTPUT_DIR/figures"
