#!/usr/bin/env bash
# Reproduce FD001 selected tables + validation plots + held-out TEST compute curves.
# Run from any directory: bash ~/reproduceSL/fd001analysis.sh
# READS existing 600 training runs; never submits training or modifies checkpoints.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$ROOT"
source "$ROOT/dscpuscripts/env.sh"
if [[ ! -x "$FULLTD_ENV/bin/python" ]]; then
  echo "ERROR: FD001 Python environment missing: $FULLTD_ENV/bin/python" >&2
  exit 1
fi

echo "== 1/7 Regression tests (no training) =="
"$FULLTD_ENV/bin/python" -m pytest -q \
  dscpuscripts/test_fd001_batch_compare.py \
  dscpuscripts/test_fd001_plotting.py \
  dscpuscripts/test_fd001_compute_curves.py

echo "== 2/7 Reproduce canonical original selections, tables and figures =="
bash dscpuscripts/analyze_standard.sh fd001 run

echo "== 3/7 Audit and lock validation-only batch and matched-update selections =="
bash dscpuscripts/analyze_fd001_batches.sh check

echo "== 4/7 Evaluate selected full and matched-update endpoints =="
bash dscpuscripts/analyze_fd001_batches.sh run

echo "== 5/7 Display eight-row comparison table (mean +/- SE) =="
cat "$FULLTD_RESULTS_ROOT/fd001_outputs/batch_comparison/comparison_table.md"

echo "== 6/7 Regenerate original validation plots (full x, zoomed y) =="
bash dscpuscripts/analyze_standard.sh fd001 plots

echo "== 7/7 Evaluate TEST checkpoints at every saved step and plot vs compute proxy =="
"$FULLTD_ENV/bin/python" -m dscpuscripts.fd001_compute_curves

echo "Done. Original tables/figures: $FULLTD_RESULTS_ROOT/fd001_outputs/"
echo "Eight-row comparisons: $FULLTD_RESULTS_ROOT/fd001_outputs/batch_comparison/"
echo "Held-out test curves: $FULLTD_RESULTS_ROOT/fd001_outputs/figures/compute_proxy/"
echo "Held-out step-by-step metrics: $FULLTD_RESULTS_ROOT/fd001_outputs/test_compute/"
echo "NOTE: x axis of test curves is normalized compute, not measured seconds."
