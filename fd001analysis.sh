#!/usr/bin/env bash
# Verify and report FD001 results without changing training or HP selection.
# Run from any directory: bash ~/reproduceSL/fd001analysis.sh
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$ROOT"
source "$ROOT/dscpuscripts/env.sh"
if [[ ! -x "$FULLTD_ENV/bin/python" ]]; then
  echo "ERROR: FD001 Python environment missing: $FULLTD_ENV/bin/python" >&2
  exit 1
fi

echo "== 1/6 Run synthetic regression tests (no training) =="
"$FULLTD_ENV/bin/python" -m pytest -q \
  dscpuscripts/test_fd001_batch_compare.py \
  dscpuscripts/test_fd001_plotting.py

echo "== 2/6 Audit 600 candidates and lock validation-only choices =="
bash dscpuscripts/analyze_fd001_batches.sh check

echo "== 3/6 Evaluate selected final and matched-update checkpoints =="
bash dscpuscripts/analyze_fd001_batches.sh run

echo "== 4/6 Display eight-row table (mean +/- SE) =="
cat "$FULLTD_RESULTS_ROOT/fd001_outputs/batch_comparison/comparison_table.md"

echo "== 5/6 Replot original selected curves (only y-axis zoomed) =="
bash dscpuscripts/analyze_standard.sh fd001 plots
echo "== 6/6 Create B256-only compute-normalized learning curves (proxy, not wall-clock) =="
"$FULLTD_ENV/bin/python" -m dscpuscripts.fd001_compute_curves
echo "Done. Reports in $FULLTD_RESULTS_ROOT/fd001_outputs/batch_comparison/"
echo "Compute-proxy plots in $FULLTD_RESULTS_ROOT/fd001_outputs/figures/compute_proxy/"
