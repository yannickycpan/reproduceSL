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

echo "== 1/5 Run synthetic regression tests (no training) =="
"$FULLTD_ENV/bin/python" -m pytest -q \
  dscpuscripts/test_fd001_batch_compare.py \
  dscpuscripts/test_fd001_plotting.py

echo "== 2/5 Audit 600 candidates and lock validation-only choices =="
bash dscpuscripts/analyze_fd001_batches.sh check

echo "== 3/5 Evaluate selected final and matched-update checkpoints =="
bash dscpuscripts/analyze_fd001_batches.sh run

echo "== 4/5 Display eight-row table (mean +/- SE) =="
cat "$FULLTD_RESULTS_ROOT/fd001_outputs/batch_comparison/comparison_table.md"

echo "== 5/5 Replot original selected curves (only y-axis zoomed) =="
bash dscpuscripts/analyze_standard.sh fd001 plots
echo "Done. Reports in $FULLTD_RESULTS_ROOT/fd001_outputs/batch_comparison/"
