#!/usr/bin/env bash
# One-server FD001 Gamma reporting: no Slurm, no refit, no new model training.
# Usage: bash dscpuscripts/analyze_fd001_gamma.sh [run|check|plots|status]
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/dscpuscripts/env.sh"
cd "$ROOT"
PYTHON="$FULLTD_ENV/bin/python"
[[ -x "$PYTHON" ]] || { echo "Run bash dscpuscripts/setup_server.sh first." >&2; exit 1; }

MODE="${1:-run}"
if [[ $# -gt 1 ]]; then
  echo "Usage: bash dscpuscripts/analyze_fd001_gamma.sh [run|check|plots|status]" >&2
  exit 2
fi
case "$MODE" in
  status)
    exec "$PYTHON" -m dscpuscripts.status_fd001_gamma
    ;;
  check)
    exec "$PYTHON" -m fd001_gamma.analysis check
    ;;
  run)
    echo "Validating processed FD001 data (read-only)..."
    "$PYTHON" -m fulltd.data verify --dataset fd001 --data-dir "$FULLTD_DATA_DIR"
    echo "Auditing all 180 FD001 Gamma candidates..."
    "$PYTHON" -m fd001_gamma.analysis check
    echo "Validation selection, epoch-100 evaluation and vector plots; NO REFIT."
    exec "$PYTHON" -m fd001_gamma.analysis run
    ;;
  plots)
    echo "Regenerating Gamma plots from saved metrics only; NO REFIT."
    exec "$PYTHON" -m fd001_gamma.analysis plots
    ;;
  *)
    echo "Unknown mode '$MODE'. Use: run, check, plots, status." >&2
    exit 2
    ;;
esac
