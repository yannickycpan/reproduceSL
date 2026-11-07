#!/usr/bin/env bash
# Direct Gamma analysis. Original MSE/MAE conventions plus explicit RMSE curve.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/dscpuscripts/env.sh"
cd "$ROOT"
PYTHON="$FULLTD_ENV/bin/python"
[[ -x "$PYTHON" ]] || { echo "Run bash dscpuscripts/setup_server.sh first." >&2; exit 1; }
MODE="${1:-run}"
[[ $# -le 1 ]] || { echo 'Usage: analyze_fd001_gamma.sh [run|check|plots|status]' >&2; exit 2; }
case "$MODE" in
  status) exec "$PYTHON" -m dscpuscripts.status_fd001_gamma ;;
  check) exec "$PYTHON" -m fd001_gamma.analysis check ;;
  run)
    "$PYTHON" -m fulltd.data verify --dataset fd001 --data-dir "$FULLTD_DATA_DIR"
    "$PYTHON" -m fd001_gamma.analysis run
    exec "$PYTHON" -m dscpuscripts.fd001_report gamma
    ;;
  plots)
    "$PYTHON" -m fd001_gamma.analysis plots
    exec "$PYTHON" -m dscpuscripts.fd001_report gamma
    ;;
  *) echo 'Use run, check, plots or status.' >&2; exit 2 ;;
esac
