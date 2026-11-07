#!/usr/bin/env bash
# Consolidated 320-run Gamma audit: original 180 + extension 140, no refit.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/dscpuscripts/env.sh"
cd "$ROOT"
PYTHON="$FULLTD_ENV/bin/python"
[[ -x "$PYTHON" ]] || { echo "Run bash dscpuscripts/setup_server.sh first." >&2; exit 1; }
MODE="${1:-run}"
if [[ "$MODE" == "run" ]]; then
  # After moving the flat run directories, outputs/ may not exist yet.
  # Regenerate the original Gamma selection and endpoint metrics from the
  # actual completed checkpoints before reusing them in the extension.
  # This runs NO optimizer step, refit, or Slurm job.
  bash "$ROOT/dscpuscripts/analyze_fd001_gamma.sh" run
fi
exec "$PYTHON" -m fd001_gamma_ext.analysis "$MODE"
