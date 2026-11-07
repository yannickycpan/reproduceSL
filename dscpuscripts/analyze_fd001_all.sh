#!/usr/bin/env bash
# One-command unified FD001 analysis: old completed MSE/NLL + new Gamma.
# No Slurm, no refit, no new training; only evaluates existing Gamma final nets
# when the Gamma endpoint cache is absent.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/dscpuscripts/env.sh"
cd "$ROOT"
PYTHON="$FULLTD_ENV/bin/python"
[[ -x "$PYTHON" ]] || { echo "Run bash dscpuscripts/setup_server.sh first." >&2; exit 1; }
MODE="run"
if [[ $# -gt 0 && ( "$1" == "plots" || "$1" == "run" ) ]]; then
  MODE="$1"
  shift
fi
if [[ $# -gt 0 && "$1" != "--legacy" ]]; then
  echo "Usage: bash dscpuscripts/analyze_fd001_all.sh [run|plots] [--legacy PATH]" >&2
  exit 2
fi
if [[ $# -ne 0 && $# -ne 2 ]]; then
  echo "Usage: bash dscpuscripts/analyze_fd001_all.sh [run|plots] [--legacy PATH]" >&2
  exit 2
fi
ARGS=()
if [[ $# -eq 2 ]]; then
  ARGS=(--legacy "$2")
fi
if [[ "$MODE" == "plots" ]]; then
  echo "Regenerating original Gamma plots from saved tables..."
  bash "$ROOT/dscpuscripts/analyze_fd001_gamma.sh" plots
  exec "$PYTHON" -m dscpuscripts.combined_fd001 plots
fi
echo "Checking and analyzing the 180 new Gamma runs..."
bash "$ROOT/dscpuscripts/analyze_fd001_gamma.sh" run
echo "Importing only the completed previous MSE and Gaussian-NLL baselines;"
echo "combining all five FD001 methods and regenerating comparison plots..."
exec "$PYTHON" -m dscpuscripts.combined_fd001 run "${ARGS[@]}"
