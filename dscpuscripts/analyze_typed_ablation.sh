#!/usr/bin/env bash
# Analyze only typed TD ablations and read-only old MSE/typed TD references.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/dscpuscripts/env.sh"
cd "$ROOT"
if [[ $# -ne 2 ]]; then
  echo "Usage: bash dscpuscripts/analyze_typed_ablation.sh DATASET --select|--evaluate|--compare-select|--compare-evaluate|--gamma" >&2
  exit 2
fi
dataset="$1"
action="$2"
case "$action" in
  --select|--evaluate)
    exec "$FULLTD_ENV/bin/python" -m dscpuscripts.typed_ablation_analysis "$dataset" "$action"
    ;;
  --compare-select)
    exec "$FULLTD_ENV/bin/python" -m dscpuscripts.typed_ablation_comparison "$dataset" --select
    ;;
  --compare-evaluate)
    exec "$FULLTD_ENV/bin/python" -m dscpuscripts.typed_ablation_comparison "$dataset" --evaluate
    ;;
  --gamma)
    exec "$FULLTD_ENV/bin/python" -m dscpuscripts.typed_ablation_gamma_diagnostics "$dataset"
    ;;
  *)
    echo "Unknown analysis action: $action" >&2
    exit 2
    ;;
esac
