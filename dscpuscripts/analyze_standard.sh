#!/usr/bin/env bash
# Canonical baseline/Electricity/METR-LA analysis on the server, no refit.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/dscpuscripts/env.sh"
cd "$ROOT"
"$FULLTD_ENV/bin/python" -m dscpuscripts.standard_analysis "$@"
if [[ "${1:-}" == fd001_baselines && "${2:-run}" != check ]]; then
  "$FULLTD_ENV/bin/python" -m dscpuscripts.fd001_report baselines
fi
