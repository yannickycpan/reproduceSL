#!/usr/bin/env bash
# Consolidated 320-run Gamma audit: original 180 + extension 140, no refit.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/dscpuscripts/env.sh"
cd "$ROOT"
PYTHON="$FULLTD_ENV/bin/python"
[[ -x "$PYTHON" ]] || { echo "Run bash dscpuscripts/setup_server.sh first." >&2; exit 1; }
exec "$PYTHON" -m fd001_gamma_ext.analysis "${1:-run}"
