#!/usr/bin/env bash
# Read-only status report: no scheduler, training, refit, or test evaluation.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/dscpuscripts/env.sh"
cd "$ROOT"
PYTHON="$FULLTD_ENV/bin/python"
[[ -x "$PYTHON" ]] || { echo "Run bash dscpuscripts/setup_server.sh first." >&2; exit 1; }
exec "$PYTHON" -m dscpuscripts.status_fd001_gamma "$@"
