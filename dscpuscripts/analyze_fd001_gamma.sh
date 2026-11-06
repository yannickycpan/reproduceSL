#!/usr/bin/env bash
# Run selection, endpoint evaluation, and figures directly, with no scheduler.
# Requires all 180 results locally, including those imported from other servers.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/dscpuscripts/env.sh"
cd "$ROOT"
PYTHON="$FULLTD_ENV/bin/python"
[[ -x "$PYTHON" ]] || { echo "Run bash dscpuscripts/setup_server.sh first." >&2; exit 1; }
"$PYTHON" -m fulltd.data verify --dataset fd001 --data-dir "$FULLTD_DATA_DIR"
"$PYTHON" -m fd001_gamma.analysis check
exec "$PYTHON" -m fd001_gamma.analysis run
