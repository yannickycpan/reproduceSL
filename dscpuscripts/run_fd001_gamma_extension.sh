#!/usr/bin/env bash
# Train ONLY the 140 new FD001 Gamma extension runs (new alpha, new pairwise head).
# Reuses frozen original trainer, and never touches the 180 original checkpoints.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/dscpuscripts/env.sh"
cd "$ROOT"
PYTHON="$FULLTD_ENV/bin/python"
[[ -x "$PYTHON" ]] || { echo "Run bash dscpuscripts/setup_server.sh first." >&2; exit 1; }
exec "$PYTHON" -m dscpuscripts.direct_runner --campaign extension "$@"
