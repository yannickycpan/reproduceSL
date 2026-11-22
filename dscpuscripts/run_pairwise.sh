#!/usr/bin/env bash
# Resumable new Pairwise TD runs (MSE and Gaussian NLL use existing launchers).
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/dscpuscripts/env.sh"
cd "$ROOT"
if [[ ! -x "$FULLTD_ENV/bin/python" ]]; then
  echo "Missing Python environment: $FULLTD_ENV/bin/python" >&2
  exit 1
fi
exec "$FULLTD_ENV/bin/python" -m dscpuscripts.pairwise_runner "$@"
