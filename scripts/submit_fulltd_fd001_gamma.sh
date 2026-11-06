#!/usr/bin/env bash
# Only the isolated 180-run FD001 Gamma campaign; no legacy run is resubmitted.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/fulltd_environment.sh"
cd "$ROOT"
exec "$FULLTD_ENV/bin/python" -m fd001_gamma.hpc "$@"
