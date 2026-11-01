#!/usr/bin/env bash
# Optional, separate data preparation and CPU check job; never submits training.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/fulltd_environment.sh"
cd "$ROOT"
exec "$FULLTD_ENV/bin/python" -m fulltd.hpc preflight "$@"
