#!/usr/bin/env bash
# Direct execution of original FD001 baselines, Electricity or METR-LA.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/dscpuscripts/env.sh"
cd "$ROOT"
exec "$FULLTD_ENV/bin/python" -m dscpuscripts.standard_runner "$@"
