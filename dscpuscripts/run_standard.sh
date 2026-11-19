#!/usr/bin/env bash
# Direct CPU execution of the current FD001, Electricity or METR-LA sweep.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/dscpuscripts/env.sh"
cd "$ROOT"
exec "$FULLTD_ENV/bin/python" -m dscpuscripts.standard_runner "$@"
