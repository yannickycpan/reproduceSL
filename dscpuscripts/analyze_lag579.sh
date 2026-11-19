#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/dscpuscripts/env.sh"
cd "$ROOT"
if [[ ! -x "$FULLTD_ENV/bin/python" ]]; then
  printf "Python environment missing: %s\n" "$FULLTD_ENV/bin/python" >&2
  exit 1
fi
exec "$FULLTD_ENV/bin/python" -m dscpuscripts.lag579_analysis "$@"
