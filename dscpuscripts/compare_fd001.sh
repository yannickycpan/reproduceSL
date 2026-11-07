#!/usr/bin/env bash
# Compare existing Gamma results and rerun baselines on THIS server.
# Optional --legacy PATH accepts older output folders or ZIPs; no transfer needed
# when the original baselines have been rerun with run_standard.sh.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/dscpuscripts/env.sh"
cd "$ROOT"
exec "$FULLTD_ENV/bin/python" -m dscpuscripts.fd001_report compare "$@"
