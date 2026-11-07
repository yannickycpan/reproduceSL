#!/usr/bin/env bash
# Canonical six-method FD001 comparison on any machine; no Slurm or refits.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/dscpuscripts/env.sh"
cd "$ROOT"
exec "$FULLTD_ENV/bin/python" -m dscpuscripts.compare_fd001_extension "${1:-run}"
