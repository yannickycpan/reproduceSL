#!/usr/bin/env bash
# Non-destructive, validation-locked FD001 batch-specific / equal-update reports.
# Usage: bash dscpuscripts/analyze_fd001_batches.sh check
#        bash dscpuscripts/analyze_fd001_batches.sh run
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/dscpuscripts/env.sh"
cd "$ROOT"
exec "$FULLTD_ENV/bin/python" -m dscpuscripts.fd001_batch_compare "$@"
