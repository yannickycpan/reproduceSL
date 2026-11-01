#!/usr/bin/env bash
# 300 candidates: 90/90/90/30 ranks; at most 3 nodes with the default concurrency.
# 72 hours is an allocation limit, not an estimated campaign completion time.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/fulltd_environment.sh"
cd "$ROOT"
exec "$FULLTD_ENV/bin/python" -m fulltd.hpc --time 3-00:00:00 --concurrency 3 "$@" --dataset electricity
