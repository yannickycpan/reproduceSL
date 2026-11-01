#!/usr/bin/env bash
# 330 candidates: 90/90/90/60 ranks; at most 4 nodes with the default concurrency.
# 12 hours is an allocation limit, not an estimated campaign completion time.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/fulltd_environment.sh"
cd "$ROOT"
exec "$FULLTD_ENV/bin/python" -m fulltd.hpc --time 12:00:00 --concurrency 4 "$@" --dataset metr_la_fd001
